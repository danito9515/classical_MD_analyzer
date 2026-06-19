#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Tinker-GPU / Tinker-HP NPT trajectories using Tinker XYZ snapshots
as the primary time grid.

Version 2.0.0
-------------
This version does not require PDB files. It reads the complete Tinker XYZ
snapshot files written as, for example,

    system.001
    system.002
    system.003

or a multi-frame Tinker archive/XYZ file selected with --xyz-pattern.

For every XYZ frame, it reads:
    - atom count and atom types
    - bond connectivity
    - periodic cell parameters
    - cell volume

It can then calculate the instantaneous concentration of selected entities:

    c(t) = N_entity / [N_A * V(t) * 1.0e-27]

where V is in Angstrom^3 and c is in mol dm^-3.

Two selection modes are available:

1. Atom mode
   Count atoms carrying selected Tinker atom type(s).

       --conc-atom-type LiFSA:6

   This is appropriate when one selected atom corresponds to one formula unit.
   For example, if each LiFSA unit contains exactly one Li atom of type 6,
   the Li count equals the LiFSA formula-unit count.

2. Molecule mode
   Construct the molecular bond graph from the connectivity columns in the
   Tinker XYZ file and count connected components containing the selected
   atom type(s).

       --conc-molecule-type FSA:164
       --conc-molecule-type SN:71

   A molecule is counted only once, even when it contains two or more atoms
   of the selected type.

Expected numbered-snapshot directory structure
-----------------------------------------------

    .
    ├── LiFSAC2_poltype2.prm
    ├── LiFSA_SN_1_10_N90.xyz
    ├── T300K/
    │   ├── *_log_*.out
    │   ├── LiFSA_SN_1_10_N90.001
    │   ├── LiFSA_SN_1_10_N90.002
    │   └── ...
    ├── T300K_run_002/
    │   ├── *_log_*.out
    │   ├── LiFSA_SN_1_10_N90.001
    │   └── ...
    ├── T350K/
    └── analysis_result/

Basic example
-------------

python3 analyze_TinkerGpu_npt_joinruns_xyz_concentration_v2.0.0.py \
  --root . \
  --temps T300K T350K \
  --save-interval-ps 50 \
  --conc-atom-type LiFSA:6 \
  --conc-molecule-type FSA:164 \
  --conc-molecule-type SN:71 \
  --list-xyz-types \
  --force

If more than one numbered snapshot series exists in a run directory, select
one explicitly by its basename:

    --snapshot-prefix LiFSA_SN_1_10_N90

For a multi-frame Tinker ARC/XYZ trajectory instead of numbered snapshots:

python3 analyze_TinkerGpu_npt_joinruns_xyz_concentration_v2.0.0.py \
  --root . \
  --temps T300K \
  --xyz-pattern '*.arc' \
  --save-interval-ps 50 \
  --conc-atom-type LiFSA:6 \
  --conc-molecule-type FSA:164 \
  --force

Outputs
-------
    analysis_result/T300K_timeseries.csv
    analysis_result/T300K_concentration_timeseries.csv
    analysis_result/T300K_xyz_atom_type_inventory.csv
    analysis_result/all_temperatures_timeseries.csv
    analysis_result/all_concentrations_timeseries.csv
    analysis_result/concentration_selection_metadata.csv
    analysis_result/summary.csv
    analysis_result/png/*.png

Important physical point
------------------------
In a standard fixed-composition NPT simulation, particle counts are constant.
Therefore, concentration changes arise from the time-dependent NPT cell volume.
The script still verifies atom counts and topology consistency across frames.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Constants and basic utilities
# ============================================================

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)

AVOGADRO = 6.02214076e23
ANG3_TO_CM3 = 1.0e-24
ANG3_TO_DM3 = 1.0e-27


def to_float(x: str) -> float:
    """Convert a string to float, accepting Fortran D exponents."""
    return float(x.replace("D", "E").replace("d", "e"))


def extract_floats(line: str) -> List[float]:
    return [to_float(x) for x in FLOAT_RE.findall(line)]


def safe_label(text: str) -> str:
    """Convert a label into a stable CSV-column and filename token."""
    out = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text.strip())
    out = out.strip("_")
    return out or "selection"


def infer_temp_from_path(path: Path) -> float:
    """Infer temperature from a path containing text such as T300K."""
    m = re.search(r"T(\d+(?:\.\d+)?)K", str(path), flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return np.nan


def temperature_series_label(path: Path) -> str:
    """Map T300K_run_002 and similar names to T300K."""
    m = re.search(r"(T\d+(?:\.\d+)?K)", path.name, flags=re.IGNORECASE)
    if m:
        raw = m.group(1)
        return "T" + raw[1:-1] + "K"
    return path.name


def run_number_from_path(path: Path) -> int:
    """Infer continuation index from _run_002; base run is 1."""
    m = re.search(r"_run_(\d+)", path.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1


def sort_key_for_run_dir(path: Path):
    return (
        infer_temp_from_path(path),
        temperature_series_label(path),
        run_number_from_path(path),
        path.name,
    )


def estimate_time_step_ps(time_values, fallback=50.0) -> float:
    """Estimate a positive time spacing from an existing series."""
    try:
        vals = pd.to_numeric(pd.Series(time_values), errors="coerce").dropna().unique()
        vals = np.sort(vals)
        diffs = np.diff(vals)
        diffs = diffs[diffs > 1.0e-12]
        if len(diffs) > 0:
            return float(np.median(diffs))
    except Exception:
        pass
    return float(fallback)


def cell_volume(a, b, c, alpha, beta, gamma) -> float:
    """Compute triclinic cell volume from lengths in A and angles in degrees."""
    try:
        ar = math.radians(float(alpha))
        br = math.radians(float(beta))
        gr = math.radians(float(gamma))
        factor = (
            1.0
            - math.cos(ar) ** 2
            - math.cos(br) ** 2
            - math.cos(gr) ** 2
            + 2.0 * math.cos(ar) * math.cos(br) * math.cos(gr)
        )
        if factor < -1.0e-10:
            return np.nan
        return float(a) * float(b) * float(c) * math.sqrt(max(factor, 0.0))
    except Exception:
        return np.nan


def concentration_mol_dm3(n_entities: int, volume_A3: float) -> float:
    """Convert an entity count and volume in A^3 to mol dm^-3."""
    if n_entities < 0 or not np.isfinite(volume_A3) or volume_A3 <= 0.0:
        return np.nan
    return float(n_entities) / (AVOGADRO * float(volume_A3) * ANG3_TO_DM3)


def open_text(path: Path):
    """Open plain text or gzip-compressed text."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", errors="ignore")
    return path.open("r", errors="ignore")


# ============================================================
# Concentration selection definitions
# ============================================================

@dataclass(frozen=True)
class ConcentrationSelection:
    label: str
    token: str
    atom_types: Tuple[int, ...]
    count_mode: str          # atom or molecule
    molecule_match: str      # any or all

    @property
    def count_column(self) -> str:
        return f"count_{self.token}"

    @property
    def concentration_column(self) -> str:
        return f"concentration_{self.token}_mol_dm3"

    @property
    def cumulative_column(self) -> str:
        return f"{self.concentration_column}_cumavg"


def parse_selection_spec(
    spec: str,
    count_mode: str,
    molecule_match: str,
) -> ConcentrationSelection:
    """
    Parse LABEL:TYPE or LABEL:TYPE1,TYPE2.

    Example:
        LiFSA:6
        FSA:164,165
    """
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid concentration selection '{spec}'. Use LABEL:TYPE or LABEL:TYPE1,TYPE2."
        )

    label, type_text = spec.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Missing label in selection '{spec}'.")

    raw_types = [x.strip() for x in type_text.split(",") if x.strip()]
    if not raw_types:
        raise argparse.ArgumentTypeError(f"Missing atom type in selection '{spec}'.")

    try:
        atom_types = tuple(sorted(set(int(x) for x in raw_types)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Atom types must be integers in selection '{spec}'."
        ) from exc

    return ConcentrationSelection(
        label=label,
        token=safe_label(label),
        atom_types=atom_types,
        count_mode=count_mode,
        molecule_match=molecule_match,
    )


def build_selections(args) -> List[ConcentrationSelection]:
    selections: List[ConcentrationSelection] = []

    for spec in args.conc_atom_type or []:
        selections.append(
            parse_selection_spec(
                spec=spec,
                count_mode="atom",
                molecule_match="any",
            )
        )

    for spec in args.conc_molecule_type or []:
        selections.append(
            parse_selection_spec(
                spec=spec,
                count_mode="molecule",
                molecule_match=args.molecule_type_match,
            )
        )

    seen = {}
    for sel in selections:
        if sel.token in seen:
            raise ValueError(
                f"Duplicate or ambiguous concentration label '{sel.label}'. "
                f"It maps to the same token as '{seen[sel.token]}'."
            )
        seen[sel.token] = sel.label

    return selections


# ============================================================
# Temperature-directory discovery
# ============================================================


def discover_temperature_groups(root: Path, temps=None, join_continuations=True):
    """
    Discover TxxxK directories and group continuation runs.

    T300K + T300K_run_002 + T300K_run_003 -> one T300K series.
    """

    def is_temperature_dir(p: Path):
        return p.is_dir() and re.search(r"T\d+(?:\.\d+)?K", p.name, flags=re.IGNORECASE)

    if temps is None:
        candidate_dirs = sorted(
            [p for p in root.iterdir() if is_temperature_dir(p)],
            key=sort_key_for_run_dir,
        )
    else:
        requested = [root / t for t in temps]
        if join_continuations:
            labels = {temperature_series_label(p) for p in requested}
            candidate_dirs = []
            for label in labels:
                pat = re.compile(
                    rf"^{re.escape(label)}(?:_run_\d+)?$",
                    flags=re.IGNORECASE,
                )
                candidate_dirs.extend(
                    [p for p in root.iterdir() if p.is_dir() and pat.match(p.name)]
                )
            candidate_dirs.extend(requested)
            candidate_dirs = sorted(set(candidate_dirs), key=sort_key_for_run_dir)
        else:
            candidate_dirs = requested

    if not join_continuations:
        return [(p.name, [p]) for p in sorted(candidate_dirs, key=sort_key_for_run_dir)]

    groups: Dict[str, List[Path]] = {}
    for d in candidate_dirs:
        label = temperature_series_label(d)
        groups.setdefault(label, [])
        if d not in groups[label]:
            groups[label].append(d)

    grouped = []
    for label, dirs in groups.items():
        grouped.append((label, sorted(dirs, key=sort_key_for_run_dir)))

    grouped.sort(key=lambda x: (infer_temp_from_path(Path(x[0])), x[0]))
    return grouped


# ============================================================
# Tinker PRM and system mass
# ============================================================


def parse_atom_masses_from_prm(prm_file: Path) -> Dict[int, float]:
    """Parse Tinker atom-type masses from a parameter file."""
    masses: Dict[int, float] = {}

    with prm_file.open("r", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not re.match(r"^\s*atom\s+", line, flags=re.IGNORECASE):
                continue

            try:
                tokens = shlex.split(stripped)
            except Exception:
                tokens = stripped.split()

            if len(tokens) < 6:
                continue

            try:
                atom_type = int(tokens[1])
            except ValueError:
                continue

            numeric_tail: List[float] = []
            for tok in tokens[4:]:
                try:
                    numeric_tail.append(to_float(tok))
                except Exception:
                    pass

            # Normal Tinker atom record ends in atomic_number, mass, valence.
            if len(numeric_tail) >= 3:
                mass = numeric_tail[-2]
            elif len(numeric_tail) >= 2:
                mass = numeric_tail[-1]
            else:
                continue

            if mass > 0.0:
                masses[atom_type] = float(mass)

    return masses


# ============================================================
# Tinker XYZ / ARC parsing
# ============================================================

@dataclass
class TinkerAtom:
    atom_id: int
    atom_name: str
    atom_type: int
    bonds: Tuple[int, ...]


@dataclass
class TinkerXYZFrame:
    source_file: Path
    source_frame: int
    n_atoms: int
    title: str
    a_A: float
    b_A: float
    c_A: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    atoms: List[TinkerAtom]

    @property
    def volume_A3(self) -> float:
        return cell_volume(
            self.a_A,
            self.b_A,
            self.c_A,
            self.alpha_deg,
            self.beta_deg,
            self.gamma_deg,
        )


def _read_next_nonblank(handle) -> Optional[str]:
    for line in handle:
        if line.strip():
            return line
    return None


def parse_tinker_box_line(line: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Return six periodic-cell values when a line is a Tinker box line."""
    tokens = line.split()
    if len(tokens) < 6:
        return None
    try:
        vals = tuple(to_float(tok) for tok in tokens[:6])
    except Exception:
        return None
    a, b, c, alpha, beta, gamma = vals
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return None
    return a, b, c, alpha, beta, gamma


def parse_tinker_atom_line(line: str, source_file: Path, frame_no: int) -> TinkerAtom:
    """
    Parse a standard Tinker XYZ atom line:

        id name x y z atom_type bonded_atom_ids...
    """
    tokens = line.split()
    if len(tokens) < 6:
        raise ValueError(
            f"Malformed Tinker atom line in {source_file}, frame {frame_no}: {line.rstrip()}"
        )

    try:
        atom_id = int(tokens[0])
        # coordinates are parsed to validate the line, even though they are not stored
        to_float(tokens[2])
        to_float(tokens[3])
        to_float(tokens[4])
        atom_type = int(tokens[5])
    except Exception as exc:
        raise ValueError(
            f"Malformed Tinker atom line in {source_file}, frame {frame_no}: {line.rstrip()}"
        ) from exc

    bonds: List[int] = []
    for tok in tokens[6:]:
        try:
            bonded_id = int(tok)
        except ValueError:
            break
        if bonded_id > 0:
            bonds.append(bonded_id)

    return TinkerAtom(
        atom_id=atom_id,
        atom_name=tokens[1],
        atom_type=atom_type,
        bonds=tuple(bonds),
    )


def iter_tinker_xyz_frames(path: Path) -> Iterator[TinkerXYZFrame]:
    """
    Stream one or more frames from a Tinker XYZ/ARC/snapshot file.

    Periodic frames are expected to contain a six-number box line immediately
    after the atom-count/title line. A nonperiodic single-frame XYZ can still be
    parsed, but its cell values will be NaN and concentration cannot be computed.
    """
    frame_no = 0

    with open_text(path) as handle:
        while True:
            header = _read_next_nonblank(handle)
            if header is None:
                break

            header_tokens = header.split()
            try:
                n_atoms = int(header_tokens[0])
            except Exception as exc:
                raise ValueError(
                    f"Expected a Tinker XYZ atom-count line in {path}, got: {header.rstrip()}"
                ) from exc

            if n_atoms <= 0:
                raise ValueError(f"Invalid atom count {n_atoms} in {path}.")

            title = header[len(header_tokens[0]):].strip()
            frame_no += 1

            second = _read_next_nonblank(handle)
            if second is None:
                raise ValueError(f"Unexpected end of file after frame header in {path}.")

            box = parse_tinker_box_line(second)
            atoms: List[TinkerAtom] = []

            if box is None:
                # Nonperiodic XYZ: the second line is the first atom line.
                box = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                atoms.append(parse_tinker_atom_line(second, path, frame_no))

            while len(atoms) < n_atoms:
                line = _read_next_nonblank(handle)
                if line is None:
                    raise ValueError(
                        f"Unexpected end of file in {path}, frame {frame_no}; "
                        f"read {len(atoms)} of {n_atoms} atoms."
                    )
                atoms.append(parse_tinker_atom_line(line, path, frame_no))

            atom_ids = [atom.atom_id for atom in atoms]
            if len(set(atom_ids)) != n_atoms:
                raise ValueError(
                    f"Duplicate Tinker atom IDs in {path}, frame {frame_no}."
                )

            a, b, c, alpha, beta, gamma = box
            yield TinkerXYZFrame(
                source_file=path,
                source_frame=frame_no,
                n_atoms=n_atoms,
                title=title,
                a_A=a,
                b_A=b,
                c_A=c,
                alpha_deg=alpha,
                beta_deg=beta,
                gamma_deg=gamma,
                atoms=atoms,
            )


def read_first_tinker_xyz_frame(path: Path) -> TinkerXYZFrame:
    try:
        return next(iter_tinker_xyz_frames(path))
    except StopIteration as exc:
        raise RuntimeError(f"No Tinker XYZ frame found in {path}.") from exc


def compute_system_mass_from_tinker_xyz_prm(xyz_file: Path, prm_file: Path) -> float:
    """Compute the molar mass of one simulation cell from the first XYZ frame."""
    atom_masses = parse_atom_masses_from_prm(prm_file)
    if not atom_masses:
        raise RuntimeError(f"No atom masses were parsed from {prm_file}.")

    frame = read_first_tinker_xyz_frame(xyz_file)
    total_mass = 0.0
    missing_types = set()

    for atom in frame.atoms:
        mass = atom_masses.get(atom.atom_type)
        if mass is None:
            missing_types.add(atom.atom_type)
        else:
            total_mass += mass

    if missing_types:
        raise RuntimeError(
            "Missing masses in the PRM file for atom types: "
            + ", ".join(map(str, sorted(missing_types)))
        )

    return total_mass


# ============================================================
# Molecular connected components and inventories
# ============================================================

class UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {x: x for x in values}
        self.rank = {x: 0 for x in values}

    def find(self, x: int) -> int:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_connected_components(
    frame: TinkerXYZFrame,
) -> Tuple[List[List[TinkerAtom]], int, List[Tuple[int, int]]]:
    """
    Build molecular connected components from Tinker bond connectivity.

    Returns:
        components
        number of unique bonds
        invalid bond references
    """
    atom_by_id = {atom.atom_id: atom for atom in frame.atoms}
    uf = UnionFind(atom_by_id.keys())
    bond_pairs = set()
    invalid_bonds: List[Tuple[int, int]] = []

    for atom in frame.atoms:
        for bonded_id in atom.bonds:
            if bonded_id not in atom_by_id:
                invalid_bonds.append((atom.atom_id, bonded_id))
                continue
            a, b = sorted((atom.atom_id, bonded_id))
            if a != b:
                bond_pairs.add((a, b))
                uf.union(a, b)

    grouped: Dict[int, List[TinkerAtom]] = defaultdict(list)
    for atom in frame.atoms:
        grouped[uf.find(atom.atom_id)].append(atom)

    components = list(grouped.values())
    components.sort(key=lambda comp: min(atom.atom_id for atom in comp))
    return components, len(bond_pairs), invalid_bonds


def count_selection_entities(
    frame: TinkerXYZFrame,
    components: List[List[TinkerAtom]],
    selection: ConcentrationSelection,
) -> int:
    target = set(selection.atom_types)

    if selection.count_mode == "atom":
        return sum(1 for atom in frame.atoms if atom.atom_type in target)

    if selection.count_mode != "molecule":
        raise ValueError(f"Unknown count mode: {selection.count_mode}")

    count = 0
    for component in components:
        component_types = {atom.atom_type for atom in component}
        if selection.molecule_match == "all":
            matched = target.issubset(component_types)
        else:
            matched = bool(target.intersection(component_types))
        if matched:
            count += 1
    return count


def make_xyz_atom_type_inventory(
    frame: TinkerXYZFrame,
    components: List[List[TinkerAtom]],
) -> pd.DataFrame:
    """Create a first-frame inventory of atom types and connected components."""
    atom_counts = Counter(atom.atom_type for atom in frame.atoms)
    names_by_type: Dict[int, Counter] = defaultdict(Counter)
    components_by_type: Dict[int, int] = Counter()

    for atom in frame.atoms:
        names_by_type[atom.atom_type][atom.atom_name] += 1

    for component in components:
        for atom_type in {atom.atom_type for atom in component}:
            components_by_type[atom_type] += 1

    rows = []
    for atom_type in sorted(atom_counts):
        name_summary = ";".join(
            f"{name}:{count}"
            for name, count in sorted(names_by_type[atom_type].items())
        )
        rows.append(
            {
                "atom_type": atom_type,
                "atom_count": atom_counts[atom_type],
                "atom_names_and_counts": name_summary,
                "connected_components_containing_type": components_by_type[atom_type],
                "source_file": str(frame.source_file),
                "source_frame": frame.source_frame,
            }
        )

    return pd.DataFrame(rows)


def topology_signature(frame: TinkerXYZFrame) -> Tuple:
    """A strict, hashable signature of atom IDs, atom types, and bonds."""
    return tuple(
        (
            atom.atom_id,
            atom.atom_type,
            tuple(sorted(set(atom.bonds))),
        )
        for atom in frame.atoms
    )


# ============================================================
# Numbered snapshot and ARC/XYZ source discovery
# ============================================================

NUMERIC_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)\.(?P<index>\d{3,})$")


def parse_snapshot_index(path: Path) -> Optional[int]:
    m = re.fullmatch(r"\.(\d{3,})", path.suffix)
    if m:
        return int(m.group(1))
    return None


def discover_numbered_snapshot_series(
    temp_dir: Path,
    snapshot_prefix: Optional[str] = None,
) -> List[Tuple[int, Path]]:
    """
    Find one Tinker numbered snapshot series in a run directory.

    If multiple basenames are present, the largest series is chosen and a
    warning is printed. Use --snapshot-prefix for an explicit choice.
    """
    groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)

    for path in temp_dir.iterdir():
        if not path.is_file():
            continue
        m = NUMERIC_SUFFIX_RE.match(path.name)
        if not m:
            continue
        prefix = m.group("prefix")
        index = int(m.group("index"))
        groups[prefix].append((index, path))

    if snapshot_prefix is not None:
        selected = groups.get(snapshot_prefix, [])
        if not selected:
            available = ", ".join(sorted(groups)) or "none"
            raise FileNotFoundError(
                f"Snapshot prefix '{snapshot_prefix}' not found in {temp_dir}. "
                f"Available prefixes: {available}"
            )
        return sorted(selected, key=lambda x: x[0])

    if not groups:
        return []

    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    selected_prefix, selected = ranked[0]

    if len(ranked) > 1:
        description = ", ".join(f"{name} ({len(files)} frames)" for name, files in ranked)
        print(
            f"[WARNING] Multiple numbered XYZ series found in {temp_dir}: {description}"
        )
        print(
            f"[WARNING] Automatically selected '{selected_prefix}'. "
            "Use --snapshot-prefix to choose explicitly."
        )

    return sorted(selected, key=lambda x: x[0])


def discover_xyz_pattern_files(temp_dir: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(p for p in temp_dir.glob(pattern) if p.is_file()))
    # remove duplicates while preserving order
    seen = set()
    unique = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def find_reference_file(root: Path, temp_dirs: Sequence[Path], suffix: str) -> Optional[Path]:
    candidates: List[Path] = []
    candidates.extend(sorted(root.glob(f"*{suffix}")))
    for d in temp_dirs:
        if d.exists():
            candidates.extend(sorted(d.glob(f"*{suffix}")))

    if not candidates:
        return None

    for path in candidates:
        if path.parent == root:
            return path
    return candidates[0]


def find_first_xyz_source(
    temp_dirs: Sequence[Path],
    snapshot_prefix: Optional[str],
    xyz_patterns: Sequence[str],
) -> Optional[Path]:
    for temp_dir in temp_dirs:
        if not temp_dir.exists():
            continue
        numbered = discover_numbered_snapshot_series(temp_dir, snapshot_prefix)
        if numbered:
            return numbered[0][1]
        pattern_files = discover_xyz_pattern_files(temp_dir, xyz_patterns)
        if pattern_files:
            return pattern_files[0]
    return None


# ============================================================
# Parse Tinker XYZ time series for one run directory
# ============================================================

@dataclass
class XYZSeriesResult:
    dataframe: pd.DataFrame
    inventory: pd.DataFrame
    selection_counts: Dict[str, int]
    source_description: str


def parse_xyz_snapshot_series(
    temp_dir: Path,
    selections: Sequence[ConcentrationSelection],
    save_interval_ps: float,
    first_frame_time_ps: Optional[float],
    snapshot_prefix: Optional[str],
    xyz_patterns: Sequence[str],
    topology_xyz: Optional[Path],
    strict_topology: bool,
    allow_missing_types: bool,
    allow_bondless_molecule_count: bool,
) -> XYZSeriesResult:
    """Parse numbered snapshots or one/more multi-frame XYZ/ARC files."""
    numbered = discover_numbered_snapshot_series(temp_dir, snapshot_prefix)
    pattern_files: List[Path] = []

    if not numbered:
        pattern_files = discover_xyz_pattern_files(temp_dir, xyz_patterns)

    if not numbered and not pattern_files:
        return XYZSeriesResult(
            dataframe=pd.DataFrame(),
            inventory=pd.DataFrame(),
            selection_counts={},
            source_description="none",
        )

    # Establish topology from an explicit reference or from the first trajectory frame.
    topology_frame: Optional[TinkerXYZFrame] = None
    if topology_xyz is not None:
        topology_frame = read_first_tinker_xyz_frame(topology_xyz)

    rows: List[dict] = []
    inventory_df = pd.DataFrame()
    selection_counts: Dict[str, int] = {}
    reference_signature = None
    reference_n_atoms = None
    components: Optional[List[List[TinkerAtom]]] = None
    n_unique_bonds = 0
    source_description = ""

    def initialize_topology(frame: TinkerXYZFrame):
        nonlocal topology_frame
        nonlocal inventory_df
        nonlocal selection_counts
        nonlocal reference_signature
        nonlocal reference_n_atoms
        nonlocal components
        nonlocal n_unique_bonds

        if topology_frame is None:
            topology_frame = frame

        if topology_frame.n_atoms != frame.n_atoms:
            raise RuntimeError(
                f"Topology file {topology_frame.source_file} has {topology_frame.n_atoms} atoms, "
                f"but trajectory frame {frame.source_file} has {frame.n_atoms}."
            )

        components, n_unique_bonds, invalid_bonds = build_connected_components(topology_frame)
        if invalid_bonds:
            preview = ", ".join(f"{a}->{b}" for a, b in invalid_bonds[:10])
            raise RuntimeError(
                f"Invalid bond references in {topology_frame.source_file}: {preview}"
            )

        if any(sel.count_mode == "molecule" for sel in selections):
            if n_unique_bonds == 0 and topology_frame.n_atoms > 1 and not allow_bondless_molecule_count:
                raise RuntimeError(
                    "Molecule concentration was requested, but no bond connectivity was found "
                    f"in {topology_frame.source_file}. Use a complete Tinker XYZ topology via "
                    "--topology-xyz, use atom mode, or explicitly allow bondless counting with "
                    "--allow-bondless-molecule-count."
                )

        inventory_df = make_xyz_atom_type_inventory(topology_frame, components)
        available_types = {atom.atom_type for atom in topology_frame.atoms}

        for sel in selections:
            missing = sorted(set(sel.atom_types) - available_types)
            if missing and not allow_missing_types:
                raise RuntimeError(
                    f"Selection '{sel.label}' requests missing atom type(s) {missing}. "
                    f"Available types are: {sorted(available_types)}"
                )
            selection_counts[sel.token] = count_selection_entities(
                topology_frame,
                components,
                sel,
            )

        reference_n_atoms = topology_frame.n_atoms
        reference_signature = topology_signature(topology_frame) if strict_topology else None

    def verify_frame(frame: TinkerXYZFrame):
        if reference_n_atoms is None:
            return
        if frame.n_atoms != reference_n_atoms:
            raise RuntimeError(
                f"Atom count changed from {reference_n_atoms} to {frame.n_atoms} in "
                f"{frame.source_file}, frame {frame.source_frame}."
            )
        if strict_topology and topology_signature(frame) != reference_signature:
            raise RuntimeError(
                f"Atom types or bond topology changed in {frame.source_file}, "
                f"frame {frame.source_frame}."
            )

    def append_frame(frame: TinkerXYZFrame, local_frame: int, time_ps: float):
        if components is None:
            initialize_topology(frame)
        verify_frame(frame)

        row = {
            "frame": int(local_frame),
            "time_ps": float(time_ps),
            "xyz_source_file": str(frame.source_file),
            "xyz_source_frame": int(frame.source_frame),
            "n_atoms_xyz": int(frame.n_atoms),
            "n_connected_components": int(len(components or [])),
            "n_unique_bonds": int(n_unique_bonds),
            "a_A": frame.a_A,
            "b_A": frame.b_A,
            "c_A": frame.c_A,
            "alpha_deg": frame.alpha_deg,
            "beta_deg": frame.beta_deg,
            "gamma_deg": frame.gamma_deg,
            "volume_A3": frame.volume_A3,
        }

        for sel in selections:
            row[sel.count_column] = int(selection_counts[sel.token])
            row[sel.concentration_column] = concentration_mol_dm3(
                selection_counts[sel.token],
                frame.volume_A3,
            )

        rows.append(row)

    if numbered:
        source_description = f"numbered snapshots: {numbered[0][1].name} ... {numbered[-1][1].name}"
        min_index = min(index for index, _ in numbered)
        default_first = min_index * save_interval_ps
        first_time = default_first if first_frame_time_ps is None else first_frame_time_ps

        local_frame = 0
        for index, path in numbered:
            frames = list(iter_tinker_xyz_frames(path))
            if not frames:
                continue
            if len(frames) > 1:
                print(
                    f"[WARNING] Numbered snapshot {path} contains {len(frames)} XYZ frames. "
                    "All frames will be used."
                )
            for sub_index, frame in enumerate(frames):
                local_frame += 1
                time_ps = first_time + (index - min_index + sub_index) * save_interval_ps
                append_frame(frame, local_frame, time_ps)

    else:
        source_description = "multi-frame XYZ/ARC: " + " + ".join(p.name for p in pattern_files)
        first_time = save_interval_ps if first_frame_time_ps is None else first_frame_time_ps
        local_frame = 0
        for path in pattern_files:
            for frame in iter_tinker_xyz_frames(path):
                local_frame += 1
                time_ps = first_time + (local_frame - 1) * save_interval_ps
                append_frame(frame, local_frame, time_ps)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time_ps").reset_index(drop=True)

    return XYZSeriesResult(
        dataframe=df,
        inventory=inventory_df,
        selection_counts=selection_counts,
        source_description=source_description,
    )


# ============================================================
# Parse Tinker log files
# ============================================================


def parse_tinker_log(log_file: Path) -> pd.DataFrame:
    """Parse common Tinker-GPU/Tinker-HP dynamic output fields."""
    rows: List[dict] = []
    current: Dict[str, object] = {}

    def flush_current():
        nonlocal current
        if current:
            rows.append(current)
            current = {}

    with log_file.open("r", errors="ignore") as f:
        lines = list(f)

    i = 0
    while i < len(lines):
        line = lines[i]
        low = line.lower()
        nums = extract_floats(line)

        if "current time" in low and nums:
            flush_current()
            current["time_ps"] = nums[0]
            current["log_file"] = str(log_file)

        elif "current potential" in low and nums:
            current["potential_kcal_mol"] = nums[0]
        elif "current kinetic" in low and nums:
            current["kinetic_kcal_mol"] = nums[0]
        elif "total energy" in low and nums:
            current["total_energy_kcal_mol"] = nums[0]
        elif "potential energy" in low and nums:
            current.setdefault("potential_kcal_mol", nums[0])
        elif "kinetic energy" in low and nums:
            current.setdefault("kinetic_kcal_mol", nums[0])

        elif "current temperature" in low and nums:
            current["temperature_K"] = nums[0]
        elif "temperature" in low and "current" not in low and nums:
            current.setdefault("temperature_K", nums[0])
        elif "current pressure" in low and nums:
            current["pressure_atm"] = nums[0]
        elif "pressure" in low and "current" not in low and nums:
            current.setdefault("pressure_atm", nums[0])

        elif ("lattice lengths" in low or "cell lengths" in low) and len(nums) >= 3:
            current["a_A"] = nums[0]
            current["b_A"] = nums[1]
            current["c_A"] = nums[2]
        elif ("lattice angles" in low or "cell angles" in low) and len(nums) >= 3:
            current["alpha_deg"] = nums[0]
            current["beta_deg"] = nums[1]
            current["gamma_deg"] = nums[2]
        elif "periodic box dimensions" in low or "box dimensions" in low:
            nearby = line
            for j in range(1, 4):
                if i + j < len(lines):
                    nearby += " " + lines[i + j]
            vals = extract_floats(nearby)
            if len(vals) >= 6:
                current["a_A"] = vals[0]
                current["b_A"] = vals[1]
                current["c_A"] = vals[2]
                current["alpha_deg"] = vals[3]
                current["beta_deg"] = vals[4]
                current["gamma_deg"] = vals[5]

        i += 1

    flush_current()
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if "time_ps" in df.columns:
        df["time_ps"] = pd.to_numeric(df["time_ps"], errors="coerce")
        df = df.dropna(subset=["time_ps"])
        df = df.sort_values("time_ps")
        # Multiple logs can contain duplicate times; keep the latest parsed record.
        df = df.drop_duplicates(subset=["time_ps"], keep="last")

    useful_cols = [
        "potential_kcal_mol",
        "kinetic_kcal_mol",
        "total_energy_kcal_mol",
        "temperature_K",
        "pressure_atm",
        "a_A",
        "b_A",
        "c_A",
    ]
    present = [c for c in useful_cols if c in df.columns]
    if present:
        df = df.dropna(subset=present, how="all")

    return df


def parse_run_logs(temp_dir: Path) -> pd.DataFrame:
    log_files = sorted(temp_dir.glob("*.out"))
    preferred = [
        p for p in log_files
        if "log" in p.name.lower() or "dynamic" in p.name.lower()
    ]
    if preferred:
        log_files = preferred

    dfs = []
    for path in log_files:
        parsed = parse_tinker_log(path)
        if not parsed.empty:
            dfs.append(parsed)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    if "time_ps" in df.columns:
        df = df.sort_values("time_ps").drop_duplicates("time_ps", keep="last")
    return df


# ============================================================
# Merge logs onto the XYZ primary time grid
# ============================================================


def merge_log_onto_xyz(
    xyz_df: pd.DataFrame,
    log_df: pd.DataFrame,
    merge_tolerance_ps: Optional[float],
) -> pd.DataFrame:
    """
    Preserve one output row per XYZ frame and attach nearest log values.

    XYZ cell values always take priority because density and concentration are
    evaluated on the same instantaneous snapshot volume.
    """
    if xyz_df is None or xyz_df.empty:
        return log_df.copy() if log_df is not None else pd.DataFrame()
    if log_df is None or log_df.empty:
        return xyz_df.copy()

    left = xyz_df.copy()
    right = log_df.copy()
    left["time_ps"] = pd.to_numeric(left["time_ps"], errors="coerce")
    right["time_ps"] = pd.to_numeric(right["time_ps"], errors="coerce")
    left = left.dropna(subset=["time_ps"]).sort_values("time_ps")
    right = right.dropna(subset=["time_ps"]).sort_values("time_ps")

    merged = pd.merge_asof(
        left,
        right,
        on="time_ps",
        direction="nearest",
        tolerance=merge_tolerance_ps,
        suffixes=("", "_log"),
    )

    # Cell values from XYZ are authoritative. If any are missing, fill from log.
    for col in ["a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"]:
        log_col = f"{col}_log"
        if log_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[log_col])
            else:
                merged[col] = merged[log_col]
            merged = merged.drop(columns=[log_col])

    return merged


# ============================================================
# Derived quantities
# ============================================================


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = ["a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"]
    if all(c in df.columns for c in needed):
        df["volume_A3"] = [
            cell_volume(a, b, c, alpha, beta, gamma)
            for a, b, c, alpha, beta, gamma in zip(
                df["a_A"],
                df["b_A"],
                df["c_A"],
                df["alpha_deg"],
                df["beta_deg"],
                df["gamma_deg"],
            )
        ]
    return df


def add_density(df: pd.DataFrame, system_mass_g_mol: Optional[float]) -> pd.DataFrame:
    df = df.copy()
    if system_mass_g_mol is None or "volume_A3" not in df.columns:
        return df

    df["system_mass_g_mol"] = float(system_mass_g_mol)
    df["density_g_cm3"] = (
        float(system_mass_g_mol)
        / (AVOGADRO * df["volume_A3"] * ANG3_TO_CM3)
    )
    return df


def refresh_concentrations(
    df: pd.DataFrame,
    selections: Sequence[ConcentrationSelection],
) -> pd.DataFrame:
    """Recalculate concentration from count columns and the final volume column."""
    df = df.copy()
    if "volume_A3" not in df.columns:
        return df

    for sel in selections:
        if sel.count_column in df.columns:
            df[sel.concentration_column] = [
                concentration_mol_dm3(int(count), volume)
                if pd.notna(count)
                else np.nan
                for count, volume in zip(df[sel.count_column], df["volume_A3"])
            ]
    return df


def add_cumulative_averages(
    df: pd.DataFrame,
    selections: Sequence[ConcentrationSelection],
) -> pd.DataFrame:
    df = df.copy()

    target_cols = [
        "potential_kcal_mol",
        "kinetic_kcal_mol",
        "total_energy_kcal_mol",
        "temperature_K",
        "pressure_atm",
        "a_A",
        "b_A",
        "c_A",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "volume_A3",
        "density_g_cm3",
    ]
    target_cols.extend(sel.concentration_column for sel in selections)

    for col in target_cols:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_cumavg"] = numeric.expanding(min_periods=1).mean()

    return df


# ============================================================
# Analyze one physical run directory
# ============================================================

@dataclass
class RunAnalysisResult:
    dataframe: pd.DataFrame
    inventory: pd.DataFrame
    selection_counts: Dict[str, int]
    source_description: str


def extract_temperature_dir(
    temp_dir: Path,
    selections: Sequence[ConcentrationSelection],
    save_interval_ps: float,
    first_frame_time_ps: Optional[float],
    snapshot_prefix: Optional[str],
    xyz_patterns: Sequence[str],
    topology_xyz: Optional[Path],
    strict_topology: bool,
    allow_missing_types: bool,
    allow_bondless_molecule_count: bool,
    merge_tolerance_ps: Optional[float],
) -> RunAnalysisResult:
    print(f"[ANALYZE RUN] {temp_dir}")

    xyz_result = parse_xyz_snapshot_series(
        temp_dir=temp_dir,
        selections=selections,
        save_interval_ps=save_interval_ps,
        first_frame_time_ps=first_frame_time_ps,
        snapshot_prefix=snapshot_prefix,
        xyz_patterns=xyz_patterns,
        topology_xyz=topology_xyz,
        strict_topology=strict_topology,
        allow_missing_types=allow_missing_types,
        allow_bondless_molecule_count=allow_bondless_molecule_count,
    )

    if not xyz_result.dataframe.empty:
        print(f"  XYZ source: {xyz_result.source_description}")
        print(f"  XYZ frames: {len(xyz_result.dataframe)}")

    log_df = parse_run_logs(temp_dir)
    df = merge_log_onto_xyz(
        xyz_df=xyz_result.dataframe,
        log_df=log_df,
        merge_tolerance_ps=merge_tolerance_ps,
    )

    if df.empty:
        print(f"[WARNING] No analyzable XYZ snapshots or log records in {temp_dir}")
        return RunAnalysisResult(
            dataframe=df,
            inventory=xyz_result.inventory,
            selection_counts=xyz_result.selection_counts,
            source_description=xyz_result.source_description,
        )

    if "time_ps" in df.columns:
        df["time_ps"] = pd.to_numeric(df["time_ps"], errors="coerce")
        df = df.dropna(subset=["time_ps"]).sort_values("time_ps")

    df["run_dir"] = temp_dir.name
    df["source_dir"] = str(temp_dir)
    df["run_index"] = run_number_from_path(temp_dir)

    return RunAnalysisResult(
        dataframe=df,
        inventory=xyz_result.inventory,
        selection_counts=xyz_result.selection_counts,
        source_description=xyz_result.source_description,
    )


# ============================================================
# Analyze and stitch one temperature series
# ============================================================

@dataclass
class TemperatureAnalysisResult:
    dataframe: pd.DataFrame
    concentration_dataframe: pd.DataFrame
    inventory: pd.DataFrame
    selection_metadata: pd.DataFrame


def analyze_temperature_series(
    temp_label: str,
    run_dirs: Sequence[Path],
    out_dir: Path,
    selections: Sequence[ConcentrationSelection],
    save_interval_ps: float,
    first_frame_time_ps: Optional[float],
    snapshot_prefix: Optional[str],
    xyz_patterns: Sequence[str],
    topology_xyz: Optional[Path],
    strict_topology: bool,
    allow_missing_types: bool,
    allow_bondless_molecule_count: bool,
    merge_tolerance_ps: Optional[float],
    force: bool,
    system_mass_g_mol: Optional[float],
    join_time: bool,
) -> TemperatureAnalysisResult:
    temp_K = infer_temp_from_path(Path(temp_label))
    csv_file = out_dir / f"{temp_label}_timeseries.csv"
    conc_csv_file = out_dir / f"{temp_label}_concentration_timeseries.csv"
    inventory_file = out_dir / f"{temp_label}_xyz_atom_type_inventory.csv"

    if csv_file.exists() and not force:
        print(f"[READ CSV] {csv_file}")
        df = pd.read_csv(csv_file)
        conc_df = pd.read_csv(conc_csv_file) if conc_csv_file.exists() else pd.DataFrame()
        inventory_df = pd.read_csv(inventory_file) if inventory_file.exists() else pd.DataFrame()
        return TemperatureAnalysisResult(
            dataframe=df,
            concentration_dataframe=conc_df,
            inventory=inventory_df,
            selection_metadata=pd.DataFrame(),
        )

    print(f"[ANALYZE SERIES] {temp_label}")
    for d in run_dirs:
        print(f"  - {d}")

    stitched: List[pd.DataFrame] = []
    inventory_parts: List[pd.DataFrame] = []
    metadata_rows: List[dict] = []
    current_max_time: Optional[float] = None
    reference_counts: Optional[Dict[str, int]] = None

    for continuation_order, run_dir in enumerate(
        sorted(run_dirs, key=sort_key_for_run_dir),
        start=1,
    ):
        if not run_dir.exists():
            print(f"[WARNING] Directory not found: {run_dir}")
            continue

        result = extract_temperature_dir(
            temp_dir=run_dir,
            selections=selections,
            save_interval_ps=save_interval_ps,
            first_frame_time_ps=first_frame_time_ps,
            snapshot_prefix=snapshot_prefix,
            xyz_patterns=xyz_patterns,
            topology_xyz=topology_xyz,
            strict_topology=strict_topology,
            allow_missing_types=allow_missing_types,
            allow_bondless_molecule_count=allow_bondless_molecule_count,
            merge_tolerance_ps=merge_tolerance_ps,
        )

        df = result.dataframe
        if df.empty:
            continue

        df = df.copy()
        df["temperature_dir"] = temp_label
        df["target_temperature_K"] = temp_K
        df["continuation_order"] = continuation_order

        if "time_ps" in df.columns:
            df = df.sort_values("time_ps")
            df["time_ps_local"] = df["time_ps"]

            if join_time:
                local_min = float(df["time_ps_local"].min())
                if current_max_time is None:
                    offset = 0.0
                else:
                    local_dt = estimate_time_step_ps(
                        df["time_ps_local"],
                        fallback=save_interval_ps,
                    )
                    if local_min <= current_max_time + 1.0e-9:
                        offset = current_max_time + local_dt - local_min
                    else:
                        offset = 0.0
                df["time_offset_ps"] = offset
                df["time_ps"] = df["time_ps_local"] + offset
                current_max_time = float(df["time_ps"].max())
            else:
                df["time_offset_ps"] = 0.0
                current_max_time = float(df["time_ps"].max())

        if result.selection_counts:
            if reference_counts is None:
                reference_counts = dict(result.selection_counts)
            elif result.selection_counts != reference_counts:
                raise RuntimeError(
                    f"Selected entity counts differ between continuation runs of {temp_label}: "
                    f"{reference_counts} versus {result.selection_counts} in {run_dir}."
                )

        if not result.inventory.empty:
            inv = result.inventory.copy()
            inv.insert(0, "temperature_dir", temp_label)
            inv.insert(1, "run_dir", run_dir.name)
            inventory_parts.append(inv)

        for sel in selections:
            metadata_rows.append(
                {
                    "temperature_dir": temp_label,
                    "run_dir": run_dir.name,
                    "label": sel.label,
                    "token": sel.token,
                    "count_mode": sel.count_mode,
                    "atom_types": ",".join(map(str, sel.atom_types)),
                    "molecule_match": sel.molecule_match,
                    "entity_count": result.selection_counts.get(sel.token, np.nan),
                    "count_column": sel.count_column,
                    "concentration_column": sel.concentration_column,
                    "xyz_source": result.source_description,
                }
            )

        stitched.append(df)

    if not stitched:
        print(f"[WARNING] No analyzable data found in series {temp_label}")
        return TemperatureAnalysisResult(
            dataframe=pd.DataFrame(),
            concentration_dataframe=pd.DataFrame(),
            inventory=pd.DataFrame(),
            selection_metadata=pd.DataFrame(metadata_rows),
        )

    df = pd.concat(stitched, ignore_index=True)
    if "time_ps" in df.columns:
        df = df.sort_values(["time_ps", "continuation_order"]).reset_index(drop=True)

    df = add_volume(df)
    df = refresh_concentrations(df, selections)
    df = add_density(df, system_mass_g_mol)
    df = add_cumulative_averages(df, selections)

    first_cols = [
        "temperature_dir",
        "target_temperature_K",
        "time_ps",
        "time_ps_local",
        "time_offset_ps",
        "continuation_order",
        "run_dir",
        "run_index",
        "frame",
        "xyz_source_file",
        "xyz_source_frame",
        "n_atoms_xyz",
        "n_connected_components",
        "n_unique_bonds",
        "potential_kcal_mol",
        "kinetic_kcal_mol",
        "total_energy_kcal_mol",
        "temperature_K",
        "pressure_atm",
        "a_A",
        "b_A",
        "c_A",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "volume_A3",
        "density_g_cm3",
        "system_mass_g_mol",
    ]
    for sel in selections:
        first_cols.extend(
            [
                sel.count_column,
                sel.concentration_column,
                sel.cumulative_column,
            ]
        )
    first_cols.append("source_dir")

    cols = [c for c in first_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    df.to_csv(csv_file, index=False)
    print(f"[WRITE] {csv_file}")

    if selections:
        concentration_cols = [
            "temperature_dir",
            "target_temperature_K",
            "time_ps",
            "time_ps_local",
            "time_offset_ps",
            "continuation_order",
            "run_dir",
            "frame",
            "xyz_source_file",
            "xyz_source_frame",
            "volume_A3",
            "density_g_cm3",
        ]
        for sel in selections:
            concentration_cols.extend(
                [sel.count_column, sel.concentration_column, sel.cumulative_column]
            )
        concentration_cols = [c for c in concentration_cols if c in df.columns]
        conc_df = df[concentration_cols].copy()
        conc_df.to_csv(conc_csv_file, index=False)
        print(f"[WRITE] {conc_csv_file}")
    else:
        conc_df = pd.DataFrame()

    inventory_df = pd.concat(inventory_parts, ignore_index=True) if inventory_parts else pd.DataFrame()
    if not inventory_df.empty:
        inventory_df.to_csv(inventory_file, index=False)
        print(f"[WRITE] {inventory_file}")

    return TemperatureAnalysisResult(
        dataframe=df,
        concentration_dataframe=conc_df,
        inventory=inventory_df,
        selection_metadata=pd.DataFrame(metadata_rows),
    )


# ============================================================
# Plotting
# ============================================================


def plot_property(all_df: pd.DataFrame, out_dir: Path, ycol: str, ylabel: str):
    if ycol not in all_df.columns:
        return

    plot_df = all_df.dropna(subset=[ycol])
    if plot_df.empty:
        return

    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.2, 4.6))
    for label, sub in plot_df.groupby("temperature_dir"):
        sub = sub.sort_values("time_ps") if "time_ps" in sub.columns else sub
        x = sub["time_ps"] if "time_ps" in sub.columns else np.arange(len(sub))
        plt.plot(x, sub[ycol], label=label)
    plt.xlabel("Time / ps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    fig_file = png_dir / f"{ycol}.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"[WRITE] {fig_file}")

    cum_col = f"{ycol}_cumavg"
    if cum_col not in all_df.columns:
        return

    cum_df = all_df.dropna(subset=[cum_col])
    if cum_df.empty:
        return

    plt.figure(figsize=(7.2, 4.6))
    for label, sub in cum_df.groupby("temperature_dir"):
        sub = sub.sort_values("time_ps") if "time_ps" in sub.columns else sub
        x = sub["time_ps"] if "time_ps" in sub.columns else np.arange(len(sub))
        plt.plot(x, sub[cum_col], label=label)
    plt.xlabel("Time / ps")
    plt.ylabel(f"Cumulative average of {ylabel}")
    plt.legend()
    plt.tight_layout()
    fig_file = png_dir / f"{cum_col}.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"[WRITE] {fig_file}")


def make_plots(
    all_df: pd.DataFrame,
    out_dir: Path,
    selections: Sequence[ConcentrationSelection],
):
    plot_targets = {
        "potential_kcal_mol": "Potential energy / kcal mol$^{-1}$",
        "kinetic_kcal_mol": "Kinetic energy / kcal mol$^{-1}$",
        "total_energy_kcal_mol": "Total energy / kcal mol$^{-1}$",
        "temperature_K": "Temperature / K",
        "pressure_atm": "Pressure / atm",
        "a_A": "a / Å",
        "b_A": "b / Å",
        "c_A": "c / Å",
        "alpha_deg": "alpha / degree",
        "beta_deg": "beta / degree",
        "gamma_deg": "gamma / degree",
        "volume_A3": "Volume / Å$^3$",
        "density_g_cm3": "Density / g cm$^{-3}$",
    }

    for sel in selections:
        plot_targets[sel.concentration_column] = (
            f"{sel.label} concentration / mol dm$^{{-3}}$"
        )

    for col, ylabel in plot_targets.items():
        plot_property(all_df, out_dir, col, ylabel)


# ============================================================
# Summary
# ============================================================


def make_summary(
    all_df: pd.DataFrame,
    selections: Sequence[ConcentrationSelection],
) -> pd.DataFrame:
    summary_cols = [
        "potential_kcal_mol",
        "kinetic_kcal_mol",
        "total_energy_kcal_mol",
        "temperature_K",
        "pressure_atm",
        "a_A",
        "b_A",
        "c_A",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "volume_A3",
        "density_g_cm3",
    ]
    summary_cols.extend(sel.concentration_column for sel in selections)

    rows = []
    for temp_label, sub in all_df.groupby("temperature_dir"):
        row = {
            "temperature_dir": temp_label,
            "target_temperature_K": (
                sub["target_temperature_K"].iloc[0]
                if "target_temperature_K" in sub.columns
                else np.nan
            ),
            "n_points": len(sub),
            "time_min_ps": sub["time_ps"].min() if "time_ps" in sub.columns else np.nan,
            "time_max_ps": sub["time_ps"].max() if "time_ps" in sub.columns else np.nan,
        }

        for sel in selections:
            if sel.count_column in sub.columns:
                values = pd.to_numeric(sub[sel.count_column], errors="coerce").dropna()
                row[f"{sel.count_column}_first"] = values.iloc[0] if not values.empty else np.nan
                row[f"{sel.count_column}_min"] = values.min() if not values.empty else np.nan
                row[f"{sel.count_column}_max"] = values.max() if not values.empty else np.nan

        for col in summary_cols:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
                cum_col = f"{col}_cumavg"
                if cum_col in sub.columns:
                    cum_values = pd.to_numeric(sub[cum_col], errors="coerce").dropna()
                    row[f"{col}_last_cumavg"] = (
                        cum_values.iloc[-1] if not cum_values.empty else np.nan
                    )

        rows.append(row)

    result = pd.DataFrame(rows)
    if "target_temperature_K" in result.columns:
        result = result.sort_values("target_temperature_K")
    return result


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Analyze Tinker NPT output using Tinker XYZ snapshots as the primary "
            "time grid, including density and selected atom/molecule concentrations."
        ),
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing TxxxK and continuation directories.",
    )
    parser.add_argument(
        "--temps",
        nargs="*",
        default=None,
        help="Temperature directories/labels, e.g. T250K T300K.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="analysis_result",
        help="Output directory relative to --root.",
    )
    parser.add_argument(
        "--save-interval-ps",
        type=float,
        default=50.0,
        help="Time interval between XYZ snapshots or ARC frames. Default: 50 ps.",
    )
    parser.add_argument(
        "--first-frame-time-ps",
        type=float,
        default=None,
        help=(
            "Time assigned to the first selected XYZ frame. For numbered .001 files, "
            "the default is index*interval; for ARC files, the default is one interval."
        ),
    )
    parser.add_argument(
        "--snapshot-prefix",
        type=str,
        default=None,
        help=(
            "Exact basename before .001/.002/... when multiple numbered XYZ series "
            "exist in one directory."
        ),
    )
    parser.add_argument(
        "--xyz-pattern",
        action="append",
        default=None,
        help=(
            "Glob pattern for multi-frame Tinker ARC/XYZ files, used only when no "
            "numbered snapshots are found. Repeatable. Default: *.arc, *.arc.gz."
        ),
    )
    parser.add_argument(
        "--topology-xyz",
        type=str,
        default=None,
        help=(
            "Complete Tinker XYZ file used for atom types and bond connectivity. "
            "Useful when trajectory frames omit connectivity."
        ),
    )
    parser.add_argument(
        "--strict-topology",
        action="store_true",
        help="Verify atom IDs, atom types, and bond lists in every XYZ frame.",
    )

    parser.add_argument(
        "--conc-atom-type",
        action="append",
        default=None,
        metavar="LABEL:TYPE[,TYPE...]",
        help=(
            "Count selected atom type(s) directly and calculate concentration. "
            "Repeatable. Example: --conc-atom-type LiFSA:6"
        ),
    )
    parser.add_argument(
        "--conc-molecule-type",
        action="append",
        default=None,
        metavar="LABEL:TYPE[,TYPE...]",
        help=(
            "Count connected molecular components containing selected atom type(s). "
            "Repeatable. Example: --conc-molecule-type FSA:164"
        ),
    )
    parser.add_argument(
        "--molecule-type-match",
        choices=["any", "all"],
        default="any",
        help=(
            "For multi-type molecule selections, require any or all selected types "
            "inside one connected component. Default: any."
        ),
    )
    parser.add_argument(
        "--allow-missing-types",
        action="store_true",
        help="Allow a requested atom type to be absent and return a zero count.",
    )
    parser.add_argument(
        "--allow-bondless-molecule-count",
        action="store_true",
        help=(
            "Allow molecule mode when the topology contains no bonds. Normally this "
            "is rejected because each atom would otherwise be treated as a molecule."
        ),
    )
    parser.add_argument(
        "--list-xyz-types",
        action="store_true",
        help="Write and print the first-frame Tinker XYZ atom-type inventory.",
    )

    parser.add_argument(
        "--mass-g-mol",
        type=float,
        default=None,
        help="Total molar mass of one simulation cell. Overrides XYZ/PRM calculation.",
    )
    parser.add_argument(
        "--xyz-for-mass",
        type=str,
        default=None,
        help="Tinker XYZ/snapshot/ARC file used to calculate system mass.",
    )
    parser.add_argument(
        "--prm-for-mass",
        type=str,
        default=None,
        help="Tinker PRM file used to calculate system mass.",
    )
    parser.add_argument(
        "--merge-tolerance-ps",
        type=float,
        default=None,
        help=(
            "Maximum time difference for attaching nearest log values to an XYZ frame. "
            "Default: half of --save-interval-ps."
        ),
    )

    parser.add_argument(
        "--no-join-continuations",
        action="store_true",
        help="Do not combine T300K, T300K_run_002, etc.",
    )
    parser.add_argument(
        "--no-time-stitch",
        action="store_true",
        help="Do not shift restart-local time axes into one continuous trajectory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reanalyze even when output CSV files already exist.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate PNG figures.",
    )

    args = parser.parse_args()

    if args.save_interval_ps <= 0.0:
        parser.error("--save-interval-ps must be positive.")

    selections = build_selections(args)
    root = Path(args.root).resolve()
    out_dir = root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    join_continuations = not args.no_join_continuations
    join_time = not args.no_time_stitch
    xyz_patterns = args.xyz_pattern or ["*.arc", "*.arc.gz"]
    merge_tolerance_ps = (
        args.merge_tolerance_ps
        if args.merge_tolerance_ps is not None
        else 0.5 * args.save_interval_ps + 1.0e-9
    )

    temp_groups = discover_temperature_groups(
        root=root,
        temps=args.temps,
        join_continuations=join_continuations,
    )
    if not temp_groups:
        raise RuntimeError(f"No temperature directories found under {root}.")

    temp_dirs_flat = [d for _, dirs in temp_groups for d in dirs]

    topology_xyz = Path(args.topology_xyz).resolve() if args.topology_xyz else None
    if topology_xyz is not None and not topology_xyz.exists():
        raise FileNotFoundError(f"Topology XYZ not found: {topology_xyz}")

    # Determine mass of one simulation cell.
    system_mass_g_mol = args.mass_g_mol
    if system_mass_g_mol is not None:
        print(f"[INFO] Manually specified system mass: {system_mass_g_mol:.8f} g/mol")
    else:
        if args.xyz_for_mass:
            xyz_for_mass = Path(args.xyz_for_mass).resolve()
        else:
            xyz_for_mass = find_reference_file(root, temp_dirs_flat, ".xyz")
            if xyz_for_mass is None:
                xyz_for_mass = find_first_xyz_source(
                    temp_dirs_flat,
                    args.snapshot_prefix,
                    xyz_patterns,
                )

        if args.prm_for_mass:
            prm_for_mass = Path(args.prm_for_mass).resolve()
        else:
            prm_for_mass = find_reference_file(root, temp_dirs_flat, ".prm")

        if xyz_for_mass is not None and prm_for_mass is not None:
            try:
                system_mass_g_mol = compute_system_mass_from_tinker_xyz_prm(
                    xyz_for_mass,
                    prm_for_mass,
                )
                print(f"[INFO] Mass reference XYZ: {xyz_for_mass}")
                print(f"[INFO] Mass reference PRM: {prm_for_mass}")
                print(f"[INFO] Computed system mass: {system_mass_g_mol:.8f} g/mol")
            except Exception as exc:
                print(f"[WARNING] System mass calculation failed: {exc}")
                print("[WARNING] Density will not be calculated.")
                system_mass_g_mol = None
        else:
            print("[WARNING] XYZ and/or PRM mass reference not found.")
            print("[WARNING] Density will not be calculated.")

    print("Temperature series:")
    for label, dirs in temp_groups:
        print(f"  - {label}: " + " + ".join(d.name for d in dirs))

    all_dfs: List[pd.DataFrame] = []
    all_conc_dfs: List[pd.DataFrame] = []
    all_inventory: List[pd.DataFrame] = []
    all_metadata: List[pd.DataFrame] = []

    for temp_label, run_dirs in temp_groups:
        result = analyze_temperature_series(
            temp_label=temp_label,
            run_dirs=run_dirs,
            out_dir=out_dir,
            selections=selections,
            save_interval_ps=args.save_interval_ps,
            first_frame_time_ps=args.first_frame_time_ps,
            snapshot_prefix=args.snapshot_prefix,
            xyz_patterns=xyz_patterns,
            topology_xyz=topology_xyz,
            strict_topology=args.strict_topology,
            allow_missing_types=args.allow_missing_types,
            allow_bondless_molecule_count=args.allow_bondless_molecule_count,
            merge_tolerance_ps=merge_tolerance_ps,
            force=args.force,
            system_mass_g_mol=system_mass_g_mol,
            join_time=join_time,
        )

        if not result.dataframe.empty:
            all_dfs.append(result.dataframe)
        if not result.concentration_dataframe.empty:
            all_conc_dfs.append(result.concentration_dataframe)
        if not result.inventory.empty:
            all_inventory.append(result.inventory)
        if not result.selection_metadata.empty:
            all_metadata.append(result.selection_metadata)

    if not all_dfs:
        raise RuntimeError("No data were extracted.")

    all_df = pd.concat(all_dfs, ignore_index=True)
    if "target_temperature_K" in all_df.columns and "time_ps" in all_df.columns:
        all_df = all_df.sort_values(["target_temperature_K", "time_ps"])

    all_csv = out_dir / "all_temperatures_timeseries.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"[WRITE] {all_csv}")

    if all_conc_dfs:
        all_conc_df = pd.concat(all_conc_dfs, ignore_index=True)
        all_conc_csv = out_dir / "all_concentrations_timeseries.csv"
        all_conc_df.to_csv(all_conc_csv, index=False)
        print(f"[WRITE] {all_conc_csv}")

    if all_inventory:
        inventory_df = pd.concat(all_inventory, ignore_index=True)
        inventory_csv = out_dir / "all_xyz_atom_type_inventory.csv"
        inventory_df.to_csv(inventory_csv, index=False)
        print(f"[WRITE] {inventory_csv}")

        if args.list_xyz_types:
            display_cols = [
                c for c in [
                    "temperature_dir",
                    "run_dir",
                    "atom_type",
                    "atom_count",
                    "atom_names_and_counts",
                    "connected_components_containing_type",
                ]
                if c in inventory_df.columns
            ]
            print("\nXYZ atom-type inventory:")
            print(inventory_df[display_cols].to_string(index=False))

    if all_metadata:
        metadata_df = pd.concat(all_metadata, ignore_index=True)
        metadata_csv = out_dir / "concentration_selection_metadata.csv"
        metadata_df.to_csv(metadata_csv, index=False)
        print(f"[WRITE] {metadata_csv}")

    summary_df = make_summary(all_df, selections)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[WRITE] {summary_csv}")

    if not args.no_plot:
        make_plots(all_df, out_dir, selections)

    print("\nDone.")
    print(f"Results are saved in: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
