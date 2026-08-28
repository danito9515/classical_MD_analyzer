#!/usr/bin/env python3
"""Coarse-sampling residence-time pilot for multi-model PDB trajectories.

The script is designed for Tinker-style PDB files in which the atom type is
stored in the PDB resname field.  Molecular targets are identified by one
anchor atom type and all atoms sharing its PDB residue id are treated as one
molecule.  Li--FSA contacts use FSA oxygen atoms by default; Li--SN contacts
use SN nitrogen atoms by default.

Important: with --dt-ns 0.2, lifetimes below 0.2 ns are not resolved.  Event
durations are therefore reported with sampling-interval bounds and censored
events are retained explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise SystemExit("This script requires scipy (ndimage, optimize, signal, spatial).") from exc


VERSION = "1.0.2"
ELEMENTS = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG",
    "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "FE", "CO", "NI", "CU",
    "ZN", "BR", "I",
}


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resid: int
    icode: str
    element: str

    @property
    def residue_key(self) -> Tuple[str, int, str]:
        return (self.chain, self.resid, self.icode)


@dataclass
class Trajectory:
    atoms: List[Atom]
    xyz: np.ndarray                 # (nframe, natom, 3), float32, Angstrom
    cells: np.ndarray               # (nframe, 3, 3), float64, row vectors
    model_ids: List[Optional[int]]


@dataclass(frozen=True)
class MolecularTarget:
    label: str
    anchor_value: str
    charge: Optional[float]
    source_spec: str


def infer_element(atom_name: str, explicit: str = "") -> str:
    explicit = explicit.strip().upper()
    if explicit in ELEMENTS:
        return explicit.title()
    token = re.sub(r"[^A-Za-z]", "", atom_name).upper()
    if not token:
        return "X"
    if len(token) >= 2 and token[:2] in ELEMENTS:
        return token[:2].title()
    if token[:1] in ELEMENTS:
        return token[:1].title()
    return token[:1].title()


def cell_matrix(a: float, b: float, c: float,
                alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    alpha, beta, gamma = np.deg2rad([alpha_deg, beta_deg, gamma_deg])
    sg = math.sin(gamma)
    if abs(sg) < 1.0e-10:
        raise ValueError("Invalid CRYST1 cell: sin(gamma) is zero.")
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(gamma), b * sg, 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sg
    cz2 = c * c - cx * cx - cy * cy
    vc = np.array([cx, cy, math.sqrt(max(cz2, 0.0))])
    return np.vstack([va, vb, vc])


def parse_cryst1(line: str) -> np.ndarray:
    try:
        vals = [float(line[6:15]), float(line[15:24]), float(line[24:33]),
                float(line[33:40]), float(line[40:47]), float(line[47:54])]
    except ValueError:
        fields = line.split()
        vals = [float(x) for x in fields[1:7]]
    return cell_matrix(*vals)


def parse_atom_line(line: str) -> Tuple[Atom, np.ndarray]:
    try:
        serial = int(line[6:11])
        name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21:22].strip()
        resid = int(line[22:26])
        icode = line[26:27].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]),
                        float(line[46:54])], dtype=np.float32)
        explicit = line[76:78].strip() if len(line) >= 78 else ""
    except (ValueError, IndexError):
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"Cannot parse PDB atom line: {line.rstrip()}")
        serial, name, resname = int(fields[1]), fields[2], fields[3]
        chain, resid, icode = "", int(fields[4]), ""
        xyz = np.array([float(x) for x in fields[5:8]], dtype=np.float32)
        explicit = fields[-1] if len(fields) >= 11 else ""
    atom = Atom(serial, name, resname, chain, resid, icode,
                infer_element(name, explicit))
    return atom, xyz


def read_pdb(path: Path) -> Trajectory:
    atoms0: Optional[List[Atom]] = None
    frames: List[np.ndarray] = []
    cells: List[np.ndarray] = []
    model_ids: List[Optional[int]] = []
    current_atoms: List[Atom] = []
    current_xyz: List[np.ndarray] = []
    current_cell: Optional[np.ndarray] = None
    global_cell: Optional[np.ndarray] = None
    current_model: Optional[int] = None

    def finish_frame() -> None:
        nonlocal atoms0, current_atoms, current_xyz, current_model
        if not current_xyz:
            return
        if atoms0 is None:
            atoms0 = list(current_atoms)
        elif len(current_atoms) != len(atoms0):
            raise ValueError(
                f"Topology changed in {path}: expected {len(atoms0)} atoms, "
                f"found {len(current_atoms)}."
            )
        cell = current_cell if current_cell is not None else global_cell
        if cell is None:
            raise ValueError(f"No CRYST1 record available for a frame in {path}.")
        frames.append(np.asarray(current_xyz, dtype=np.float32))
        cells.append(np.asarray(cell, dtype=float))
        model_ids.append(current_model)
        current_atoms, current_xyz, current_model = [], [], None

    with path.open("r", errors="replace") as handle:
        for line in handle:
            rec = line[:6].strip().upper()
            if rec == "CRYST1":
                parsed = parse_cryst1(line)
                if current_xyz:
                    current_cell = parsed
                else:
                    global_cell = parsed
                    current_cell = parsed
            elif rec == "MODEL":
                finish_frame()
                try:
                    current_model = int(line[10:14])
                except ValueError:
                    parts = line.split()
                    current_model = int(parts[1]) if len(parts) > 1 else None
                current_cell = global_cell
            elif rec in {"ATOM", "HETATM"}:
                atom, pos = parse_atom_line(line)
                current_atoms.append(atom)
                current_xyz.append(pos)
            elif rec in {"ENDMDL", "END"}:
                finish_frame()
                current_cell = global_cell
    finish_frame()
    if atoms0 is None or not frames:
        raise ValueError(f"No coordinate frames found in {path}.")
    xyz = np.stack(frames).astype(np.float32, copy=False)
    return Trajectory(atoms0, xyz, np.stack(cells), model_ids)


def parse_molecular_target(spec: str, role: str) -> MolecularTarget:
    fields = [x.strip() for x in spec.split(":")]
    charge: Optional[float] = None
    if len(fields) == 2:
        label, anchor = fields
    elif len(fields) == 3:
        if (role == "solvent" and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", fields[1])
                and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", fields[2])):
            # Compatibility with the user's provisional FSA:SN:71 notation.
            label, anchor = fields[1], fields[2]
            print(f"[warning] Interpreting {role} '{spec}' as {label}:{anchor}.",
                  file=sys.stderr)
        else:
            try:
                charge = float(fields[2])
                label, anchor = fields[0], fields[1]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {role} '{spec}'. Use LABEL:ATOM_TYPE[:CHARGE]."
                ) from exc
    else:
        raise ValueError(f"Invalid {role} '{spec}'. Use LABEL:ATOM_TYPE[:CHARGE].")
    if not label or not anchor:
        raise ValueError(f"Invalid empty field in {role} '{spec}'.")
    return MolecularTarget(label, anchor, charge, spec)


def atom_field(atom: Atom, field: str) -> str:
    return getattr(atom, field)


def select_cations(atoms: Sequence[Atom], selector: str) -> List[int]:
    key = selector.strip().upper()
    indices = [i for i, a in enumerate(atoms)
               if a.element.upper() == key or a.name.upper() == key
               or a.resname.upper() == key]
    if not indices:
        raise ValueError(f"Cation selector '{selector}' matched no atoms.")
    return indices


def build_molecules(atoms: Sequence[Atom], target: MolecularTarget,
                    atom_type_field: str,
                    contact_elements: Sequence[str]) -> Tuple[List[List[int]], List[int], np.ndarray]:
    key = target.anchor_value.upper()
    anchors = [i for i, atom in enumerate(atoms)
               if atom_field(atom, atom_type_field).upper() == key]
    if not anchors:
        raise ValueError(
            f"Target {target.label} anchor '{target.anchor_value}' matched no atoms "
            f"in field '{atom_type_field}'."
        )
    residue_members: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
    for idx, atom in enumerate(atoms):
        residue_members[atom.residue_key].append(idx)
    # A molecular identifier may occur more than once in one residue.  This is
    # expected for SN, which has two symmetry-related nitrile N atoms of the
    # same Tinker atom type.  The residue, not the individual anchor atom, is
    # the molecular identity.
    anchor_keys = list(dict.fromkeys(atoms[i].residue_key for i in anchors))
    molecules = [residue_members[k] for k in anchor_keys]
    allowed = {x.strip().upper() for x in contact_elements if x.strip()}
    contact_atom_indices: List[int] = []
    contact_owner: List[int] = []
    for imol, members in enumerate(molecules):
        chosen = [i for i in members if atoms[i].element.upper() in allowed]
        if not chosen:
            chosen = [next(i for i in anchors if atoms[i].residue_key == anchor_keys[imol])]
            print(
                f"[warning] {target.label} residue {anchor_keys[imol]} has none of "
                f"contact elements {sorted(allowed)}; using its anchor atom.",
                file=sys.stderr,
            )
        contact_atom_indices.extend(chosen)
        contact_owner.extend([imol] * len(chosen))
    return molecules, contact_atom_indices, np.asarray(contact_owner, dtype=np.int32)


def is_orthorhombic(cell: np.ndarray, tol: float = 1.0e-7) -> bool:
    return np.allclose(cell, np.diag(np.diag(cell)), atol=tol, rtol=0.0)


def mic_displacements(delta: np.ndarray, cell: np.ndarray,
                      inv_cell: Optional[np.ndarray] = None) -> np.ndarray:
    frac = delta @ (np.linalg.inv(cell) if inv_cell is None else inv_cell)
    frac -= np.rint(frac)
    return frac @ cell


def neighbor_atom_lists(cation_xyz: np.ndarray, contact_xyz: np.ndarray,
                        cell: np.ndarray, radius: float) -> List[np.ndarray]:
    if is_orthorhombic(cell) and np.all(np.diag(cell) > 2.0 * radius):
        box = np.diag(cell)
        q = np.mod(cation_xyz, box)
        p = np.mod(contact_xyz, box)
        tree = cKDTree(p, boxsize=box)
        return [np.asarray(x, dtype=np.int32) for x in tree.query_ball_point(q, radius)]
    delta = cation_xyz[:, None, :] - contact_xyz[None, :, :]
    delta = mic_displacements(delta.reshape(-1, 3), cell).reshape(delta.shape)
    d2 = np.einsum("...i,...i->...", delta, delta)
    return [np.flatnonzero(row <= radius * radius).astype(np.int32) for row in d2]


def candidate_molecule_distances(cation_xyz: np.ndarray, contact_xyz: np.ndarray,
                                 owner: np.ndarray, cell: np.ndarray,
                                 radius: float, nmol: int) -> Dict[int, float]:
    lists = neighbor_atom_lists(cation_xyz, contact_xyz, cell, radius)
    result: Dict[int, float] = {}
    inv_cell = np.linalg.inv(cell)
    for ili, atom_ids in enumerate(lists):
        if atom_ids.size == 0:
            continue
        delta = contact_xyz[atom_ids] - cation_xyz[ili]
        delta = mic_displacements(delta, cell, inv_cell)
        dist = np.linalg.norm(delta, axis=1)
        for atom_local, d in zip(atom_ids, dist):
            pid = ili * nmol + int(owner[atom_local])
            old = result.get(pid)
            if old is None or d < old:
                result[pid] = float(d)
    return result


def radial_distribution(traj: Trajectory, cation_idx: Sequence[int],
                        contact_idx: Sequence[int], bins: int, rmax: float,
                        max_sample_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, rmax, bins + 1)
    counts = np.zeros(bins, dtype=float)
    expected = np.zeros(bins, dtype=float)
    shell = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    nframe = traj.xyz.shape[0]
    take = np.unique(np.linspace(0, nframe - 1,
                                 min(max_sample_frames, nframe), dtype=int))
    for it in take:
        cat = traj.xyz[it, cation_idx].astype(float)
        con = traj.xyz[it, contact_idx].astype(float)
        lists = neighbor_atom_lists(cat, con, traj.cells[it], rmax)
        inv_cell = np.linalg.inv(traj.cells[it])
        local_dist: List[float] = []
        for ili, ids in enumerate(lists):
            if ids.size:
                delta = mic_displacements(con[ids] - cat[ili], traj.cells[it], inv_cell)
                local_dist.extend(np.linalg.norm(delta, axis=1).tolist())
        counts += np.histogram(local_dist, bins=edges)[0]
        vol = abs(float(np.linalg.det(traj.cells[it])))
        density = len(contact_idx) / vol
        expected += len(cation_idx) * density * shell
    g = np.divide(counts, expected, out=np.zeros_like(counts), where=expected > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, g


def infer_first_minimum(r: np.ndarray, g: np.ndarray, bandwidth_a: float,
                        min_peak_r: float = 1.2) -> Tuple[float, np.ndarray]:
    dr = float(np.median(np.diff(r)))
    sigma_bins = max(bandwidth_a / dr, 0.5)
    smooth = gaussian_filter1d(g, sigma=sigma_bins, mode="nearest")
    valid = r >= min_peak_r
    peak_candidates, _ = find_peaks(smooth)
    peak_candidates = peak_candidates[valid[peak_candidates]]
    if peak_candidates.size == 0:
        raise ValueError("Could not find a first-shell RDF peak; provide an explicit r_on.")
    # Prefer the first substantial peak, not a tiny short-range numerical ripple.
    max_height = float(np.max(smooth[peak_candidates]))
    substantial = peak_candidates[smooth[peak_candidates] >= 0.10 * max_height]
    peak = int(substantial[0] if substantial.size else peak_candidates[0])
    minima, _ = find_peaks(-smooth)
    minima = minima[minima > peak]
    if minima.size == 0:
        raise ValueError("Could not find the RDF minimum after the first peak; provide r_on.")
    for idx in minima:
        if r[idx] - r[peak] >= max(2.0 * dr, 0.10):
            return float(r[idx]), smooth
    raise ValueError("Could not find a separated first RDF minimum; provide r_on.")


def intermittent_correlation(frames_by_pair: Dict[int, List[int]], nframe: int,
                             max_lag: int, block_size: int = 2048) -> np.ndarray:
    total_on = sum(len(v) for v in frames_by_pair.values())
    if total_on == 0:
        return np.full(max_lag + 1, np.nan)
    pair_lists = list(frames_by_pair.values())
    nfft = 1 << (2 * nframe - 1).bit_length()
    numer = np.zeros(max_lag + 1, dtype=float)
    for start in range(0, len(pair_lists), block_size):
        block = pair_lists[start:start + block_size]
        x = np.zeros((nframe, len(block)), dtype=np.float32)
        for j, ids in enumerate(block):
            x[np.asarray(ids, dtype=int), j] = 1.0
        fx = np.fft.rfft(x, n=nfft, axis=0)
        ac = np.fft.irfft(fx.conj() * fx, n=nfft, axis=0)[:max_lag + 1]
        numer += ac.sum(axis=1)
    lag = np.arange(max_lag + 1)
    return numer * nframe / ((nframe - lag) * total_on)


def continuous_correlation(run_lengths: Sequence[int], max_lag: int) -> np.ndarray:
    denom = float(sum(run_lengths))
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    lengths = np.asarray(run_lengths, dtype=int)
    return np.array([np.maximum(lengths - lag, 0).sum() / denom
                     for lag in range(max_lag + 1)], dtype=float)


def kaplan_meier(events: Sequence[dict], max_lag: int, dt_ns: float) -> np.ndarray:
    usable = [e for e in events if not e["left_censored"]]
    if not usable:
        return np.full(max_lag + 1, np.nan)
    durations = np.asarray([e["n_observed_frames"] for e in usable], dtype=int)
    observed = np.asarray([not e["right_censored"] for e in usable], dtype=bool)
    survival = np.ones(max_lag + 1, dtype=float)
    value = 1.0
    for lag in range(1, max_lag + 1):
        at_risk = int(np.count_nonzero(durations >= lag))
        deaths = int(np.count_nonzero((durations == lag) & observed))
        if at_risk > 0:
            value *= 1.0 - deaths / at_risk
        survival[lag] = value
    return survival


def scan_contacts(traj: Trajectory, cation_idx: Sequence[int],
                  contact_idx: Sequence[int], owner: np.ndarray,
                  nmol: int, r_on: float, r_off: float,
                  dt_ns: float, max_lag: int) -> dict:
    nframe, ncat = traj.xyz.shape[0], len(cation_idx)
    active: set[int] = set()
    starts: Dict[int, int] = {}
    frames_by_pair: Dict[int, List[int]] = defaultdict(list)
    events: List[dict] = []
    coordination = np.zeros((nframe, ncat), dtype=np.int16)
    formed = broken = 0

    def close_event(pid: int, end: int, right: bool) -> None:
        start = starts.pop(pid)
        nobs = end - start + 1
        low = max(nobs - 1, 0) * dt_ns
        high = None if (start == 0 or right) else (nobs + 1) * dt_ns
        events.append({
            "pair_id": pid,
            "cation_index": pid // nmol,
            "molecule_index": pid % nmol,
            "start_frame": start,
            "end_frame": end,
            "n_observed_frames": nobs,
            "duration_grid_ns": nobs * dt_ns,
            "duration_lower_bound_ns": low,
            "duration_upper_bound_ns": high,
            "left_censored": start == 0,
            "right_censored": right,
        })

    for it in range(nframe):
        candidates = candidate_molecule_distances(
            traj.xyz[it, cation_idx].astype(float),
            traj.xyz[it, contact_idx].astype(float), owner,
            traj.cells[it], r_off, nmol,
        )
        new_active: set[int] = set()
        for pid, dist in candidates.items():
            if (pid in active and dist <= r_off) or (pid not in active and dist <= r_on):
                new_active.add(pid)
        for pid in active - new_active:
            close_event(pid, it - 1, right=False)
            broken += 1
        for pid in new_active - active:
            starts[pid] = it
            formed += 1
        active = new_active
        for pid in active:
            frames_by_pair[pid].append(it)
            coordination[it, pid // nmol] += 1
    for pid in sorted(active):
        close_event(pid, nframe - 1, right=True)

    run_lengths = [e["n_observed_frames"] for e in events]
    cont = continuous_correlation(run_lengths, max_lag)
    inter = intermittent_correlation(frames_by_pair, nframe, max_lag)
    km = kaplan_meier(events, max_lag, dt_ns)
    return {
        "events": events,
        "coordination": coordination,
        "continuous": cont,
        "intermittent": inter,
        "kaplan_meier": km,
        "formed": formed,
        "broken": broken,
        "occupied_pair_frames": sum(len(v) for v in frames_by_pair.values()),
    }


def safe_label(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")
    return out or "trajectory"


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, base: Path, fmt: str, dpi: int) -> None:
    formats = ["png", "pdf"] if fmt == "both" else [fmt]
    for ext in formats:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def trapz_until(corr: np.ndarray, dt_ns: float) -> float:
    finite = np.isfinite(corr)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    last = np.flatnonzero(finite)[-1]
    y = corr[:last + 1]
    return float(0.5 * dt_ns * np.sum(y[:-1] + y[1:]))


def integrate_xy(time_ns: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(time_ns) & np.isfinite(values)
    x, y = time_ns[mask], values[mask]
    if len(x) < 2:
        return float("nan")
    return float(0.5 * np.sum(np.diff(x) * (y[:-1] + y[1:])))


def kww_curve(time_ns: np.ndarray, tau_ns: float, beta: float,
              plateau: float) -> np.ndarray:
    scaled = np.power(np.maximum(time_ns, 0.0) / max(tau_ns, 1.0e-30), beta)
    return plateau + (1.0 - plateau) * np.exp(-np.clip(scaled, 0.0, 700.0))


def first_crossing(time_ns: np.ndarray, values: np.ndarray,
                   level: float) -> float:
    """First downward crossing, linearly interpolated; NaN if not observed."""
    finite = np.isfinite(time_ns) & np.isfinite(values)
    x, y = time_ns[finite], values[finite]
    if len(x) == 0 or y[0] < level:
        return float("nan")
    hits = np.flatnonzero(y <= level)
    if len(hits) == 0:
        return float("nan")
    i = int(hits[0])
    if i == 0 or y[i] == y[i - 1]:
        return float(x[i])
    fraction = (level - y[i - 1]) / (y[i] - y[i - 1])
    return float(x[i - 1] + fraction * (x[i] - x[i - 1]))


def fit_kww_relaxation(time_ns: np.ndarray, correlation: np.ndarray,
                        allow_plateau: bool, tail_fraction: float,
                        convergence_fraction: float,
                        fit_max_ns: Optional[float], min_r_squared: float) -> dict:
    """Fit C(t)=Cinf+(1-Cinf)exp[-(t/tau)^beta].

    The reported restricted integral is evaluated only over the observed
    window.  If the fitted remaining fraction at the end exceeds the requested
    convergence fraction, it is explicitly marked as a lower bound.
    """
    mask = np.isfinite(time_ns) & np.isfinite(correlation)
    if fit_max_ns is not None:
        mask &= time_ns <= fit_max_ns + 1.0e-12
    x = np.asarray(time_ns[mask], dtype=float)
    y = np.asarray(correlation[mask], dtype=float)
    if len(x) < 6 or x[-1] <= 0:
        return {
            "fit_status": "fit_failed_too_few_points",
            "tau_scale_ns": float("nan"), "tau_scale_stderr_ns": float("nan"),
            "beta": float("nan"), "beta_stderr": float("nan"),
            "plateau": 0.0 if not allow_plateau else float("nan"),
            "plateau_stderr": float("nan"), "tau_mean_ns": float("nan"),
            "restricted_integral_ns": float("nan"),
            "integral_plateau_used": 0.0 if not allow_plateau else float("nan"),
            "remaining_fraction_at_fit_end": float("nan"),
            "t_half_observed_ns": float("nan"), "t_1e_observed_ns": float("nan"),
            "r_squared": float("nan"), "fit_end_ns": float(x[-1]) if len(x) else float("nan"),
            "fit_curve": np.full_like(time_ns, np.nan, dtype=float),
        }

    y = np.clip(y, 0.0, 1.05)
    ntail = max(3, int(math.ceil(len(y) * tail_fraction)))
    plateau0 = float(np.clip(np.median(y[-ntail:]), 0.0, 0.95)) if allow_plateau else 0.0
    normalized0 = np.clip((y - plateau0) / max(1.0 - plateau0, 1.0e-12), 0.0, 1.0)
    tau0 = first_crossing(x, normalized0, math.exp(-1.0))
    if not np.isfinite(tau0) or tau0 <= 0:
        tau0 = max(0.35 * x[-1], x[1] - x[0])

    try:
        if allow_plateau:
            def model(t: np.ndarray, tau: float, beta: float, plateau: float) -> np.ndarray:
                return kww_curve(t, tau, beta, plateau)

            popt, pcov = curve_fit(
                model, x, y, p0=[tau0, 0.7, plateau0],
                bounds=([max((x[1] - x[0]) * 0.05, 1.0e-12), 0.10, 0.0],
                        [max(100.0 * x[-1], x[1] - x[0]), 2.0, 0.999]),
                maxfev=50000,
            )
            tau, beta, plateau = [float(v) for v in popt]
            stderr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
            tau_err, beta_err, plateau_err = [float(v) for v in stderr]
        else:
            def model(t: np.ndarray, tau: float, beta: float) -> np.ndarray:
                return kww_curve(t, tau, beta, 0.0)

            popt, pcov = curve_fit(
                model, x, y, p0=[tau0, 0.7],
                bounds=([max((x[1] - x[0]) * 0.05, 1.0e-12), 0.10],
                        [max(100.0 * x[-1], x[1] - x[0]), 2.0]),
                maxfev=50000,
            )
            tau, beta = [float(v) for v in popt]
            plateau = 0.0
            stderr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
            tau_err, beta_err = [float(v) for v in stderr]
            plateau_err = 0.0
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return {
            "fit_status": "fit_failed_optimization",
            "tau_scale_ns": float("nan"), "tau_scale_stderr_ns": float("nan"),
            "beta": float("nan"), "beta_stderr": float("nan"),
            "plateau": plateau0, "plateau_stderr": float("nan"),
            "tau_mean_ns": float("nan"), "restricted_integral_ns": float("nan"),
            "integral_plateau_used": plateau0,
            "remaining_fraction_at_fit_end": float("nan"),
            "t_half_observed_ns": float("nan"), "t_1e_observed_ns": float("nan"),
            "r_squared": float("nan"), "fit_end_ns": float(x[-1]),
            "fit_curve": np.full_like(time_ns, np.nan, dtype=float),
        }

    fitted_x = kww_curve(x, tau, beta, plateau)
    residual = float(np.sum((y - fitted_x) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    remaining = float(math.exp(-min((x[-1] / tau) ** beta, 700.0)))
    if np.isfinite(r_squared) and r_squared < min_r_squared:
        status = "fit_poor_r_squared"
    else:
        status = ("relaxed_within_fit_window" if remaining <= convergence_fraction
                  else "lower_bound_window_limited")
    integral_plateau = plateau if status != "fit_poor_r_squared" else plateau0
    normalized = np.clip(
        (y - integral_plateau) / max(1.0 - integral_plateau, 1.0e-12), 0.0, 1.0
    )
    restricted = integrate_xy(x, normalized)
    tau_mean = float(tau * math.gamma(1.0 + 1.0 / beta))
    return {
        "fit_status": status,
        "tau_scale_ns": tau,
        "tau_scale_stderr_ns": tau_err,
        "beta": beta,
        "beta_stderr": beta_err,
        "plateau": plateau,
        "plateau_stderr": plateau_err,
        "tau_mean_ns": tau_mean,
        "restricted_integral_ns": restricted,
        "integral_plateau_used": integral_plateau,
        "remaining_fraction_at_fit_end": remaining,
        "t_half_observed_ns": first_crossing(x, normalized, 0.5),
        "t_1e_observed_ns": first_crossing(x, normalized, math.exp(-1.0)),
        "r_squared": r_squared,
        "fit_end_ns": float(x[-1]),
        "fit_curve": kww_curve(np.asarray(time_ns, dtype=float), tau, beta, plateau),
    }


def km_restricted_mean(time_ns: np.ndarray, survival: np.ndarray,
                       convergence_fraction: float) -> dict:
    rmst = integrate_xy(time_ns, np.clip(survival, 0.0, 1.0))
    finite = survival[np.isfinite(survival)]
    end = float(finite[-1]) if len(finite) else float("nan")
    status = ("relaxed_within_window" if np.isfinite(end) and end <= convergence_fraction
              else "lower_bound_window_limited")
    return {
        "rmst_ns": rmst,
        "survival_at_end": end,
        "status": status,
        "t_median_ns": first_crossing(time_ns, survival, 0.5),
        "t_1e_ns": first_crossing(time_ns, survival, math.exp(-1.0)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Residence-time pilot for multiple coarse-sampled PDB trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdbs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--dt-ns", type=float, required=True)
    p.add_argument("--target-cation", default="Li")
    p.add_argument("--target-anion", required=True,
                   help="LABEL:ANCHOR_ATOM_TYPE[:CHARGE], e.g. FSA:72:-1")
    p.add_argument("--target-solvent",
                   help="LABEL:ANCHOR_ATOM_TYPE, e.g. SN:71")
    p.add_argument("--atom-type-field", choices=["resname", "name", "element"],
                   default="resname")
    p.add_argument("--anion-contact-elements", default="O")
    p.add_argument("--solvent-contact-elements", default="N")
    p.add_argument("--anion-r-on-a", type=float)
    p.add_argument("--anion-r-off-a", type=float)
    p.add_argument("--solvent-r-on-a", type=float)
    p.add_argument("--solvent-r-off-a", type=float)
    p.add_argument("--hysteresis-width-a", type=float, default=0.30)
    p.add_argument("--rdf-rmax-a", type=float, default=8.0)
    p.add_argument("--bins", type=int, default=180)
    p.add_argument("--kde-bandwidth-a", type=float, default=0.08,
                   help="Gaussian smoothing width for radial distributions (Angstrom).")
    p.add_argument("--kde-bandwidth-deg", type=float,
                   help="Accepted only for command compatibility; ignored because this is not angular data.")
    p.add_argument("--cutoff-sample-frames", type=int, default=200)
    p.add_argument("--max-lag-ns", type=float)
    p.add_argument("--relax-fit-max-ns", type=float,
                   help="Upper time used for KWW relaxation fitting; defaults to max lag.")
    p.add_argument("--relax-tail-fraction", type=float, default=0.10,
                   help="Tail fraction used only for intermittent-plateau initialization.")
    p.add_argument("--relax-convergence-fraction", type=float, default=0.05,
                   help="Maximum fitted unrelaxed fraction accepted as converged.")
    p.add_argument("--relax-min-r2", type=float, default=0.90,
                   help="Minimum KWW fit R-squared accepted for reporting fitted relaxation.")
    p.add_argument("--format", choices=["png", "pdf", "both"], default="png")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--outdir", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.dt_ns <= 0:
        raise SystemExit("--dt-ns must be positive.")
    if args.bins < 30:
        raise SystemExit("--bins must be at least 30.")
    if not 0.02 <= args.relax_tail_fraction <= 0.50:
        raise SystemExit("--relax-tail-fraction must be between 0.02 and 0.50.")
    if not 0.001 <= args.relax_convergence_fraction <= 0.25:
        raise SystemExit("--relax-convergence-fraction must be between 0.001 and 0.25.")
    if not 0.0 <= args.relax_min_r2 < 1.0:
        raise SystemExit("--relax-min-r2 must be in [0, 1).")
    if args.relax_fit_max_ns is not None and args.relax_fit_max_ns <= 0:
        raise SystemExit("--relax-fit-max-ns must be positive.")
    if args.kde_bandwidth_deg is not None:
        print("[warning] --kde-bandwidth-deg is ignored. Use --kde-bandwidth-a for RDF smoothing.",
              file=sys.stderr)
    pdbs = [Path(x) for x in args.pdbs]
    labels = args.labels or [p.stem for p in pdbs]
    if len(labels) != len(pdbs):
        raise SystemExit("--labels must have the same number of entries as --pdbs.")
    for path in pdbs:
        if not path.is_file():
            raise SystemExit(f"PDB not found: {path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    anion = parse_molecular_target(args.target_anion, "anion")
    solvent = parse_molecular_target(args.target_solvent, "solvent") if args.target_solvent else None
    target_specs = [(anion, args.anion_contact_elements,
                     args.anion_r_on_a, args.anion_r_off_a)]
    if solvent:
        target_specs.append((solvent, args.solvent_contact_elements,
                             args.solvent_r_on_a, args.solvent_r_off_a))

    summary_rows: List[dict] = []
    relaxation_rows: List[dict] = []
    corr_rows: List[dict] = []
    coord_rows: List[dict] = []
    rdf_rows: List[dict] = []
    plot_corr: Dict[str, Dict[str, dict]] = defaultdict(dict)
    plot_rdf: Dict[str, Dict[str, dict]] = defaultdict(dict)
    heatmaps: List[Tuple[str, np.ndarray]] = []
    metadata = {
        "script": Path(__file__).name,
        "version": VERSION,
        "sampling_warning": (
            "Reported lifetimes are coarse-grained. Events faster than one frame interval "
            "are unresolved; use duration bounds and censor flags."
        ),
        "dt_ns": args.dt_ns,
        "relaxation_analysis": {
            "model": "KWW: C(t)=Cinf+(1-Cinf)*exp[-(t/tau)^beta]",
            "continuous_plateau": 0.0,
            "intermittent_plateau": "fitted",
            "convergence_fraction": args.relax_convergence_fraction,
            "minimum_r_squared": args.relax_min_r2,
            "fit_max_ns": args.relax_fit_max_ns,
            "interpretation": (
                "restricted integrals are lower bounds when the fitted remaining "
                "fraction exceeds the convergence threshold"
            ),
        },
        "inputs": [],
    }

    for ipdb, (path, label) in enumerate(zip(pdbs, labels), start=1):
        print(f"[{ipdb}/{len(pdbs)}] Reading {label}: {path}", flush=True)
        traj = read_pdb(path)
        nframe = traj.xyz.shape[0]
        if nframe < 3:
            raise SystemExit(f"{path} contains only {nframe} frame(s); at least 3 are needed.")
        total_span = (nframe - 1) * args.dt_ns
        if args.max_lag_ns is None:
            max_lag = max(1, (nframe - 1) // 2)
        else:
            max_lag = min(nframe - 1, int(math.floor(args.max_lag_ns / args.dt_ns + 1e-12)))
            if max_lag < 1:
                raise SystemExit("--max-lag-ns must be at least one frame interval.")
        cat_idx = select_cations(traj.atoms, args.target_cation)
        per_target: Dict[str, dict] = {}
        metadata["inputs"].append({
            "pdb": str(path), "label": label, "frames": nframe,
            "atoms": len(traj.atoms), "cations": len(cat_idx),
            "trajectory_span_ns": total_span,
        })

        for target, contact_text, r_on_user, r_off_user in target_specs:
            contact_elements = [x for x in re.split(r"[,;\s]+", contact_text) if x]
            molecules, contact_idx, owner = build_molecules(
                traj.atoms, target, args.atom_type_field, contact_elements
            )
            r, g = radial_distribution(traj, cat_idx, contact_idx, args.bins,
                                       args.rdf_rmax_a, args.cutoff_sample_frames)
            if r_on_user is None:
                try:
                    r_on, smooth = infer_first_minimum(r, g, args.kde_bandwidth_a)
                except ValueError as exc:
                    raise SystemExit(
                        f"{label}/{target.label}: {exc} For example, set "
                        f"--{('anion' if target is anion else 'solvent')}-r-on-a."
                    ) from exc
                cutoff_source = "RDF_first_minimum"
            else:
                r_on = r_on_user
                dr = float(np.median(np.diff(r)))
                smooth = gaussian_filter1d(g, sigma=max(args.kde_bandwidth_a / dr, 0.5))
                cutoff_source = "user"
            r_off = r_off_user if r_off_user is not None else r_on + args.hysteresis_width_a
            if r_off < r_on:
                raise SystemExit(f"{label}/{target.label}: r_off must be >= r_on.")
            print(
                f"  {target.label}: {len(molecules)} molecules, contact atoms={len(contact_idx)}, "
                f"r_on={r_on:.3f} A, r_off={r_off:.3f} A ({cutoff_source})",
                flush=True,
            )
            scan = scan_contacts(traj, cat_idx, contact_idx, owner, len(molecules),
                                 r_on, r_off, args.dt_ns, max_lag)
            per_target[target.label] = scan
            tag = safe_label(label)
            event_fields = [
                "pair_id", "cation_index", "molecule_index", "start_frame", "end_frame",
                "n_observed_frames", "duration_grid_ns", "duration_lower_bound_ns",
                "duration_upper_bound_ns", "left_censored", "right_censored",
            ]
            write_csv(outdir / f"{tag}_{safe_label(target.label)}_residence_events.csv",
                      scan["events"], event_fields)

            lag_ns = np.arange(max_lag + 1) * args.dt_ns
            continuous_fit = fit_kww_relaxation(
                lag_ns, scan["continuous"], allow_plateau=False,
                tail_fraction=args.relax_tail_fraction,
                convergence_fraction=args.relax_convergence_fraction,
                fit_max_ns=args.relax_fit_max_ns,
                min_r_squared=args.relax_min_r2,
            )
            intermittent_fit = fit_kww_relaxation(
                lag_ns, scan["intermittent"], allow_plateau=True,
                tail_fraction=args.relax_tail_fraction,
                convergence_fraction=args.relax_convergence_fraction,
                fit_max_ns=args.relax_fit_max_ns,
                min_r_squared=args.relax_min_r2,
            )
            km_stats = km_restricted_mean(
                lag_ns, scan["kaplan_meier"], args.relax_convergence_fraction
            )
            scan["relaxation_fits"] = {
                "continuous": continuous_fit,
                "intermittent": intermittent_fit,
                "kaplan_meier": km_stats,
            }
            for i, t in enumerate(lag_ns):
                corr_rows.append({
                    "label": label, "target": target.label, "lag_ns": t,
                    "continuous_survival": scan["continuous"][i],
                    "intermittent_correlation": scan["intermittent"][i],
                    "kaplan_meier_survival": scan["kaplan_meier"][i],
                    "continuous_kww_fit": continuous_fit["fit_curve"][i],
                    "intermittent_kww_fit": intermittent_fit["fit_curve"][i],
                })
            for correlation_name, fit in [
                    ("continuous", continuous_fit),
                    ("intermittent", intermittent_fit)]:
                relaxation_rows.append({
                    "label": label,
                    "target": target.label,
                    "correlation": correlation_name,
                    "fit_model": "KWW",
                    "fit_status": fit["fit_status"],
                    "fit_end_ns": fit["fit_end_ns"],
                    "tau_scale_ns": fit["tau_scale_ns"],
                    "tau_scale_stderr_ns": fit["tau_scale_stderr_ns"],
                    "beta": fit["beta"],
                    "beta_stderr": fit["beta_stderr"],
                    "plateau": fit["plateau"],
                    "plateau_stderr": fit["plateau_stderr"],
                    "tau_mean_ns": fit["tau_mean_ns"],
                    "restricted_integral_ns": fit["restricted_integral_ns"],
                    "integral_plateau_used": fit["integral_plateau_used"],
                    "remaining_fraction_at_fit_end": fit["remaining_fraction_at_fit_end"],
                    "t_half_observed_ns": fit["t_half_observed_ns"],
                    "t_1e_observed_ns": fit["t_1e_observed_ns"],
                    "r_squared": fit["r_squared"],
                })
            relaxation_rows.append({
                "label": label,
                "target": target.label,
                "correlation": "kaplan_meier",
                "fit_model": "none",
                "fit_status": km_stats["status"],
                "fit_end_ns": lag_ns[-1],
                "tau_scale_ns": float("nan"),
                "tau_scale_stderr_ns": float("nan"),
                "beta": float("nan"),
                "beta_stderr": float("nan"),
                "plateau": 0.0,
                "plateau_stderr": float("nan"),
                "tau_mean_ns": float("nan"),
                "restricted_integral_ns": km_stats["rmst_ns"],
                "integral_plateau_used": 0.0,
                "remaining_fraction_at_fit_end": km_stats["survival_at_end"],
                "t_half_observed_ns": km_stats["t_median_ns"],
                "t_1e_observed_ns": km_stats["t_1e_ns"],
                "r_squared": float("nan"),
            })
            for ri, gi, si in zip(r, g, smooth):
                rdf_rows.append({
                    "label": label, "target": target.label, "r_A": ri,
                    "g_r": gi, "g_r_smoothed": si, "r_on_A": r_on, "r_off_A": r_off,
                })
            plot_corr[target.label][label] = {"lag": lag_ns, **scan}
            plot_rdf[target.label][label] = {
                "r": r, "g": g, "smooth": smooth, "r_on": r_on, "r_off": r_off,
            }
            durations_complete = [e["duration_grid_ns"] for e in scan["events"]
                                  if not e["left_censored"] and not e["right_censored"]]
            total_pair_frames = nframe * len(cat_idx) * len(molecules)
            summary_rows.append({
                "label": label,
                "target": target.label,
                "frames": nframe,
                "dt_ns": args.dt_ns,
                "trajectory_span_ns": total_span,
                "n_cations": len(cat_idx),
                "n_molecules": len(molecules),
                "r_on_A": r_on,
                "r_off_A": r_off,
                "cutoff_source": cutoff_source,
                "contact_occupancy_fraction": scan["occupied_pair_frames"] / total_pair_frames,
                "n_events": len(scan["events"]),
                "n_complete_events": len(durations_complete),
                "n_left_censored": sum(e["left_censored"] for e in scan["events"]),
                "n_right_censored": sum(e["right_censored"] for e in scan["events"]),
                "formed_transitions": scan["formed"],
                "broken_transitions": scan["broken"],
                "mean_coordination": float(np.mean(scan["coordination"])),
                "median_complete_duration_grid_ns": (
                    float(np.median(durations_complete)) if durations_complete else float("nan")
                ),
                "mean_complete_duration_grid_ns": (
                    float(np.mean(durations_complete)) if durations_complete else float("nan")
                ),
                "continuous_integral_ns": trapz_until(scan["continuous"], args.dt_ns),
                "intermittent_integral_raw_ns": trapz_until(scan["intermittent"], args.dt_ns),
                "continuous_tau_scale_kww_ns": continuous_fit["tau_scale_ns"],
                "continuous_beta_kww": continuous_fit["beta"],
                "continuous_tau_mean_kww_ns": continuous_fit["tau_mean_ns"],
                "continuous_restricted_integral_ns": continuous_fit["restricted_integral_ns"],
                "continuous_relaxation_status": continuous_fit["fit_status"],
                "intermittent_tau_scale_kww_ns": intermittent_fit["tau_scale_ns"],
                "intermittent_beta_kww": intermittent_fit["beta"],
                "intermittent_plateau_kww": intermittent_fit["plateau"],
                "intermittent_tau_mean_kww_ns": intermittent_fit["tau_mean_ns"],
                "intermittent_excess_restricted_integral_ns": intermittent_fit["restricted_integral_ns"],
                "intermittent_relaxation_status": intermittent_fit["fit_status"],
                "km_restricted_mean_survival_ns": km_stats["rmst_ns"],
                "km_relaxation_status": km_stats["status"],
            })
            def relaxation_text(fit: dict) -> str:
                if fit["fit_status"] == "relaxed_within_fit_window":
                    return f"KWW mean tau={fit['tau_mean_ns']:.4g} ns"
                if fit["fit_status"] == "lower_bound_window_limited":
                    return f"restricted tau >= {fit['restricted_integral_ns']:.4g} ns"
                if fit["fit_status"] == "fit_poor_r_squared":
                    return f"KWW rejected; restricted integral={fit['restricted_integral_ns']:.4g} ns"
                return fit["fit_status"]

            print(
                f"    relaxation: continuous {relaxation_text(continuous_fit)}; "
                f"intermittent {relaxation_text(intermittent_fit)}; "
                f"KM RMST={km_stats['rmst_ns']:.4g} ns ({km_stats['status']})",
                flush=True,
            )

        an_coord = per_target[anion.label]["coordination"]
        if solvent:
            sol_coord = per_target[solvent.label]["coordination"]
        else:
            sol_coord = np.zeros_like(an_coord)
        pairs, counts = np.unique(np.column_stack([an_coord.ravel(), sol_coord.ravel()]),
                                  axis=0, return_counts=True)
        probs = counts / counts.sum()
        for (na, ns), count, prob in zip(pairs, counts, probs):
            coord_rows.append({
                "label": label, f"n_{anion.label}": int(na),
                f"n_{solvent.label if solvent else 'solvent'}": int(ns),
                "count": int(count), "probability": float(prob),
            })
        matrix = np.zeros((int(pairs[:, 0].max()) + 1, int(pairs[:, 1].max()) + 1))
        for (na, ns), prob in zip(pairs, probs):
            matrix[int(na), int(ns)] = prob
        heatmaps.append((label, matrix))

        # Checkpoint completed concentrations so a later input failure does not
        # discard already finished tabular analyses.
        write_csv(outdir / "residence_summary.csv", summary_rows,
                  list(summary_rows[0].keys()))
        write_csv(outdir / "residence_relaxation_times.csv", relaxation_rows,
                  list(relaxation_rows[0].keys()))
        write_csv(outdir / "residence_correlations.csv", corr_rows,
                  list(corr_rows[0].keys()))
        write_csv(outdir / "coordination_state_probabilities.csv", coord_rows,
                  list(coord_rows[0].keys()))
        write_csv(outdir / "contact_rdf_and_cutoffs.csv", rdf_rows,
                  list(rdf_rows[0].keys()))
        with (outdir / "analysis_metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)

    summary_fields = list(summary_rows[0].keys())
    relaxation_fields = list(relaxation_rows[0].keys())
    corr_fields = list(corr_rows[0].keys())
    coord_fields = list(coord_rows[0].keys())
    rdf_fields = list(rdf_rows[0].keys())
    write_csv(outdir / "residence_summary.csv", summary_rows, summary_fields)
    write_csv(outdir / "residence_relaxation_times.csv", relaxation_rows,
              relaxation_fields)
    write_csv(outdir / "residence_correlations.csv", corr_rows, corr_fields)
    write_csv(outdir / "coordination_state_probabilities.csv", coord_rows, coord_fields)
    write_csv(outdir / "contact_rdf_and_cutoffs.csv", rdf_rows, rdf_fields)
    with (outdir / "analysis_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    # Comparison figures.
    targets = list(plot_corr)
    all_labels = list(plot_corr[targets[0]].keys())
    cmap = plt.get_cmap("tab10")
    label_colors = {label: cmap(i % 10) for i, label in enumerate(all_labels)}
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.8), squeeze=False)
    for ax, target in zip(axes[0], targets):
        for label, data in plot_corr[target].items():
            color = label_colors[label]
            ax.plot(data["lag"], data["continuous"], color=color,
                    label=f"{label}, continuous")
            ax.plot(data["lag"], data["intermittent"], "--", color=color,
                    label=f"{label}, intermittent")
        ax.set(xlabel="Lag time (ns)", ylabel="Normalized correlation",
               title=f"Li-{target} residence")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.legend(fontsize=8)
    save_figure(fig, outdir / "residence_survival_comparison", args.format, args.dpi)

    fig, axes = plt.subplots(len(targets), 2, figsize=(12.4, 4.4 * len(targets)),
                             squeeze=False)
    for irow, target in enumerate(targets):
        for icol, correlation_name in enumerate(["continuous", "intermittent"]):
            ax = axes[irow, icol]
            for label, data in plot_corr[target].items():
                color = label_colors[label]
                observed = data[correlation_name]
                fit = data["relaxation_fits"][correlation_name]
                ax.plot(data["lag"], observed, color=color, lw=1.5, alpha=0.65)
                if np.all(np.isnan(fit["fit_curve"])):
                    legend_label = f"{label}: fit failed"
                elif fit["fit_status"] == "relaxed_within_fit_window":
                    legend_label = f"{label}: mean tau={fit['tau_mean_ns']:.3g} ns"
                elif fit["fit_status"] == "fit_poor_r_squared":
                    legend_label = (
                        f"{label}: KWW rejected; restricted={fit['restricted_integral_ns']:.3g} ns"
                    )
                else:
                    legend_label = (
                        f"{label}: restricted tau >= {fit['restricted_integral_ns']:.3g} ns"
                    )
                ax.plot(data["lag"], fit["fit_curve"], "--", color=color,
                        lw=1.3, label=legend_label)
            ax.set(xlabel="Lag time (ns)", ylabel="Normalized correlation",
                   title=f"Li-{target} {correlation_name} relaxation")
            ax.set_ylim(-0.03, 1.05)
            ax.legend(fontsize=8)
    save_figure(fig, outdir / "residence_relaxation_kww_fits", args.format, args.dpi)

    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.8), squeeze=False)
    for ax, target in zip(axes[0], targets):
        for label, data in plot_rdf[target].items():
            color = label_colors[label]
            ax.plot(data["r"], data["smooth"], color=color, label=label)
            ax.axvline(data["r_on"], color=color, ls="--", lw=0.9)
        ax.set(xlabel=r"$r$ ($\AA$)", ylabel=r"$g(r)$",
               title=f"Li-{target} cutoff diagnostic")
        ax.set_xlim(0, args.rdf_rmax_a)
        ax.legend(fontsize=8)
    save_figure(fig, outdir / "contact_rdf_cutoff_diagnostic", args.format, args.dpi)

    fig, axes = plt.subplots(1, len(heatmaps), figsize=(4.8 * len(heatmaps), 4.2), squeeze=False)
    common_vmax = max(float(np.max(matrix)) for _, matrix in heatmaps)
    common_n_fsa = max(matrix.shape[0] for _, matrix in heatmaps)
    common_n_solvent = max(matrix.shape[1] for _, matrix in heatmaps)
    last_im = None
    for ax, (label, matrix) in zip(axes[0], heatmaps):
        last_im = ax.imshow(matrix.T, origin="lower", aspect="auto", cmap="viridis",
                            vmin=0.0, vmax=common_vmax)
        ax.set_xlabel(f"$n_{{{anion.label}}}$")
        ax.set_ylabel(f"$n_{{{solvent.label if solvent else 'solvent'}}}$")
        ax.set_title(label)
        ax.set_xlim(-0.5, common_n_fsa - 0.5)
        ax.set_ylim(-0.5, common_n_solvent - 0.5)
    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes[0]), label="Probability", shrink=0.9)
    save_figure(fig, outdir / "coordination_state_probability", args.format, args.dpi)

    print(f"[OK] Residence-time pilot completed: {outdir}")
    print("[interpretation] Values below one frame interval are unresolved; inspect censor flags and duration bounds.")


if __name__ == "__main__":
    main()
