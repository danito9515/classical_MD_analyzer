#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust multi-PDB RDF analysis for ordinary atomic PDB trajectories and
CP2K ion + maximally localized Wannier-center (MLWC/WC) trajectories.

Key points
----------
* Does not rely on the MDAnalysis PDB reader.  This avoids failures caused by
  non-standard multi-model separators, blank element fields for WC, and PDB
  fixed-column deviations.
* Treats pseudo-sites such as WC as ordinary RDF sites.
* Applies the minimum-image convention, so wrapped and unwrapped trajectories
  give the same RDF when the correct periodic cell is supplied.
* Accepts MODEL/ENDMDL trajectories, repeated CRYST1 blocks, repeated END
  records, and trajectories whose atom serial number restarts at 1.

Example
-------
python3 pdb2RDF_analysis_v1.1.0.py \
  --pdbs ./T700K/*pdb ./T900K/*pdb ./T1100K/*pdb \
  --labels 700K 900K 1100K \
  --box 8.78765 8.78765 12.65755 \
  --pairs Li-WC S-WC Ge-S P-S Li-S \
  --xlim 0 6 --start-ratio 0.5
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
}


@dataclass
class PDBTrajectory:
    labels: list[str]
    positions: list[np.ndarray]
    boxes: list[Optional[np.ndarray]]
    source: Path

    @property
    def n_frames(self) -> int:
        return len(self.positions)

    @property
    def n_sites(self) -> int:
        return len(self.labels)


def normalize_site_token(value: object) -> str:
    """Normalize real element names and pseudo-site labels such as WC."""
    token = re.sub(r"[^A-Za-z]", "", str(value).strip())
    if not token:
        return ""

    if len(token) >= 2:
        candidate = token[:2].capitalize()
        if candidate in ELEMENTS:
            return candidate

    candidate = token[0].upper()
    if candidate in ELEMENTS and len(token) == 1:
        return candidate

    # For non-elements preserve a compact uppercase pseudo-site label.
    return token.upper()


def parse_pair(pair_str: str) -> tuple[str, str]:
    if "-" in pair_str:
        left, right = pair_str.split("-", 1)
    elif "_" in pair_str:
        left, right = pair_str.split("_", 1)
    else:
        raise ValueError(f"Pair should be like Li-WC or Li_WC: {pair_str}")

    left = normalize_site_token(left)
    right = normalize_site_token(right)
    if not left or not right:
        raise ValueError(f"Invalid pair token(s): {pair_str}")
    return left, right


def safe_name(value: object) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(value))


def parse_cryst1(line: str) -> Optional[np.ndarray]:
    """Parse CRYST1 as [a,b,c,alpha,beta,gamma]."""
    try:
        values = [
            float(line[6:15]),
            float(line[15:24]),
            float(line[24:33]),
            float(line[33:40]),
            float(line[40:47]),
            float(line[47:54]),
        ]
        if min(values[:3]) > 0.0:
            return np.asarray(values, dtype=float)
    except (ValueError, IndexError):
        pass

    fields = line.split()
    if len(fields) >= 7:
        try:
            values = [float(x) for x in fields[1:7]]
            if min(values[:3]) > 0.0:
                return np.asarray(values, dtype=float)
        except ValueError:
            pass
    return None


def parse_atom_line(line: str) -> tuple[Optional[int], str, np.ndarray]:
    """Parse ATOM/HETATM line using fixed columns, then whitespace fallback."""
    serial: Optional[int] = None
    atom_name = ""
    element = ""
    xyz: Optional[np.ndarray] = None

    try:
        serial_text = line[6:11].strip()
        serial = int(serial_text) if serial_text else None
    except (ValueError, IndexError):
        serial = None

    try:
        atom_name = line[12:16].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        xyz = np.asarray([x, y, z], dtype=float)
        if len(line) >= 78:
            element = line[76:78].strip()
    except (ValueError, IndexError):
        xyz = None

    fields = line.split()

    if not atom_name and len(fields) >= 3:
        atom_name = fields[2]

    if serial is None and len(fields) >= 2:
        try:
            serial = int(fields[1])
        except ValueError:
            serial = None

    if xyz is None:
        # Standard whitespace layout:
        # ATOM serial name resname chain resid x y z occupancy temp [element]
        candidates = []
        if len(fields) >= 9:
            candidates.append((6, 7, 8))
        # Layout without chain identifier:
        if len(fields) >= 8:
            candidates.append((5, 6, 7))

        for ix, iy, iz in candidates:
            try:
                xyz = np.asarray(
                    [float(fields[ix]), float(fields[iy]), float(fields[iz])],
                    dtype=float,
                )
                break
            except (ValueError, IndexError):
                continue

    if xyz is None:
        # Last-resort: coordinates are usually the first run of three floats
        # after the atom/residue metadata. Start at field 5 to avoid serials.
        for i in range(5, max(5, len(fields) - 2)):
            try:
                trial = np.asarray(
                    [float(fields[i]), float(fields[i + 1]), float(fields[i + 2])],
                    dtype=float,
                )
            except (ValueError, IndexError):
                continue
            xyz = trial
            break

    if xyz is None:
        raise ValueError(f"Could not parse coordinates from PDB line:\n{line.rstrip()}")

    site = normalize_site_token(element)
    if not site:
        site = normalize_site_token(atom_name)
    if not site:
        raise ValueError(f"Could not identify atom/WC label from line:\n{line.rstrip()}")

    return serial, site, xyz


def read_pdb_trajectory(path: Path, user_box: Optional[Iterable[float]] = None) -> PDBTrajectory:
    """Read a potentially non-standard multi-frame PDB trajectory."""
    path = Path(path)

    manual_box: Optional[np.ndarray] = None
    if user_box is not None:
        box_values = list(user_box)
        if len(box_values) == 3:
            manual_box = np.asarray(
                [box_values[0], box_values[1], box_values[2], 90.0, 90.0, 90.0],
                dtype=float,
            )
        elif len(box_values) == 6:
            manual_box = np.asarray(box_values, dtype=float)
        else:
            raise ValueError("user_box must contain 3 or 6 values")

    frames: list[np.ndarray] = []
    frame_labels: list[list[str]] = []
    boxes: list[Optional[np.ndarray]] = []

    current_xyz: list[np.ndarray] = []
    current_labels: list[str] = []
    current_box: Optional[np.ndarray] = manual_box.copy() if manual_box is not None else None
    last_cryst1: Optional[np.ndarray] = current_box.copy() if current_box is not None else None
    inside_model = False
    previous_serial: Optional[int] = None

    def flush_frame(reason: str) -> None:
        nonlocal current_xyz, current_labels, current_box, previous_serial
        if not current_xyz:
            previous_serial = None
            return
        frames.append(np.asarray(current_xyz, dtype=float))
        frame_labels.append(list(current_labels))
        boxes.append(
            manual_box.copy()
            if manual_box is not None
            else (None if current_box is None else current_box.copy())
        )
        current_xyz = []
        current_labels = []
        current_box = manual_box.copy() if manual_box is not None else (
            None if last_cryst1 is None else last_cryst1.copy()
        )
        previous_serial = None

    with path.open("r", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = line[:6].strip().upper()

            if record == "CRYST1":
                parsed_box = parse_cryst1(line)
                if parsed_box is not None:
                    last_cryst1 = parsed_box

                # A repeated CRYST1 after coordinates is frequently the actual
                # frame delimiter in CP2K-derived PDB trajectories.
                if current_xyz and not inside_model:
                    flush_frame(f"CRYST1 at line {line_number}")

                if manual_box is None:
                    current_box = None if parsed_box is None else parsed_box.copy()
                continue

            if record == "MODEL":
                if current_xyz:
                    flush_frame(f"MODEL at line {line_number}")
                inside_model = True
                if manual_box is None and last_cryst1 is not None:
                    current_box = last_cryst1.copy()
                continue

            if record in {"ENDMDL", "END"}:
                if current_xyz:
                    flush_frame(f"{record} at line {line_number}")
                if record == "ENDMDL":
                    inside_model = False
                continue

            if record not in {"ATOM", "HETATM"}:
                continue

            try:
                serial, site, xyz = parse_atom_line(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

            # Some generated trajectories have neither MODEL nor ENDMDL; atom
            # serials simply restart at 1 for each frame.
            if (
                current_xyz
                and not inside_model
                and serial is not None
                and previous_serial is not None
                and serial <= previous_serial
                and serial == 1
            ):
                flush_frame(f"serial restart at line {line_number}")

            current_xyz.append(xyz)
            current_labels.append(site)
            previous_serial = serial

    flush_frame("EOF")

    if not frames:
        raise RuntimeError(f"No ATOM/HETATM coordinates found in {path}")

    reference_labels = frame_labels[0]
    reference_count = len(reference_labels)

    for frame_index, (labels, xyz) in enumerate(zip(frame_labels, frames)):
        if len(labels) != reference_count:
            raise RuntimeError(
                f"Inconsistent site count in {path}: frame 0 has {reference_count}, "
                f"frame {frame_index} has {len(labels)}. Check frame delimiters or incomplete output."
            )
        if labels != reference_labels:
            mismatch = next(
                (i for i, (a, b) in enumerate(zip(reference_labels, labels)) if a != b),
                None,
            )
            raise RuntimeError(
                f"Site ordering changes in {path}, frame {frame_index}, site {mismatch}. "
                "RDF trajectories must keep a constant atom/WC ordering."
            )
        if xyz.shape != (reference_count, 3):
            raise RuntimeError(
                f"Unexpected coordinate shape in {path}, frame {frame_index}: {xyz.shape}"
            )

    return PDBTrajectory(
        labels=reference_labels,
        positions=frames,
        boxes=boxes,
        source=path,
    )


def cell_matrix_from_dimensions(dimensions: np.ndarray) -> np.ndarray:
    """Return a 3x3 row-vector cell matrix from PDB cell dimensions."""
    a, b, c, alpha_deg, beta_deg, gamma_deg = map(float, dimensions)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("Invalid cell: sin(gamma) is zero")

    va = np.asarray([a, 0.0, 0.0], dtype=float)
    vb = np.asarray([b * math.cos(gamma), b * sin_gamma, 0.0], dtype=float)

    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq < -1.0e-8:
        raise ValueError(f"Invalid triclinic cell; c_z^2 = {cz_sq}")
    vc = np.asarray([cx, cy, math.sqrt(max(0.0, cz_sq))], dtype=float)

    return np.vstack([va, vb, vc])


def minimum_image_displacements(
    pos1: np.ndarray,
    pos2: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    """Pairwise minimum-image displacement vectors for a triclinic cell."""
    cell = cell_matrix_from_dimensions(dimensions)
    inv_cell = np.linalg.inv(cell)

    displacement = pos1[:, None, :] - pos2[None, :, :]
    fractional = displacement @ inv_cell
    fractional -= np.rint(fractional)
    return fractional @ cell


def box_volume(dimensions: np.ndarray) -> float:
    return abs(float(np.linalg.det(cell_matrix_from_dimensions(dimensions))))


def indices_for_site(labels: list[str], site: str) -> np.ndarray:
    normalized = normalize_site_token(site)
    return np.asarray([i for i, label in enumerate(labels) if label == normalized], dtype=int)


def calculate_rdf(
    trajectory: PDBTrajectory,
    site1: str,
    site2: str,
    start_frame: int,
    stop_frame: Optional[int] = None,
    stride: int = 1,
    rmax: float = 8.0,
    bin_width: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, float]]:
    idx1 = indices_for_site(trajectory.labels, site1)
    idx2 = indices_for_site(trajectory.labels, site2)

    if idx1.size == 0:
        raise RuntimeError(f"No sites found for first token: {site1}")
    if idx2.size == 0:
        raise RuntimeError(f"No sites found for second token: {site2}")

    same_group = np.array_equal(idx1, idx2)
    if same_group and idx1.size < 2:
        raise RuntimeError(f"At least two {site1} sites are required for {site1}-{site2} RDF")

    print(f"[INFO] {site1}: {idx1.size} sites")
    print(f"[INFO] {site2}: {idx2.size} sites")

    if stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if bin_width <= 0.0:
        raise ValueError("--bin-width must be positive")
    if rmax <= 0.0:
        raise ValueError("--rmax must be positive")

    edges = np.arange(0.0, rmax + bin_width * 1.000001, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    histogram = np.zeros(centers.size, dtype=float)

    n_frames_used = 0
    normalization_total = np.zeros(centers.size, dtype=float)
    volume_sum = 0.0

    selected_frames = range(
        start_frame,
        trajectory.n_frames if stop_frame is None else min(stop_frame, trajectory.n_frames),
        stride,
    )

    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    for frame_index in selected_frames:
        dimensions = trajectory.boxes[frame_index]
        if dimensions is None:
            raise RuntimeError(
                f"No cell found for frame {frame_index} in {trajectory.source}. "
                "Add CRYST1 records or specify --box LX LY LZ."
            )

        volume = box_volume(dimensions)
        positions = trajectory.positions[frame_index]
        pos1 = positions[idx1]
        pos2 = positions[idx2]

        displacements = minimum_image_displacements(pos1, pos2, dimensions)
        distances = np.linalg.norm(displacements, axis=-1)

        if same_group:
            # Remove only i=i diagonal entries. The remaining ordered pairs
            # are consistent with n1*(n1-1) normalization.
            mask = ~np.eye(idx1.size, dtype=bool)
            distances = distances[mask]
        else:
            distances = distances.ravel()

        frame_histogram, _ = np.histogram(distances, bins=edges)
        histogram += frame_histogram

        n1 = idx1.size
        n2_effective = idx2.size - 1 if same_group else idx2.size
        rho2 = n2_effective / volume
        normalization_total += n1 * rho2 * shell_volume

        volume_sum += volume
        n_frames_used += 1

    if n_frames_used == 0:
        raise RuntimeError(
            "No frames were used. Check --start-ratio, --start-frame, "
            "--stop-frame, and --stride."
        )

    rdf = np.divide(
        histogram,
        normalization_total,
        out=np.zeros_like(histogram),
        where=normalization_total > 0.0,
    )

    dataframe = pd.DataFrame({"r": centers, "g": rdf, "count": histogram})
    metadata = {
        "n_frames_used": float(n_frames_used),
        "n_site1": float(idx1.size),
        "n_site2": float(idx2.size),
        "mean_volume_A3": float(volume_sum / n_frames_used),
    }
    return dataframe, metadata


def infer_pair_from_reference_filename(path: Path) -> Optional[tuple[str, str]]:
    match = re.match(r"reference_rdf_(.+?)_(.+?)\.txt$", path.name)
    if match is None:
        return None
    return normalize_site_token(match.group(1)), normalize_site_token(match.group(2))


def load_reference_txt(path: Path) -> pd.DataFrame:
    array = np.loadtxt(path)
    if array.ndim != 2 or array.shape[1] < 2:
        raise RuntimeError(f"Reference RDF file must have at least two columns: {path}")
    return pd.DataFrame({"r": array[:, 0], "g_ref": array[:, 1]})


def plot_multi_rdf(
    curves: list[tuple[str, pd.DataFrame]],
    reference: Optional[pd.DataFrame],
    site1: str,
    site2: str,
    outdir: Path,
    xlim: Optional[list[float]] = None,
    ylim: Optional[list[float]] = None,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    for label, dataframe in curves:
        ax.plot(dataframe["r"], dataframe["g"], lw=2.0, label=label)

    if reference is not None:
        ax.plot(
            reference["r"],
            reference["g_ref"],
            lw=1.8,
            ls="--",
            label="Reference",
        )

    ax.set_xlabel(rf"$r_{{\mathrm{{{site1}-{site2}}}}}$ / Å")
    ax.set_ylabel(rf"$g_{{\mathrm{{{site1}-{site2}}}}}(r)$")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    png_path = outdir / f"rdf_multi_{site1}_{site2}_with_reference.png"
    pdf_path = outdir / f"rdf_multi_{site1}_{site2}_with_reference.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[SAVE FIG] {png_path}")
    print(f"[SAVE FIG] {pdf_path}")


def determine_start_frame(args: argparse.Namespace, n_frames: int) -> int:
    if args.start_frame is not None:
        start = args.start_frame
    else:
        start = int(args.start_ratio * n_frames)

    if start < 0:
        start = max(0, n_frames + start)
    if start >= n_frames:
        raise RuntimeError(
            f"Start frame {start} is outside a trajectory with {n_frames} frames"
        )
    return start


def print_site_summary(trajectory: PDBTrajectory) -> None:
    unique, counts = np.unique(np.asarray(trajectory.labels, dtype=object), return_counts=True)
    summary = ", ".join(f"{label}={count}" for label, count in zip(unique, counts))
    print(f"  sites  : {trajectory.n_sites} ({summary})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate RDFs for multiple ordinary or CP2K ion+MLWC PDB trajectories "
            "using a robust internal PDB parser."
        )
    )

    parser.add_argument("--pdbs", nargs="+", default=None, help="Input PDB trajectories")
    parser.add_argument("--pdb", default=None, help="Single input PDB trajectory")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Curve labels; must match the number of input PDB files",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Pairs such as Li-WC S-WC Ge-S P-S Li-S",
    )
    parser.add_argument(
        "--ref-pattern",
        default="reference_rdf_*.txt",
        help="Reference RDF glob pattern",
    )
    parser.add_argument("--outdir", default="rdf_results", help="Output directory")
    parser.add_argument(
        "--start-ratio",
        type=float,
        default=0.5,
        help="Initial trajectory fraction to discard when --start-frame is omitted",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Explicit first frame; overrides --start-ratio. Negative values count from the end.",
    )
    parser.add_argument("--stop-frame", type=int, default=None, help="Exclusive stop frame")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument(
        "--rmax",
        type=float,
        default=None,
        help="Maximum RDF distance; defaults to reference maximum or half the shortest box length",
    )
    parser.add_argument("--bin-width", type=float, default=0.01, help="RDF bin width in Å")
    parser.add_argument(
        "--box",
        nargs="+",
        type=float,
        default=None,
        metavar="CELL",
        help=(
            "Manual cell. Use 3 values for orthorhombic LX LY LZ or 6 values "
            "for A B C ALPHA BETA GAMMA. Overrides CRYST1."
        ),
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--force", action="store_true", help="Recalculate existing CSV files")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Parse trajectories and print frame/site/cell diagnostics without calculating RDF",
    )

    args = parser.parse_args()

    if not (0.0 <= args.start_ratio < 1.0):
        parser.error("--start-ratio must satisfy 0 <= value < 1")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.box is not None and len(args.box) not in {3, 6}:
        parser.error("--box requires either 3 or 6 numbers")

    return args


def main() -> None:
    args = parse_args()

    if args.pdbs is not None:
        pdb_paths = [Path(path) for path in args.pdbs]
    elif args.pdb is not None:
        pdb_paths = [Path(args.pdb)]
    else:
        raise RuntimeError("Specify --pdb or --pdbs")

    for path in pdb_paths:
        if not path.exists():
            raise FileNotFoundError(f"PDB file not found: {path}")

    if args.labels is not None:
        if len(args.labels) != len(pdb_paths):
            raise RuntimeError("The number of --labels must equal the number of PDB files")
        labels = list(args.labels)
    else:
        labels = [path.stem for path in pdb_paths]

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    outdir.mkdir(parents=True, exist_ok=True)
    csvdir.mkdir(parents=True, exist_ok=True)

    print("===== Robust multi-PDB RDF analysis for atoms + MLWC =====")
    print(f"outdir   : {outdir}")
    print(f"manual box: {args.box}")

    trajectories: dict[Path, PDBTrajectory] = {}
    for path, label in zip(pdb_paths, labels):
        print(f"[PARSE] {label}: {path}")
        trajectory = read_pdb_trajectory(path, user_box=args.box)
        trajectories[path] = trajectory
        print(f"  frames : {trajectory.n_frames}")
        print_site_summary(trajectory)

        boxes_available = sum(box is not None for box in trajectory.boxes)
        print(f"  boxes  : {boxes_available}/{trajectory.n_frames} frames")
        if trajectory.boxes[0] is not None:
            values = " ".join(f"{value:.6g}" for value in trajectory.boxes[0])
            print(f"  cell[0]: {values}")

    if args.inspect_only:
        print("===== Inspection complete =====")
        return

    reference_files = sorted(Path(".").glob(args.ref_pattern))
    reference_map: dict[tuple[str, str], Path] = {}
    for path in reference_files:
        pair = infer_pair_from_reference_filename(path)
        if pair is not None:
            reference_map[pair] = path

    if args.pairs is None:
        pairs = sorted(reference_map.keys())
        if not pairs:
            raise RuntimeError(
                "No --pairs were supplied and no reference_rdf_*.txt files were found"
            )
    else:
        pairs = [parse_pair(pair) for pair in args.pairs]

    print(f"pairs    : {pairs}")
    print(f"force    : {args.force}")

    for site1, site2 in pairs:
        print("")
        print(f"===== RDF pair: {site1}-{site2} =====")

        reference_path = reference_map.get((site1, site2))
        if reference_path is None:
            reference_path = reference_map.get((site2, site1))

        reference = None
        if reference_path is not None:
            print(f"[REF] {reference_path}")
            reference = load_reference_txt(reference_path)
            reference.to_csv(
                csvdir / f"reference_rdf_{site1}_{site2}.csv",
                index=False,
            )

        curves: list[tuple[str, pd.DataFrame]] = []

        for path, label in zip(pdb_paths, labels):
            trajectory = trajectories[path]
            label_safe = safe_name(label)
            csv_path = csvdir / f"rdf_{label_safe}_{site1}_{site2}.csv"
            txt_path = csvdir / f"rdf_{label_safe}_{site1}_{site2}.txt"
            meta_path = csvdir / f"rdf_{label_safe}_{site1}_{site2}_metadata.txt"

            if csv_path.exists() and not args.force:
                print(f"[LOAD CSV] {csv_path}")
                dataframe = pd.read_csv(csv_path)
            else:
                start_frame = determine_start_frame(args, trajectory.n_frames)

                if args.rmax is not None:
                    rmax = args.rmax
                elif reference is not None:
                    rmax = float(reference["r"].max())
                else:
                    first_box = next((box for box in trajectory.boxes if box is not None), None)
                    if first_box is None:
                        raise RuntimeError(
                            f"Cannot infer --rmax for {path}; specify --box and/or --rmax"
                        )
                    # Conservative default. For triclinic cells users should set
                    # an explicit rmax when they want the largest valid radius.
                    rmax = 0.5 * float(np.min(first_box[:3]))

                print(f"[CALC] {label}: {path}")
                print(f"  frames : {trajectory.n_frames}")
                print(f"  start  : {start_frame}")
                print(f"  stop   : {args.stop_frame}")
                print(f"  stride : {args.stride}")
                print(f"  rmax   : {rmax:.6g} Å")

                dataframe, metadata = calculate_rdf(
                    trajectory=trajectory,
                    site1=site1,
                    site2=site2,
                    start_frame=start_frame,
                    stop_frame=args.stop_frame,
                    stride=args.stride,
                    rmax=rmax,
                    bin_width=args.bin_width,
                )

                dataframe.to_csv(csv_path, index=False)
                np.savetxt(txt_path, dataframe[["r", "g"]].values, fmt="%.10e")
                with meta_path.open("w") as handle:
                    for key, value in metadata.items():
                        handle.write(f"{key} = {value}\n")
                    handle.write(f"start_frame = {start_frame}\n")
                    handle.write(f"stop_frame = {args.stop_frame}\n")
                    handle.write(f"stride = {args.stride}\n")
                    handle.write(f"rmax_A = {rmax}\n")
                    handle.write(f"bin_width_A = {args.bin_width}\n")

                print(f"[SAVE CSV] {csv_path}")
                print(f"[SAVE TXT] {txt_path}")
                print(f"[SAVE META] {meta_path}")

            curves.append((label, dataframe))

        plot_multi_rdf(
            curves=curves,
            reference=reference,
            site1=site1,
            site2=site2,
            outdir=outdir,
            xlim=args.xlim,
            ylim=args.ylim,
            dpi=args.dpi,
        )

    print("")
    print("===== Done =====")


if __name__ == "__main__":
    main()

