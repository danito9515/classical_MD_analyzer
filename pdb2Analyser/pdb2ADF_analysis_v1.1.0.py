#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust multi-PDB Angle Distribution Function (ADF) analysis for ordinary
atomic trajectories and CP2K atom + maximally localized Wannier-center
(MLWC/WC) trajectories.

This version does not use MDTraj to parse the PDB. It therefore supports:
  * pseudo-sites such as WC with a blank PDB element field;
  * CP2K-derived PDB trajectories with imperfect MODEL/ENDMDL records;
  * repeated CRYST1 or END records;
  * atom serial numbers restarting at 1 for each frame;
  * wrapped and unwrapped trajectories through an explicit minimum-image
    convention.

The angle is defined as site1-site2-site3, with site2 at the vertex. A triple
(i, j, k) is counted when

    distance(site1_i, site2_j) < rcut12
    distance(site2_j, site3_k) < rcut23

and i, j, and k are distinct sites.

Examples
--------
1) Li-S-WC angle:

python3 pdb2ADF_analysis_v1.1.0.py \
  --pdbs ./T700K/*pdb ./T900K/*pdb ./T1100K/*pdb \
  --labels 700K 900K 1100K \
  --elem1 Li --elem2 S --elem3 WC \
  --rcut12 3.5 --rcut23 2.5 \
  --box 8.78765 8.78765 12.65755 \
  --outdir adf_Li_S_WC --force

2) S-WC-S angle with duplicate terminal permutations removed:

python3 pdb2ADF_analysis_v1.1.0.py \
  --pdbs ./T900K/*pdb \
  --elem1 S --elem2 WC --elem3 S \
  --rcut12 2.5 --rcut23 2.5 \
  --box 8.78765 8.78765 12.65755 \
  --unique-terminal-pairs \
  --outdir adf_S_WC_S

3) Inspect the trajectory parser without calculating the ADF:

python3 pdb2ADF_analysis_v1.1.0.py \
  --pdbs ./T900K/*pdb \
  --box 8.78765 8.78765 12.65755 \
  --inspect-only
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
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
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
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

    return token.upper()


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))


def expand_patterns(patterns: Optional[list[str]], what: str = "files") -> list[Path]:
    if patterns is None:
        return []

    expanded: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        if not matches:
            print(f"[WARNING] No {what} matched: {pattern}", file=sys.stderr)

        for match in matches:
            normalized = str(Path(match))
            if normalized not in seen:
                expanded.append(Path(normalized))
                seen.add(normalized)

    return expanded


def parse_cryst1(line: str) -> Optional[np.ndarray]:
    """Parse CRYST1 as [a, b, c, alpha, beta, gamma]."""
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
            values = [float(value) for value in fields[1:7]]
            if min(values[:3]) > 0.0:
                return np.asarray(values, dtype=float)
        except ValueError:
            pass

    return None


def parse_atom_line(line: str) -> tuple[Optional[int], str, np.ndarray]:
    """Parse ATOM/HETATM with fixed-column parsing and whitespace fallbacks."""
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
        xyz = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        )
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
        candidates: list[tuple[int, int, int]] = []
        if len(fields) >= 9:
            candidates.append((6, 7, 8))
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
        for index in range(5, max(5, len(fields) - 2)):
            try:
                xyz = np.asarray(
                    [
                        float(fields[index]),
                        float(fields[index + 1]),
                        float(fields[index + 2]),
                    ],
                    dtype=float,
                )
                break
            except (ValueError, IndexError):
                continue

    if xyz is None:
        raise ValueError(f"Could not parse coordinates from PDB line:\n{line.rstrip()}")

    site = normalize_site_token(element)
    if not site:
        site = normalize_site_token(atom_name)
    if not site:
        raise ValueError(f"Could not identify atom/WC label from line:\n{line.rstrip()}")

    return serial, site, xyz


def read_pdb_trajectory(
    path: Path,
    user_box: Optional[Iterable[float]] = None,
) -> PDBTrajectory:
    """Read ordinary and non-standard multi-frame PDB trajectories."""
    path = Path(path)

    manual_box: Optional[np.ndarray] = None
    if user_box is not None:
        values = list(user_box)
        if len(values) == 3:
            manual_box = np.asarray(
                [values[0], values[1], values[2], 90.0, 90.0, 90.0],
                dtype=float,
            )
        elif len(values) == 6:
            manual_box = np.asarray(values, dtype=float)
        else:
            raise ValueError("--box requires either 3 or 6 numbers")

    frames: list[np.ndarray] = []
    labels_by_frame: list[list[str]] = []
    boxes: list[Optional[np.ndarray]] = []

    current_positions: list[np.ndarray] = []
    current_labels: list[str] = []
    current_box: Optional[np.ndarray] = (
        manual_box.copy() if manual_box is not None else None
    )
    last_cryst1: Optional[np.ndarray] = (
        manual_box.copy() if manual_box is not None else None
    )
    inside_model = False
    previous_serial: Optional[int] = None

    def flush_frame() -> None:
        nonlocal current_positions, current_labels, current_box, previous_serial
        if not current_positions:
            previous_serial = None
            return

        frames.append(np.asarray(current_positions, dtype=float))
        labels_by_frame.append(list(current_labels))
        boxes.append(
            manual_box.copy()
            if manual_box is not None
            else (None if current_box is None else current_box.copy())
        )

        current_positions = []
        current_labels = []
        current_box = (
            manual_box.copy()
            if manual_box is not None
            else (None if last_cryst1 is None else last_cryst1.copy())
        )
        previous_serial = None

    with path.open("r", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = line[:6].strip().upper()

            if record == "CRYST1":
                parsed_box = parse_cryst1(line)
                if parsed_box is not None:
                    last_cryst1 = parsed_box

                if current_positions and not inside_model:
                    flush_frame()

                if manual_box is None:
                    current_box = None if parsed_box is None else parsed_box.copy()
                continue

            if record == "MODEL":
                if current_positions:
                    flush_frame()
                inside_model = True
                if manual_box is None and last_cryst1 is not None:
                    current_box = last_cryst1.copy()
                continue

            if record in {"ENDMDL", "END"}:
                if current_positions:
                    flush_frame()
                if record == "ENDMDL":
                    inside_model = False
                continue

            if record not in {"ATOM", "HETATM"}:
                continue

            try:
                serial, site, xyz = parse_atom_line(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

            if (
                current_positions
                and not inside_model
                and serial == 1
                and previous_serial is not None
                and serial <= previous_serial
            ):
                flush_frame()

            current_positions.append(xyz)
            current_labels.append(site)
            previous_serial = serial

    flush_frame()

    if not frames:
        raise RuntimeError(f"No ATOM/HETATM records found in {path}")

    reference_labels = labels_by_frame[0]
    reference_count = len(reference_labels)

    for frame_index, (labels, positions) in enumerate(zip(labels_by_frame, frames)):
        if len(labels) != reference_count:
            raise RuntimeError(
                f"Inconsistent site count in {path}: frame 0 has {reference_count}, "
                f"frame {frame_index} has {len(labels)}. The final frame may be incomplete, "
                "or the frame delimiters may be malformed."
            )
        if labels != reference_labels:
            mismatch = next(
                (
                    index
                    for index, (reference, current) in enumerate(
                        zip(reference_labels, labels)
                    )
                    if reference != current
                ),
                None,
            )
            raise RuntimeError(
                f"Site ordering changes in {path}, frame {frame_index}, site {mismatch}. "
                "All frames must preserve the atom/WC ordering."
            )
        if positions.shape != (reference_count, 3):
            raise RuntimeError(
                f"Unexpected coordinate shape in {path}, frame {frame_index}: "
                f"{positions.shape}"
            )

    return PDBTrajectory(
        labels=reference_labels,
        positions=frames,
        boxes=boxes,
        source=path,
    )


def cell_matrix_from_dimensions(dimensions: np.ndarray) -> np.ndarray:
    """Build a row-vector triclinic cell matrix from PDB dimensions."""
    a, b, c, alpha_deg, beta_deg, gamma_deg = map(float, dimensions)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("Invalid cell: sin(gamma) is zero")

    vector_a = np.asarray([a, 0.0, 0.0], dtype=float)
    vector_b = np.asarray(
        [b * math.cos(gamma), b * sin_gamma, 0.0],
        dtype=float,
    )

    cx = c * math.cos(beta)
    cy = c * (
        math.cos(alpha) - math.cos(beta) * math.cos(gamma)
    ) / sin_gamma
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared < -1.0e-8:
        raise ValueError(f"Invalid triclinic cell; c_z^2 = {cz_squared}")

    vector_c = np.asarray(
        [cx, cy, math.sqrt(max(0.0, cz_squared))],
        dtype=float,
    )

    return np.vstack([vector_a, vector_b, vector_c])


def minimum_image_vectors(
    origins: np.ndarray,
    targets: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    """
    Return target-origin minimum-image vectors.

    Output shape is (n_origins, n_targets, 3).
    """
    cell = cell_matrix_from_dimensions(dimensions)
    inverse_cell = np.linalg.inv(cell)

    displacement = targets[None, :, :] - origins[:, None, :]
    fractional = displacement @ inverse_cell
    fractional -= np.rint(fractional)
    return fractional @ cell


def indices_for_site(labels: list[str], site: str) -> np.ndarray:
    normalized = normalize_site_token(site)
    return np.asarray(
        [index for index, label in enumerate(labels) if label == normalized],
        dtype=int,
    )


def determine_frame_range(
    n_frames: int,
    start_frame: Optional[int],
    start_ratio: float,
    end_frame: Optional[int],
    stride: int,
) -> range:
    if start_frame is None:
        start = int(start_ratio * n_frames)
    else:
        start = start_frame
        if start < 0:
            start = max(0, n_frames + start)

    stop = n_frames if end_frame is None else min(end_frame, n_frames)
    if stop < 0:
        stop = max(0, n_frames + stop)

    if start < 0 or start >= n_frames:
        raise ValueError(
            f"Start frame {start} is outside a trajectory with {n_frames} frames"
        )
    if stop <= start:
        raise ValueError(
            f"Invalid frame range: start={start}, stop={stop}, n_frames={n_frames}"
        )
    if stride <= 0:
        raise ValueError("--stride must be a positive integer")

    return range(start, stop, stride)


def calculate_adf(
    trajectory: PDBTrajectory,
    site1: str,
    site2: str,
    site3: str,
    rcut12_A: float,
    rcut23_A: float,
    bin_width_deg: float,
    start_frame: Optional[int],
    start_ratio: float,
    end_frame: Optional[int],
    stride: int,
    unique_terminal_pairs: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calculate site1-site2-site3 ADF with site2 as the angle vertex."""
    if rcut12_A <= 0.0 or rcut23_A <= 0.0:
        raise ValueError("--rcut12 and --rcut23 must be positive")
    if bin_width_deg <= 0.0:
        raise ValueError("--bin-width must be positive")

    site1 = normalize_site_token(site1)
    site2 = normalize_site_token(site2)
    site3 = normalize_site_token(site3)

    idx1 = indices_for_site(trajectory.labels, site1)
    idx2 = indices_for_site(trajectory.labels, site2)
    idx3 = indices_for_site(trajectory.labels, site3)

    if idx1.size == 0:
        raise RuntimeError(f"No sites found for elem1/site1: {site1}")
    if idx2.size == 0:
        raise RuntimeError(f"No sites found for elem2/site2 (center): {site2}")
    if idx3.size == 0:
        raise RuntimeError(f"No sites found for elem3/site3: {site3}")

    print(
        f"[INFO] selected sites: {site1}={idx1.size}, "
        f"{site2}={idx2.size}, {site3}={idx3.size}"
    )

    edges = np.arange(
        0.0,
        180.0 + bin_width_deg * 1.000001,
        bin_width_deg,
        dtype=float,
    )
    edges[-1] = 180.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(centers.size, dtype=np.int64)

    frame_range = determine_frame_range(
        n_frames=trajectory.n_frames,
        start_frame=start_frame,
        start_ratio=start_ratio,
        end_frame=end_frame,
        stride=stride,
    )

    same_terminal_group = np.array_equal(idx1, idx3)
    n_angles_total = 0
    n_frames_with_angles = 0
    n_centers_with_angles = 0
    analyzed_frames = 0

    for progress_index, frame_index in enumerate(frame_range, start=1):
        dimensions = trajectory.boxes[frame_index]
        if dimensions is None:
            raise RuntimeError(
                f"No cell dimensions for frame {frame_index} in {trajectory.source}. "
                "Add CRYST1 records or specify --box LX LY LZ."
            )

        positions = trajectory.positions[frame_index]
        centers_xyz = positions[idx2]

        vectors12 = minimum_image_vectors(
            origins=centers_xyz,
            targets=positions[idx1],
            dimensions=dimensions,
        )
        vectors23 = minimum_image_vectors(
            origins=centers_xyz,
            targets=positions[idx3],
            dimensions=dimensions,
        )

        distances12 = np.linalg.norm(vectors12, axis=2)
        distances23 = np.linalg.norm(vectors23, axis=2)

        frame_angle_count = 0

        for local_center, center_index in enumerate(idx2):
            mask1 = (distances12[local_center] < rcut12_A) & (
                distances12[local_center] > 1.0e-12
            )
            mask3 = (distances23[local_center] < rcut23_A) & (
                distances23[local_center] > 1.0e-12
            )

            # Explicit identity filtering is required when a terminal group is
            # the same object as the central group.
            mask1 &= idx1 != center_index
            mask3 &= idx3 != center_index

            local1 = np.flatnonzero(mask1)
            local3 = np.flatnonzero(mask3)
            if local1.size == 0 or local3.size == 0:
                continue

            vectors1 = vectors12[local_center, local1]
            vectors3 = vectors23[local_center, local3]
            global1 = idx1[local1]
            global3 = idx3[local3]

            dot_products = vectors1 @ vectors3.T
            norms = np.outer(
                np.linalg.norm(vectors1, axis=1),
                np.linalg.norm(vectors3, axis=1),
            )

            valid = norms > 1.0e-14
            valid &= global1[:, None] != global3[None, :]

            # For A-B-A, i-j-k and k-j-i are geometrically identical. The
            # original MDTraj implementation counted both. This switch can
            # retain only one ordering without changing other triplets.
            if unique_terminal_pairs and same_terminal_group:
                valid &= global1[:, None] < global3[None, :]

            if not np.any(valid):
                continue

            cosines = np.divide(
                dot_products,
                norms,
                out=np.zeros_like(dot_products),
                where=valid,
            )
            angles_deg = np.degrees(np.arccos(np.clip(cosines[valid], -1.0, 1.0)))

            histogram, _ = np.histogram(angles_deg, bins=edges)
            counts += histogram.astype(np.int64)
            n_found = int(angles_deg.size)
            n_angles_total += n_found
            frame_angle_count += n_found
            n_centers_with_angles += 1

        if frame_angle_count > 0:
            n_frames_with_angles += 1

        analyzed_frames += 1
        if progress_index % 100 == 0:
            print(
                f"[PROGRESS] analyzed {progress_index} frames; "
                f"angles so far = {n_angles_total}"
            )

    if n_angles_total > 0:
        density = counts.astype(float) / (
            float(np.sum(counts)) * bin_width_deg
        )
        probability = counts.astype(float) / float(np.sum(counts))
    else:
        print(
            f"[WARNING] No {site1}-{site2}-{site3} angles were found. "
            "Check the site names and cutoff radii.",
            file=sys.stderr,
        )
        density = np.zeros_like(counts, dtype=float)
        probability = np.zeros_like(counts, dtype=float)

    dataframe = pd.DataFrame(
        {
            "angle_deg": centers,
            "density_per_degree": density,
            "probability_per_bin": probability,
            "count": counts,
        }
    )

    selected_start = frame_range.start
    selected_stop = frame_range.stop
    metadata: dict[str, object] = {
        "pdb_file": str(trajectory.source),
        "site1": site1,
        "site2_center": site2,
        "site3": site3,
        "rcut12_A": rcut12_A,
        "rcut23_A": rcut23_A,
        "bin_width_deg": bin_width_deg,
        "start_frame": selected_start,
        "end_frame_exclusive": selected_stop,
        "stride": stride,
        "n_total_frames_in_pdb": trajectory.n_frames,
        "n_analyzed_frames": analyzed_frames,
        "n_frames_with_angles": n_frames_with_angles,
        "n_centers_with_angles_accumulated": n_centers_with_angles,
        "n_angles_total": n_angles_total,
        "n_site1": int(idx1.size),
        "n_site2": int(idx2.size),
        "n_site3": int(idx3.size),
        "unique_terminal_pairs": unique_terminal_pairs,
    }

    return dataframe, metadata


def csv_path_for_pdb(
    outdir: Path,
    pdb_file: Path,
    site1: str,
    site2: str,
    site3: str,
    rcut12: float,
    rcut23: float,
    bin_width: float,
    unique_terminal_pairs: bool,
) -> Path:
    stem = safe_name(pdb_file.stem)
    angle_name = safe_name(f"{site1}-{site2}-{site3}")
    unique_suffix = "_uniqueTerm" if unique_terminal_pairs else ""
    filename = (
        f"{stem}_ADF_{angle_name}"
        f"_r12_{rcut12:g}A_r23_{rcut23:g}A"
        f"_bin_{bin_width:g}deg{unique_suffix}.csv"
    )
    return outdir / "csv" / filename


def save_csv_with_metadata(
    dataframe: pd.DataFrame,
    metadata: dict[str, object],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"# {key}: {value}\n")
        dataframe.to_csv(handle, index=False)
    print(f"[SAVE CSV] {csv_path}")


def load_adf_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    dataframe = pd.read_csv(csv_path, comment="#")

    # Compatibility with v1.0.0 output.
    if "density" in dataframe.columns and "density_per_degree" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"density": "density_per_degree"})
    return dataframe


def load_reference_txt(txt_path: Path) -> pd.DataFrame:
    array = np.loadtxt(txt_path)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            f"Reference txt must have at least two columns: angle_deg density. "
            f"Got: {txt_path}"
        )
    return pd.DataFrame(
        {
            "angle_deg": array[:, 0],
            "density_per_degree": array[:, 1],
        }
    )


def plot_adfs(
    calc_results: list[tuple[str, pd.DataFrame]],
    ref_results: list[tuple[str, pd.DataFrame]],
    site1: str,
    site2: str,
    site3: str,
    fig_path: Path,
    title: Optional[str],
    xlim: Optional[list[float]],
    ylim: Optional[list[float]],
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=figsize)

    for label, dataframe in calc_results:
        axis.plot(
            dataframe["angle_deg"],
            dataframe["density_per_degree"],
            label=label,
            linewidth=2.0,
        )

    for label, dataframe in ref_results:
        axis.plot(
            dataframe["angle_deg"],
            dataframe["density_per_degree"],
            linestyle="--",
            linewidth=2.0,
            label=label,
        )

    axis.set_xlabel(rf"Angle $\angle$({site1}-{site2}-{site3}) / degree")
    axis.set_ylabel(r"Probability density / degree$^{-1}$")
    axis.set_title(title if title is not None else f"ADF: {site1}-{site2}-{site3}")

    axis.set_xlim(*(xlim if xlim is not None else [0.0, 180.0]))
    if ylim is not None:
        axis.set_ylim(*ylim)

    axis.legend(frameon=False)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=dpi)

    pdf_path = fig_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[SAVE FIG] {fig_path}")
    print(f"[SAVE FIG] {pdf_path}")


def save_combined_csv(
    calc_results: list[tuple[str, pd.DataFrame]],
    ref_results: list[tuple[str, pd.DataFrame]],
    combined_csv: Path,
) -> None:
    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    combined: Optional[pd.DataFrame] = None

    for label, dataframe in calc_results + ref_results:
        column = safe_name(label)
        temporary = dataframe[["angle_deg", "density_per_degree"]].rename(
            columns={"density_per_degree": column}
        )
        combined = (
            temporary
            if combined is None
            else pd.merge(combined, temporary, on="angle_deg", how="outer")
        )

    if combined is None:
        combined = pd.DataFrame(columns=["angle_deg"])

    combined.sort_values("angle_deg").to_csv(combined_csv, index=False)
    print(f"[SAVE COMBINED CSV] {combined_csv}")


def print_site_summary(trajectory: PDBTrajectory) -> None:
    unique, counts = np.unique(
        np.asarray(trajectory.labels, dtype=object),
        return_counts=True,
    )
    summary = ", ".join(
        f"{label}={count}" for label, count in zip(unique, counts)
    )
    print(f"  sites  : {trajectory.n_sites} ({summary})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate ADFs for multiple ordinary or CP2K atom+MLWC PDB "
            "trajectories using a robust internal PDB parser."
        )
    )

    parser.add_argument(
        "--pdbs",
        nargs="+",
        required=True,
        help="Input PDB files or glob patterns",
    )
    parser.add_argument("--elem1", default="Li", help="First terminal atom/site")
    parser.add_argument("--elem2", default="N", help="Central atom/site")
    parser.add_argument("--elem3", default="C", help="Third terminal atom/site")

    # Retained only so old command lines fail gracefully rather than as unknown
    # arguments. The robust parser selects directly by atom/site token.
    parser.add_argument(
        "--sel1",
        default=None,
        help="Deprecated. Use --elem1 with an atom or pseudo-site token such as WC.",
    )
    parser.add_argument(
        "--sel2",
        default=None,
        help="Deprecated. Use --elem2 with an atom or pseudo-site token such as WC.",
    )
    parser.add_argument(
        "--sel3",
        default=None,
        help="Deprecated. Use --elem3 with an atom or pseudo-site token such as WC.",
    )

    parser.add_argument("--rcut12", type=float, default=3.0)
    parser.add_argument("--rcut23", type=float, default=3.0)
    parser.add_argument("--bin-width", type=float, default=1.0)

    parser.add_argument(
        "--start-ratio",
        type=float,
        default=0.0,
        help="Discard this initial fraction when --start-frame is omitted",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Explicit first frame; negative values count from the end",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive end frame; negative values count from the end",
    )
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument(
        "--box",
        nargs="+",
        type=float,
        default=None,
        metavar="CELL",
        help=(
            "Manual cell: LX LY LZ or A B C ALPHA BETA GAMMA. "
            "Overrides CRYST1."
        ),
    )

    parser.add_argument("--outdir", default="adf_results")
    parser.add_argument("--fig", default=None)
    parser.add_argument("--combined-csv", default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--reference-txts", nargs="*", default=None)
    parser.add_argument("--reference-labels", nargs="+", default=None)

    parser.add_argument("--title", default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(6.0, 4.0),
        metavar=("W", "H"),
    )
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument(
        "--unique-terminal-pairs",
        action="store_true",
        help=(
            "For A-B-A angles, count each unordered terminal pair once. "
            "Without this option, i-j-k and k-j-i are both counted for "
            "compatibility with the original script."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Parse and report frames/sites/cells without calculating ADF",
    )
    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Accepted for compatibility; the internal parser emits no MDTraj warnings",
    )

    args = parser.parse_args()

    if not (0.0 <= args.start_ratio < 1.0):
        parser.error("--start-ratio must satisfy 0 <= value < 1")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.bin_width <= 0.0 or args.bin_width > 180.0:
        parser.error("--bin-width must satisfy 0 < value <= 180")
    if args.rcut12 <= 0.0 or args.rcut23 <= 0.0:
        parser.error("--rcut12 and --rcut23 must be positive")
    if args.box is not None and len(args.box) not in {3, 6}:
        parser.error("--box requires either 3 or 6 numbers")

    return args


def main() -> None:
    args = parse_args()

    if args.sel1 or args.sel2 or args.sel3:
        print(
            "[WARNING] --sel1/--sel2/--sel3 are deprecated in v1.1.0. "
            "Sites are selected directly by --elem1/--elem2/--elem3.",
            file=sys.stderr,
        )

    pdb_files = expand_patterns(args.pdbs, what="PDB files")
    if not pdb_files:
        raise FileNotFoundError("No PDB files were found from --pdbs")

    ref_files = expand_patterns(args.reference_txts, what="reference txt files")

    if args.labels is None:
        labels = [path.stem for path in pdb_files]
    else:
        labels = list(args.labels)
        if len(labels) != len(pdb_files):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of PDB "
                f"files ({len(pdb_files)})"
            )

    if args.reference_labels is None:
        ref_labels = [path.stem for path in ref_files]
    else:
        ref_labels = list(args.reference_labels)
        if len(ref_labels) != len(ref_files):
            raise ValueError(
                f"Number of reference labels ({len(ref_labels)}) must match "
                f"number of reference files ({len(ref_files)})"
            )

    site1 = normalize_site_token(args.elem1)
    site2 = normalize_site_token(args.elem2)
    site3 = normalize_site_token(args.elem3)
    if not site1 or not site2 or not site3:
        raise ValueError("--elem1, --elem2, and --elem3 must be non-empty site tokens")

    outdir = Path(args.outdir)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)

    angle_name = f"{site1}-{site2}-{site3}"
    fig_path = (
        Path(args.fig)
        if args.fig is not None
        else outdir / f"ADF_{safe_name(angle_name)}.png"
    )
    combined_csv = (
        Path(args.combined_csv)
        if args.combined_csv is not None
        else outdir / f"ADF_{safe_name(angle_name)}_all.csv"
    )

    print("===== Robust multi-PDB ADF analysis for atoms + MLWC =====")
    print(f"angle    : {site1}-{site2}-{site3} (center = {site2})")
    print(f"rcut12   : {args.rcut12:g} Å")
    print(f"rcut23   : {args.rcut23:g} Å")
    print(f"manual box: {args.box}")
    print(f"outdir   : {outdir}")

    trajectories: dict[Path, PDBTrajectory] = {}
    for pdb_file, label in zip(pdb_files, labels):
        print(f"[PARSE] {label}: {pdb_file}")
        trajectory = read_pdb_trajectory(pdb_file, user_box=args.box)
        trajectories[pdb_file] = trajectory
        print(f"  frames : {trajectory.n_frames}")
        print_site_summary(trajectory)
        boxes_available = sum(box is not None for box in trajectory.boxes)
        print(f"  boxes  : {boxes_available}/{trajectory.n_frames} frames")
        if trajectory.boxes[0] is not None:
            box_text = " ".join(
                f"{value:.6g}" for value in trajectory.boxes[0]
            )
            print(f"  cell[0]: {box_text}")

    if args.inspect_only:
        print("===== Inspection complete =====")
        return

    calc_results: list[tuple[str, pd.DataFrame]] = []
    ref_results: list[tuple[str, pd.DataFrame]] = []

    for pdb_file, label in zip(pdb_files, labels):
        csv_path = csv_path_for_pdb(
            outdir=outdir,
            pdb_file=pdb_file,
            site1=site1,
            site2=site2,
            site3=site3,
            rcut12=args.rcut12,
            rcut23=args.rcut23,
            bin_width=args.bin_width,
            unique_terminal_pairs=args.unique_terminal_pairs,
        )

        if csv_path.exists() and not args.force:
            print(f"[LOAD CSV] {csv_path}")
            dataframe = load_adf_csv(csv_path)
        else:
            if args.plot_only:
                raise FileNotFoundError(
                    f"--plot-only was specified but CSV does not exist: {csv_path}"
                )

            print(f"[CALC] {label}: {pdb_file}")
            dataframe, metadata = calculate_adf(
                trajectory=trajectories[pdb_file],
                site1=site1,
                site2=site2,
                site3=site3,
                rcut12_A=args.rcut12,
                rcut23_A=args.rcut23,
                bin_width_deg=args.bin_width,
                start_frame=args.start_frame,
                start_ratio=args.start_ratio,
                end_frame=args.end_frame,
                stride=args.stride,
                unique_terminal_pairs=args.unique_terminal_pairs,
            )
            save_csv_with_metadata(dataframe, metadata, csv_path)

        calc_results.append((label, dataframe))

    for txt_file, label in zip(ref_files, ref_labels):
        print(f"[LOAD REF] {txt_file}")
        ref_results.append((label, load_reference_txt(txt_file)))

    save_combined_csv(calc_results, ref_results, combined_csv)
    plot_adfs(
        calc_results=calc_results,
        ref_results=ref_results,
        site1=site1,
        site2=site2,
        site3=site3,
        fig_path=fig_path,
        title=args.title,
        xlim=args.xlim,
        ylim=args.ylim,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )

    print("===== Done =====")


if __name__ == "__main__":
    main()
