#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: 
    python3 adf_multi_pdb_v3.py --pdbs nvt*.pdb   --elem1 Li --elem2 N --elem3 C   --rcut12 3.0 --rcut23 3.0   --reference-txts reference_adf_CNLi.txt   --reference-labels reference_CNLi   --outdir adf_compare   --fig adf_compare/ADF_CNLi_with_reference.png   --suppress-warnings
adf_multi_pdb_v3.py

複数のPDB trajectoryから Angle Distribution Function (ADF) を計算し、
各PDBの結果をCSVに保存し、1枚のFigureに重ね描きするスクリプト。

さらに、reference用の2列txtファイル（angle, density）を読み込み、
解析結果と重ねてFigure化できる。
"""

import argparse
import glob
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdtraj as md


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate ADF from multiple PDB trajectory files and plot them in one figure."
    )

    parser.add_argument(
        "--pdbs",
        nargs="+",
        required=True,
        help=(
            "Input PDB trajectory files or glob patterns. "
            "Example: --pdbs NVE300K_e1.pdb NVE300K_e2.pdb NVT300K_e2.pdb "
            "or --pdbs '*.pdb'"
        ),
    )

    parser.add_argument("--elem1", type=str, default="Li", help="First atom element/name. Default: Li")
    parser.add_argument("--elem2", type=str, default="N", help="Central atom element/name. Default: N")
    parser.add_argument("--elem3", type=str, default="C", help="Third atom element/name. Default: C")

    parser.add_argument("--sel1", type=str, default=None, help='Optional MDTraj selection for atom1.')
    parser.add_argument("--sel2", type=str, default=None, help='Optional MDTraj selection for atom2.')
    parser.add_argument("--sel3", type=str, default=None, help='Optional MDTraj selection for atom3.')

    parser.add_argument("--rcut12", type=float, default=3.0,
                        help="Cutoff distance for elem1-elem2 in Angstrom. Default: 3.0")
    parser.add_argument("--rcut23", type=float, default=3.0,
                        help="Cutoff distance for elem2-elem3 in Angstrom. Default: 3.0")
    parser.add_argument("--bin-width", type=float, default=1.0,
                        help="Histogram bin width in degrees. Default: 1.0")

    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame index to analyze. Default: 0")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="End frame index, Python slicing style.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Analyze every N frames. Default: 1")

    parser.add_argument("--outdir", type=str, default="adf_results",
                        help="Output directory for CSV and figures. Default: adf_results")
    parser.add_argument("--fig", type=str, default=None,
                        help="Output figure path. Default: <outdir>/ADF_elem1-elem2-elem3.png")
    parser.add_argument("--combined-csv", type=str, default=None,
                        help="Output combined CSV path. Default: <outdir>/ADF_elem1-elem2-elem3_all.csv")

    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for PDB curves. Number must match number of expanded PDB files.",
    )

    parser.add_argument(
        "--reference-txts",
        nargs="*",
        default=None,
        help="Reference txt files or glob patterns. Each txt must contain two columns: angle(deg) and density.",
    )
    parser.add_argument(
        "--reference-labels",
        nargs="+",
        default=None,
        help="Labels for reference txt curves. Number must match number of expanded reference txt files.",
    )

    parser.add_argument("--title", type=str, default=None, help="Figure title.")
    parser.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"))
    parser.add_argument("--figsize", nargs=2, type=float, default=(6.0, 4.0), metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=300, help="Figure dpi. Default: 300")

    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Suppress noisy warnings from MDTraj/PDB parser.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate ADF even if CSV already exists.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only load existing CSV files and plot. Do not analyze PDB files.",
    )

    return parser.parse_args()


def expand_patterns(patterns, what="files"):
    if patterns is None:
        return []

    files = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            files.extend(matches)
        elif Path(pat).exists():
            files.append(pat)
        else:
            print(f"[WARNING] No {what} matched: {pat}", file=sys.stderr)

    unique = []
    seen = set()
    for f in files:
        p = str(Path(f))
        if p not in seen:
            unique.append(Path(p))
            seen.add(p)

    return unique


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text))


def guess_atom_indices_from_name(topology, elem):
    target = elem.strip().lower()
    indices = []

    for atom in topology.atoms:
        candidates = []
        if atom.element is not None:
            candidates.append(atom.element.symbol)
        candidates.append(atom.name)

        for cand in candidates:
            if cand is None:
                continue
            letters = re.sub(r"[^A-Za-z]", "", cand).lower()

            if letters == target:
                indices.append(atom.index)
                break

            if letters.startswith(target):
                if target == "c" and letters.startswith("cl"):
                    continue
                indices.append(atom.index)
                break

    return np.array(sorted(set(indices)), dtype=int)


def select_indices(topology, elem, selection=None, label="atom"):
    if selection is None:
        selection = f"element {elem}"

    try:
        indices = topology.select(selection)
    except Exception as exc:
        print(f"[WARNING] MDTraj selection failed for {label}: {selection}", file=sys.stderr)
        print(f"[WARNING] {exc}", file=sys.stderr)
        indices = np.array([], dtype=int)

    if len(indices) == 0:
        print(
            f"[WARNING] No atoms found by selection '{selection}'. Trying fallback for '{elem}'.",
            file=sys.stderr,
        )
        indices = guess_atom_indices_from_name(topology, elem)

    if len(indices) == 0:
        raise ValueError(
            f"No atoms were selected for {label}. Check PDB or use --sel1/--sel2/--sel3."
        )

    return np.asarray(indices, dtype=int)


def make_pairs(idx_a, idx_b):
    pairs = [(int(i), int(j)) for i in idx_a for j in idx_b if int(i) != int(j)]
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=int)
    return np.asarray(pairs, dtype=int)


def calculate_adf_from_pdb(
    pdb_file,
    elem1,
    elem2,
    elem3,
    rcut12_A=3.0,
    rcut23_A=3.0,
    bin_width_deg=1.0,
    start_frame=0,
    end_frame=None,
    stride=1,
    sel1=None,
    sel2=None,
    sel3=None,
    suppress_warnings=False,
):
    pdb_file = Path(pdb_file)
    print(f"[LOAD] {pdb_file}")

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*two consecutive residues with same number.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                module=r"mdtraj\.formats\.pdb\.pdbstructure",
                category=UserWarning,
            )
            traj = md.load(str(pdb_file))
    else:
        traj = md.load(str(pdb_file))

    n_total = traj.n_frames
    if end_frame is None:
        end_frame = n_total
    start_frame = max(0, start_frame)
    end_frame = min(n_total, end_frame)

    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid frame range for {pdb_file}: start_frame={start_frame}, end_frame={end_frame}"
        )

    traj = traj[start_frame:end_frame:stride]
    print(f"[INFO] frames used: {traj.n_frames} / total {n_total}")

    idx1 = select_indices(traj.topology, elem1, sel1, "elem1")
    idx2 = select_indices(traj.topology, elem2, sel2, "elem2")
    idx3 = select_indices(traj.topology, elem3, sel3, "elem3")

    print(f"[INFO] selected atoms: {elem1}={len(idx1)}, {elem2}={len(idx2)}, {elem3}={len(idx3)}")

    pairs12 = make_pairs(idx1, idx2)
    pairs23 = make_pairs(idx2, idx3)

    if len(pairs12) == 0:
        raise ValueError(f"No valid pairs for {elem1}-{elem2}.")
    if len(pairs23) == 0:
        raise ValueError(f"No valid pairs for {elem2}-{elem3}.")

    rcut12_nm = rcut12_A / 10.0
    rcut23_nm = rcut23_A / 10.0

    bins = np.arange(0.0, 180.0 + bin_width_deg + 1.0e-12, bin_width_deg)
    if bins[-1] > 180.0:
        bins[-1] = 180.0
    angle_centers = 0.5 * (bins[:-1] + bins[1:])

    counts = np.zeros(len(bins) - 1, dtype=np.float64)
    n_angles_total = 0
    n_frames_with_angles = 0

    for iframe in range(traj.n_frames):
        frame = traj[iframe]

        d12 = md.compute_distances(frame, pairs12, periodic=True)[0]
        d23 = md.compute_distances(frame, pairs23, periodic=True)[0]

        cut12 = pairs12[d12 < rcut12_nm]
        cut23 = pairs23[d23 < rcut23_nm]

        if len(cut12) == 0 or len(cut23) == 0:
            continue

        by_center = {}
        for j, k in cut23:
            by_center.setdefault(int(j), []).append(int(k))

        triples = []
        for i, j in cut12:
            i = int(i)
            j = int(j)
            for k in by_center.get(j, []):
                if i != k:
                    triples.append((i, j, k))

        if len(triples) == 0:
            continue

        triples = np.asarray(triples, dtype=int)
        angles_rad = md.compute_angles(frame, triples, periodic=True, opt=True)[0]
        angles_deg = np.rad2deg(angles_rad)

        hist, _ = np.histogram(angles_deg, bins=bins)
        counts += hist
        n_angles_total += len(angles_deg)
        n_frames_with_angles += 1

        if (iframe + 1) % 100 == 0:
            print(f"[PROGRESS] frame {iframe + 1}/{traj.n_frames}, angles so far = {n_angles_total}")

    if n_angles_total > 0:
        density = counts / (np.sum(counts) * bin_width_deg)
    else:
        print(
            f"[WARNING] No angles found for {pdb_file}. Try larger cutoffs or check selections.",
            file=sys.stderr,
        )
        density = np.zeros_like(counts)

    df = pd.DataFrame(
        {
            "angle_deg": angle_centers,
            "density": density,
            "count": counts.astype(int),
        }
    )

    metadata = {
        "pdb_file": str(pdb_file),
        "elem1": elem1,
        "elem2": elem2,
        "elem3": elem3,
        "rcut12_A": rcut12_A,
        "rcut23_A": rcut23_A,
        "bin_width_deg": bin_width_deg,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "stride": stride,
        "n_total_frames_in_pdb": n_total,
        "n_analyzed_frames": traj.n_frames,
        "n_frames_with_angles": n_frames_with_angles,
        "n_angles_total": n_angles_total,
        "n_elem1": len(idx1),
        "n_elem2": len(idx2),
        "n_elem3": len(idx3),
    }

    return df, metadata


def csv_path_for_pdb(outdir, pdb_file, elem1, elem2, elem3, rcut12, rcut23, bin_width):
    stem = safe_name(Path(pdb_file).stem)
    angle_name = safe_name(f"{elem1}-{elem2}-{elem3}")
    fname = (
        f"{stem}_ADF_{angle_name}"
        f"_r12_{rcut12:g}A_r23_{rcut23:g}A_bin_{bin_width:g}deg.csv"
    )
    return Path(outdir) / "csv" / fname


def save_csv_with_metadata(df, metadata, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        df.to_csv(f, index=False)

    print(f"[SAVE] {csv_path}")


def load_adf_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path, comment="#")


def load_reference_txt(txt_path):
    txt_path = Path(txt_path)
    arr = np.loadtxt(txt_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Reference txt must have at least 2 columns: angle_deg density. Got: {txt_path}"
        )
    df = pd.DataFrame({
        "angle_deg": arr[:, 0],
        "density": arr[:, 1],
    })
    return df


def plot_adfs(
    calc_results,
    ref_results,
    elem1,
    elem2,
    elem3,
    fig_path,
    title=None,
    xlim=None,
    ylim=None,
    figsize=(6, 4),
    dpi=300,
):
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)

    for label, df in calc_results:
        plt.plot(df["angle_deg"], df["density"], label=label, linewidth=2.0)

    for label, df in ref_results:
        plt.plot(df["angle_deg"], df["density"], linestyle="--", linewidth=2.0, label=label)

    plt.xlabel(f"Angle {elem1}-{elem2}-{elem3} (degrees)")
    plt.ylabel("Probability density")
    if title is None:
        title = f"ADF: {elem1}-{elem2}-{elem3}"
    plt.title(title)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, 180)

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()

    print(f"[SAVE] {fig_path}")


def save_combined_csv(calc_results, ref_results, combined_csv):
    combined_csv = Path(combined_csv)
    combined_csv.parent.mkdir(parents=True, exist_ok=True)

    combined = None

    for label, df in calc_results:
        col = safe_name(label)
        tmp = df[["angle_deg", "density"]].rename(columns={"density": col})
        if combined is None:
            combined = tmp
        else:
            combined = pd.merge(combined, tmp, on="angle_deg", how="outer")

    for label, df in ref_results:
        col = safe_name(label)
        tmp = df[["angle_deg", "density"]].rename(columns={"density": col})
        if combined is None:
            combined = tmp
        else:
            combined = pd.merge(combined, tmp, on="angle_deg", how="outer")

    if combined is None:
        combined = pd.DataFrame(columns=["angle_deg"])

    combined = combined.sort_values("angle_deg")
    combined.to_csv(combined_csv, index=False)
    print(f"[SAVE] {combined_csv}")


def main():
    args = parse_args()

    pdb_files = expand_patterns(args.pdbs, what="PDB files")
    if not pdb_files:
        raise FileNotFoundError("No PDB files were found from --pdbs.")

    ref_files = expand_patterns(args.reference_txts, what="reference txt files")

    outdir = Path(args.outdir)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)

    if args.labels is None:
        labels = [p.stem for p in pdb_files]
    else:
        labels = args.labels
        if len(labels) != len(pdb_files):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of PDB files ({len(pdb_files)})."
            )

    if args.reference_labels is None:
        ref_labels = [p.stem for p in ref_files]
    else:
        ref_labels = args.reference_labels
        if len(ref_labels) != len(ref_files):
            raise ValueError(
                f"Number of reference labels ({len(ref_labels)}) must match number of reference txt files ({len(ref_files)})."
            )

    angle_name = f"{args.elem1}-{args.elem2}-{args.elem3}"

    if args.fig is None:
        fig_path = outdir / f"ADF_{safe_name(angle_name)}.png"
    else:
        fig_path = Path(args.fig)

    if args.combined_csv is None:
        combined_csv = outdir / f"ADF_{safe_name(angle_name)}_all.csv"
    else:
        combined_csv = Path(args.combined_csv)

    calc_results = []
    ref_results = []

    print("[INFO] PDB files:")
    for p, label in zip(pdb_files, labels):
        print(f"  - {p}  label={label}")

    if ref_files:
        print("[INFO] Reference txt files:")
        for p, label in zip(ref_files, ref_labels):
            print(f"  - {p}  label={label}")

    for pdb_file, label in zip(pdb_files, labels):
        csv_path = csv_path_for_pdb(
            outdir,
            pdb_file,
            args.elem1,
            args.elem2,
            args.elem3,
            args.rcut12,
            args.rcut23,
            args.bin_width,
        )

        if csv_path.exists() and not args.force:
            print(f"[LOAD CSV] {csv_path}")
            df = load_adf_csv(csv_path)
        else:
            if args.plot_only:
                raise FileNotFoundError(
                    f"--plot-only was given, but CSV does not exist: {csv_path}"
                )

            df, metadata = calculate_adf_from_pdb(
                pdb_file=pdb_file,
                elem1=args.elem1,
                elem2=args.elem2,
                elem3=args.elem3,
                rcut12_A=args.rcut12,
                rcut23_A=args.rcut23,
                bin_width_deg=args.bin_width,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                stride=args.stride,
                sel1=args.sel1,
                sel2=args.sel2,
                sel3=args.sel3,
                suppress_warnings=args.suppress_warnings,
            )
            save_csv_with_metadata(df, metadata, csv_path)

        calc_results.append((label, df))

    for txt_file, label in zip(ref_files, ref_labels):
        print(f"[LOAD REF] {txt_file}")
        df = load_reference_txt(txt_file)
        ref_results.append((label, df))

    save_combined_csv(calc_results, ref_results, combined_csv)

    plot_adfs(
        calc_results=calc_results,
        ref_results=ref_results,
        elem1=args.elem1,
        elem2=args.elem2,
        elem3=args.elem3,
        fig_path=fig_path,
        title=args.title,
        xlim=args.xlim,
        ylim=args.ylim,
        figsize=args.figsize,
        dpi=args.dpi,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
