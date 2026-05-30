#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# example: python3 pdb2RDF_analysis_v1.0.0.py --pdbs *.pdb --xlim 0 8 --ylim 0 30 --suppress-warnings

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array


ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "I",
}


def normalize_element(x):
    x = str(x).strip()
    if not x:
        return ""
    if len(x) >= 2:
        cand2 = x[:2].capitalize()
        if cand2 in ELEMENTS:
            return cand2
    cand1 = x[0].upper()
    if cand1 in ELEMENTS:
        return cand1
    return ""


def infer_element_from_atom(atom):
    elem = getattr(atom, "element", "")
    elem = normalize_element(elem)
    if elem:
        return elem

    name = getattr(atom, "name", "")
    name = re.sub(r"[^A-Za-z]", "", str(name))
    return normalize_element(name)


def get_indices_by_element(universe, elem):
    elem = normalize_element(elem)
    indices = []
    for atom in universe.atoms:
        if infer_element_from_atom(atom) == elem:
            indices.append(atom.index)
    return np.array(indices, dtype=int)


def box_volume_from_dimensions(dim):
    lx, ly, lz, alpha, beta, gamma = dim

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    volume = lx * ly * lz * np.sqrt(
        1.0
        + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
    )
    return volume


def get_box(ts, user_box=None):
    if user_box is not None:
        lx, ly, lz = user_box
        return np.array([lx, ly, lz, 90.0, 90.0, 90.0], dtype=float)

    dim = ts.dimensions
    if dim is None:
        return None

    dim = np.asarray(dim, dtype=float)
    if len(dim) < 6:
        return None

    if np.any(dim[:3] <= 0.0):
        return None

    return dim


def calculate_rdf(
    universe,
    elem1,
    elem2,
    start_frame,
    stop_frame=None,
    stride=1,
    rmax=8.0,
    bin_width=0.01,
    user_box=None,
):
    idx1 = get_indices_by_element(universe, elem1)
    idx2 = get_indices_by_element(universe, elem2)

    if len(idx1) == 0:
        raise RuntimeError(f"No atoms found for elem1 = {elem1}")
    if len(idx2) == 0:
        raise RuntimeError(f"No atoms found for elem2 = {elem2}")

    same_group = np.array_equal(idx1, idx2)

    print(f"[INFO] {elem1}: {len(idx1)} atoms")
    print(f"[INFO] {elem2}: {len(idx2)} atoms")

    edges = np.arange(0.0, rmax + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_total = np.zeros(len(centers), dtype=float)

    n_frames_used = 0
    volume_sum = 0.0

    for ts in universe.trajectory[start_frame:stop_frame:stride]:
        box = get_box(ts, user_box=user_box)

        if box is None:
            raise RuntimeError(
                "Box information was not found in the PDB trajectory.\n"
                "RDF normalization requires cell dimensions.\n"
                "If your PDB does not contain CRYST1, specify box manually, e.g.\n"
                "  --box 30.0 30.0 30.0"
            )

        volume = box_volume_from_dimensions(box)

        pos1 = universe.atoms[idx1].positions
        pos2 = universe.atoms[idx2].positions

        dist = distance_array(pos1, pos2, box=box).ravel()

        if same_group:
            dist = dist[dist > 1.0e-8]

        hist, _ = np.histogram(dist, bins=edges)
        hist_total += hist

        volume_sum += volume
        n_frames_used += 1

    if n_frames_used == 0:
        raise RuntimeError("No frames were used. Check --start-ratio, --stop-frame, and --stride.")

    volume_mean = volume_sum / n_frames_used

    n1 = len(idx1)
    n2 = len(idx2)

    if same_group:
        rho2 = (n2 - 1) / volume_mean
    else:
        rho2 = n2 / volume_mean

    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    rdf = hist_total / (n_frames_used * n1 * rho2 * shell_volume)

    df = pd.DataFrame(
        {
            "r": centers,
            "g": rdf,
            "count": hist_total,
        }
    )

    return df


def infer_pair_from_reference_filename(path):
    m = re.match(r"reference_rdf_(.+?)_(.+?)\.txt$", path.name)
    if m is None:
        return None
    return normalize_element(m.group(1)), normalize_element(m.group(2))


def parse_pair(pair_str):
    if "-" in pair_str:
        a, b = pair_str.split("-", 1)
    elif "_" in pair_str:
        a, b = pair_str.split("_", 1)
    else:
        raise ValueError(f"Pair should be like Li-N or Li_N: {pair_str}")

    return normalize_element(a), normalize_element(b)


def load_reference_txt(path):
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Reference RDF file must have two columns: {path}")

    return pd.DataFrame(
        {
            "r": arr[:, 0],
            "g_ref": arr[:, 1],
        }
    )


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x


def plot_multi_rdf(
    curves,
    df_ref,
    elem1,
    elem2,
    outdir,
    xlim=None,
    ylim=None,
    dpi=300,
):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    for label, df_calc in curves:
        ax.plot(
            df_calc["r"],
            df_calc["g"],
            lw=2.0,
            label=label,
        )

    if df_ref is not None:
        ax.plot(
            df_ref["r"],
            df_ref["g_ref"],
            lw=1.8,
            ls="--",
            label="Reference",
        )

    ax.set_xlabel(rf"$r_{{{elem1}-{elem2}}}$ / Å")
    ax.set_ylabel(rf"$g_{{{elem1}-{elem2}}}(r)$")

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    png = outdir / f"rdf_multi_{elem1}_{elem2}_with_reference.png"
    pdf = outdir / f"rdf_multi_{elem1}_{elem2}_with_reference.pdf"

    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"[SAVE FIG] {png}")
    print(f"[SAVE FIG] {pdf}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate RDF for multiple PDB trajectories without imolcraft."
    )

    parser.add_argument(
        "--pdbs",
        nargs="+",
        default=None,
        help="Input PDB trajectory files, e.g. --pdbs run1.pdb run2.pdb run3.pdb",
    )

    parser.add_argument(
        "--pdb",
        default=None,
        help="Single input PDB trajectory. Kept for compatibility.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each PDB curve. Must have same length as --pdbs.",
    )

    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Pairs to analyze, e.g. --pairs Li-N Li-O. If omitted, infer from reference_rdf_*.txt.",
    )

    parser.add_argument(
        "--ref-pattern",
        default="reference_rdf_*.txt",
        help="Reference RDF pattern. Default: reference_rdf_*.txt",
    )

    parser.add_argument(
        "--outdir",
        default="rdf_results",
        help="Output directory. Default: rdf_results",
    )

    parser.add_argument(
        "--start-ratio",
        type=float,
        default=0.5,
        help="Discard initial fraction of frames. Default: 0.5",
    )

    parser.add_argument(
        "--stop-frame",
        type=int,
        default=None,
        help="Stop frame. Default: None",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride. Default: 1",
    )

    parser.add_argument(
        "--rmax",
        type=float,
        default=None,
        help="Maximum RDF distance. If omitted, inferred from reference data or set to 8.0 Å.",
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.01,
        help="RDF bin width in Å. Default: 0.01",
    )

    parser.add_argument(
        "--box",
        nargs=3,
        type=float,
        default=None,
        metavar=("LX", "LY", "LZ"),
        help="Manual orthorhombic box lengths in Å, e.g. --box 30 30 30.",
    )

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        help="x-axis range, e.g. --xlim 0 8",
    )

    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        help="y-axis range, e.g. --ylim 0 30",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recalculation even if csv already exists.",
    )

    parser.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Suppress warnings from PDB parser.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure dpi. Default: 300",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    if args.pdbs is not None:
        pdb_paths = [Path(p) for p in args.pdbs]
    elif args.pdb is not None:
        pdb_paths = [Path(args.pdb)]
    else:
        pdb_paths = [Path("NVE300K_e1.pdb")]

    for p in pdb_paths:
        if not p.exists():
            raise FileNotFoundError(f"PDB file not found: {p}")

    if args.labels is not None:
        if len(args.labels) != len(pdb_paths):
            raise RuntimeError("The number of --labels must match the number of --pdbs.")
        labels = args.labels
    else:
        labels = [p.stem for p in pdb_paths]

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    outdir.mkdir(parents=True, exist_ok=True)
    csvdir.mkdir(parents=True, exist_ok=True)

    ref_files = sorted(Path(".").glob(args.ref_pattern))

    ref_map = {}
    for f in ref_files:
        pair = infer_pair_from_reference_filename(f)
        if pair is not None:
            ref_map[pair] = f

    if args.pairs is None:
        pairs = sorted(ref_map.keys())
        if len(pairs) == 0:
            raise RuntimeError(
                "No --pairs were given and no reference_rdf_*.txt files were found."
            )
    else:
        pairs = [parse_pair(p) for p in args.pairs]

    print("===== Multi-PDB RDF analysis without imolcraft =====")
    print("PDB files:")
    for p, label in zip(pdb_paths, labels):
        print(f"  - {label}: {p}")
    print(f"outdir   : {outdir}")
    print(f"pairs    : {pairs}")
    print(f"force    : {args.force}")

    for elem1, elem2 in pairs:
        print("")
        print(f"===== RDF pair: {elem1}-{elem2} =====")

        ref_path = ref_map.get((elem1, elem2), None)
        if ref_path is None:
            ref_path = ref_map.get((elem2, elem1), None)

        df_ref = None
        if ref_path is not None:
            print(f"[REF] {ref_path}")
            df_ref = load_reference_txt(ref_path)
            df_ref.to_csv(csvdir / f"reference_rdf_{elem1}_{elem2}.csv", index=False)

        if args.rmax is not None:
            rmax = args.rmax
        elif df_ref is not None:
            rmax = float(df_ref["r"].max())
        else:
            rmax = 8.0

        curves = []

        for pdb_path, label in zip(pdb_paths, labels):
            label_safe = safe_name(label)
            csv_path = csvdir / f"rdf_{label_safe}_{elem1}_{elem2}.csv"
            txt_path = csvdir / f"rdf_{label_safe}_{elem1}_{elem2}.txt"

            if csv_path.exists() and not args.force:
                print(f"[LOAD CSV] {csv_path}")
                df_calc = pd.read_csv(csv_path)
            else:
                print(f"[CALC] {label}: {pdb_path}")

                u = mda.Universe(str(pdb_path))
                n_frames = len(u.trajectory)
                start_frame = int(args.start_ratio * n_frames)

                print(f"  frames : {n_frames}")
                print(f"  start  : {start_frame}")
                print(f"  stop   : {args.stop_frame}")
                print(f"  stride : {args.stride}")

                df_calc = calculate_rdf(
                    universe=u,
                    elem1=elem1,
                    elem2=elem2,
                    start_frame=start_frame,
                    stop_frame=args.stop_frame,
                    stride=args.stride,
                    rmax=rmax,
                    bin_width=args.bin_width,
                    user_box=args.box,
                )

                df_calc.to_csv(csv_path, index=False)
                np.savetxt(
                    txt_path,
                    df_calc[["r", "g"]].values,
                    fmt="%.10e",
                )

                print(f"[SAVE CSV] {csv_path}")
                print(f"[SAVE TXT] {txt_path}")

            curves.append((label, df_calc))

        plot_multi_rdf(
            curves=curves,
            df_ref=df_ref,
            elem1=elem1,
            elem2=elem2,
            outdir=outdir,
            xlim=args.xlim,
            ylim=args.ylim,
            dpi=args.dpi,
        )

    print("")
    print("===== Done =====")


if __name__ == "__main__":
    main()
