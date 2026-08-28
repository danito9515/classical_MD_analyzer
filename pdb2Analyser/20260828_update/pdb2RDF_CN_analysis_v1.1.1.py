#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# example: python3 RDF_analysis_multi_pdb_no_imolcraft.py \\

#  --pdbs NVE300K_e1.pdb NVE300K_e2.pdb NVE300K_e3.pdb \\

#  --xlim 0 8 \\

#  --ylim 0 30 \\

#  --suppress-warnings

import argparse

import re

import warnings

from io import StringIO
from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

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





def _read_cryst1_dimensions(pdb_path):
    """Read [a, b, c, alpha, beta, gamma] from the first CRYST1 record."""
    pdb_path = Path(pdb_path)
    with pdb_path.open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                try:
                    vals = [
                        float(line[6:15]),
                        float(line[15:24]),
                        float(line[24:33]),
                        float(line[33:40]),
                        float(line[40:47]),
                        float(line[47:54]),
                    ]
                except ValueError:
                    parts = line.split()
                    if len(parts) < 7:
                        return None
                    try:
                        vals = [float(x) for x in parts[1:7]]
                    except ValueError:
                        return None
                arr = np.asarray(vals, dtype=np.float32)
                if np.any(arr[:3] <= 0.0):
                    return None
                return arr
            if line.startswith("MODEL"):
                # CRYST1 is normally in the header, so do not scan a huge file.
                break
    return None


def _restore_missing_pdb_box(universe, pdb_path):
    """Restore CRYST1 dimensions when a split topology/trajectory load loses them."""
    dims = _read_cryst1_dimensions(pdb_path)
    if dims is None:
        return universe

    def _set_dimensions_if_missing(ts):
        if ts.dimensions is None or np.any(np.asarray(ts.dimensions[:3]) <= 0.0):
            ts.dimensions = dims.copy()
        return ts

    universe.trajectory.add_transformations(_set_dimensions_if_missing)
    # Apply once to the current frame too.
    if universe.trajectory.ts.dimensions is None:
        universe.trajectory.ts.dimensions = dims.copy()
    print(
        "[INFO] Restored CRYST1 box for CONECT-free trajectory load: "
        f"{dims[0]:.6g} {dims[1]:.6g} {dims[2]:.6g} Å, "
        f"{dims[3]:.6g} {dims[4]:.6g} {dims[5]:.6g} deg"
    )
    return universe


def _first_model_topology_without_conect(pdb_path):
    """Build a small PDB topology text from the first model, omitting CONECT.

    Tinker-generated PDB files can contain whitespace-separated CONECT records
    that are readable by Tinker but violate the strict fixed-column PDB layout
    expected by MDAnalysis. RDF/CN calculations only require atom identities,
    coordinates, and the periodic box; bond connectivity is not needed.

    Only the first MODEL is copied, so this does not duplicate a large
    multi-frame trajectory in memory or on disk.
    """
    pdb_path = Path(pdb_path)
    lines = []
    saw_model = False
    in_first_model = False
    atom_count = 0

    with pdb_path.open("r", errors="replace") as fh:
        for line in fh:
            record = line[:6].strip().upper()

            # Bond connectivity is deliberately ignored for RDF/CN analysis.
            if record == "CONECT":
                continue

            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                in_first_model = True
                lines.append(line)
                continue

            if record == "ENDMDL":
                if in_first_model:
                    lines.append(line)
                    break
                continue

            if record in {"ATOM", "HETATM"}:
                atom_count += 1

            # Before MODEL: keep header/CRYST1 records.
            # Inside first MODEL: keep all records except CONECT.
            # For a single-model PDB without MODEL records, keep records until END.
            if (not saw_model) or in_first_model:
                lines.append(line)

            if not saw_model and record == "END":
                break

    if atom_count == 0:
        raise RuntimeError(
            f"Could not build a CONECT-free topology from {pdb_path}: "
            "no ATOM/HETATM records were found."
        )

    # Ensure the in-memory topology has a conventional terminator.
    if not lines or lines[-1].strip().upper() != "END":
        lines.append("END\n")

    return "".join(lines)


def load_pdb_universe(pdb_path, topology_mode="auto"):
    """Load a PDB trajectory, with an automatic fallback for malformed CONECT.

    Parameters
    ----------
    pdb_path : str or Path
        Multi-model PDB trajectory.
    topology_mode : {"auto", "standard", "ignore-conect"}
        auto:
            Try the normal MDAnalysis PDB parser first.  If topology parsing
            fails, retry using a first-model topology with CONECT records removed.
        standard:
            Use the normal MDAnalysis parser only.
        ignore-conect:
            Always use the CONECT-free first-model topology.  This is useful for
            Tinker PDB trajectories and avoids parsing unnecessary bond records.
    """
    pdb_path = Path(pdb_path)

    if topology_mode not in {"auto", "standard", "ignore-conect"}:
        raise ValueError(
            "topology_mode must be one of: auto, standard, ignore-conect"
        )

    if topology_mode in {"auto", "standard"}:
        try:
            return mda.Universe(str(pdb_path))
        except ValueError as exc:
            if topology_mode == "standard":
                raise

            msg = str(exc)
            print("[WARN] Standard MDAnalysis PDB topology parsing failed.")
            print(f"       {msg.splitlines()[-1] if msg else exc}")
            print("[WARN] Retrying with CONECT records ignored (safe for RDF/CN).")

    topology_text = _first_model_topology_without_conect(pdb_path)
    topology_stream = StringIO(topology_text)

    try:
        u = mda.Universe(
            topology_stream,
            str(pdb_path),
            topology_format="PDB",
            format="PDB",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {pdb_path} even after ignoring CONECT records."
        ) from exc

    u = _restore_missing_pdb_box(u, pdb_path)
    print(f"[INFO] Loaded {pdb_path} with CONECT-free topology.")
    return u


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

    # Running coordination number around elem1 due to elem2.
    # Discrete equivalent of:
    #   CN_12(r) = 4*pi*rho_2 * integral_0^r g_12(r') r'^2 dr'
    coordination_number = np.cumsum(hist_total) / (n_frames_used * n1)

    df = pd.DataFrame(

        {

            "r": centers,

            "g": rdf,

            "cn": coordination_number,

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

    x = re.sub(r"[^\w.-]+", "_", x)

    return x



def configure_matplotlib(font_family="DejaVu Sans"):

    """Set publication-friendly defaults without requiring a specific external font."""

    plt.rcParams.update(

        {

            "font.family": font_family,

            "axes.unicode_minus": False,

            "pdf.fonttype": 42,

            "ps.fonttype": 42,

            "svg.fonttype": "none",

        }

    )



def plot_multi_rdf(
    curves,
    df_ref,
    elem1,
    elem2,
    outdir,
    xlim=None,
    ylim=None,
    with_cn=False,
    cn_ylim=None,
    figsize=(6.4, 4.8),
    title=None,
    no_title=False,
    xlabel=None,
    ylabel=None,
    cn_ylabel=None,
    font_family="DejaVu Sans",
    font_weight="normal",
    axis_label_size=18,
    tick_label_size=15,
    title_size=18,
    title_weight="bold",
    legend_size=12,
    legend_loc="best",
    legend_ncol=1,
    legend_frame=False,
    legend_title=None,
    legend_title_size=12,
    line_width=2.5,
    cn_line_width=2.0,
    cn_line_style="--",
    cn_alpha=0.90,
    reference_line_width=2.5,
    reference_label="Reference",
    axis_line_width=1.4,
    tick_width=1.2,
    tick_length=5.0,
    xtick_step=None,
    ytick_step=None,
    cn_tick_step=None,
    show_grid=True,
    grid_alpha=0.25,
    minor_ticks=False,
    dpi=300,
):
    configure_matplotlib(font_family)

    fig, ax = plt.subplots(figsize=figsize)
    ax_cn = ax.twinx() if with_cn else None

    for label, df_calc in curves:
        line, = ax.plot(
            df_calc["r"],
            df_calc["g"],
            linewidth=line_width,
            linestyle="-",
            label=label,
        )

        if with_cn:
            if "cn" not in df_calc.columns:
                raise RuntimeError(
                    f"CN data are missing for curve '{label}'. "
                    "Re-run with --force so RDF/CN is recalculated."
                )
            ax_cn.plot(
                df_calc["r"],
                df_calc["cn"],
                linewidth=cn_line_width,
                linestyle=cn_line_style,
                alpha=cn_alpha,
                color=line.get_color(),
                label="_nolegend_",
            )

    if df_ref is not None:
        ax.plot(
            df_ref["r"],
            df_ref["g_ref"],
            linewidth=reference_line_width,
            linestyle="--",
            label=reference_label,
        )

    if xlabel is None:
        xlabel = rf"$r_{{\mathrm{{{elem1}-{elem2}}}}}$ (Å)"
    if ylabel is None:
        ylabel = rf"$g_{{\mathrm{{{elem1}-{elem2}}}}}(r)$"
    if cn_ylabel is None:
        cn_ylabel = rf"$CN_{{\mathrm{{{elem1}-{elem2}}}}}(r)$"

    ax.set_xlabel(xlabel, fontsize=axis_label_size, fontweight=font_weight, labelpad=7)
    ax.set_ylabel(ylabel, fontsize=axis_label_size, fontweight=font_weight, labelpad=7)

    if with_cn:
        ax_cn.set_ylabel(
            cn_ylabel,
            fontsize=axis_label_size,
            fontweight=font_weight,
            labelpad=7,
        )

    if not no_title:
        if title is None:
            title = f"RDF + CN: {elem1}-{elem2}" if with_cn else f"RDF: {elem1}-{elem2}"
        ax.set_title(title, fontsize=title_size, fontweight=title_weight, pad=10)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if with_cn and cn_ylim is not None:
        ax_cn.set_ylim(cn_ylim[0], cn_ylim[1])

    if xtick_step is not None:
        if xtick_step <= 0:
            raise ValueError("--xtick-step must be greater than zero.")
        ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
    if ytick_step is not None:
        if ytick_step <= 0:
            raise ValueError("--ytick-step must be greater than zero.")
        ax.yaxis.set_major_locator(MultipleLocator(ytick_step))
    if with_cn and cn_tick_step is not None:
        if cn_tick_step <= 0:
            raise ValueError("--cn-tick-step must be greater than zero.")
        ax_cn.yaxis.set_major_locator(MultipleLocator(cn_tick_step))

    if minor_ticks:
        ax.minorticks_on()
        if with_cn:
            ax_cn.minorticks_on()

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=not with_cn,
        width=tick_width,
        length=tick_length,
        labelsize=tick_label_size,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=not with_cn,
        width=max(0.8, 0.8 * tick_width),
        length=max(2.0, 0.6 * tick_length),
    )

    if with_cn:
        ax_cn.tick_params(
            axis="y",
            which="major",
            direction="in",
            right=True,
            left=False,
            width=tick_width,
            length=tick_length,
            labelsize=tick_label_size,
        )
        ax_cn.tick_params(
            axis="y",
            which="minor",
            direction="in",
            right=True,
            left=False,
            width=max(0.8, 0.8 * tick_width),
            length=max(2.0, 0.6 * tick_length),
        )

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight(font_weight)
    if with_cn:
        for tick_label in ax_cn.get_yticklabels():
            tick_label.set_fontweight(font_weight)

    for spine in ax.spines.values():
        spine.set_linewidth(axis_line_width)
    if with_cn:
        ax_cn.spines["right"].set_linewidth(axis_line_width)

    if show_grid:
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linewidth=0.7, alpha=grid_alpha)

    legend = ax.legend(
        loc=legend_loc,
        ncol=legend_ncol,
        frameon=legend_frame,
        fontsize=legend_size,
        title=legend_title,
        handlelength=2.5,
        columnspacing=1.2,
        labelspacing=0.45,
        borderaxespad=0.6,
    )

    if legend is not None:
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_title_size)
            legend.get_title().set_fontweight(font_weight)
        for text in legend.get_texts():
            text.set_fontweight(font_weight)

    fig.tight_layout()

    suffix = "rdf_cn" if with_cn else "rdf"
    png = outdir / f"{suffix}_multi_{elem1}_{elem2}_with_reference.png"
    pdf = outdir / f"{suffix}_multi_{elem1}_{elem2}_with_reference.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE FIG] {png}")
    print(f"[SAVE FIG] {pdf}")


def parse_args():

    parser = argparse.ArgumentParser(

        description="Calculate RDF and running coordination number for multiple PDB trajectories without imolcraft."

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

        "--pdb-topology-mode",

        choices=("auto", "standard", "ignore-conect"),

        default="auto",

        help=(
            "How to parse PDB topology. auto (default) retries without CONECT "
            "if MDAnalysis rejects malformed Tinker CONECT records; standard "
            "disables the fallback; ignore-conect always skips bond records."
        ),

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

        "--with-cn",

        action="store_true",

        help="Plot running coordination number on a right y-axis.",

    )

    parser.add_argument(

        "--cn-ylim",

        nargs=2,

        type=float,

        default=None,

        metavar=("YMIN", "YMAX"),

        help="Right-axis CN range, e.g. --cn-ylim 0 5",

    )

    parser.add_argument(

        "--cn-ylabel",

        type=str,

        default=None,

        help="Custom right-axis label for coordination number.",

    )



    # Figure style

    parser.add_argument(

        "--figsize",

        nargs=2,

        type=float,

        default=(6.4, 4.8),

        metavar=("WIDTH", "HEIGHT"),

        help="Figure size in inches. Default: 6.4 4.8",

    )

    parser.add_argument(

        "--title",

        type=str,

        default=None,

        help="Custom figure title. By default, the RDF pair is used.",

    )

    parser.add_argument(

        "--no-title",

        action="store_true",

        help="Do not show a figure title.",

    )

    parser.add_argument(

        "--xlabel",

        type=str,

        default=None,

        help="Custom x-axis label.",

    )

    parser.add_argument(

        "--ylabel",

        type=str,

        default=None,

        help="Custom y-axis label.",

    )

    parser.add_argument(

        "--font-family",

        type=str,

        default="DejaVu Sans",

        help='Font family. Default: "DejaVu Sans".',

    )

    parser.add_argument(

        "--font-weight",

        choices=("normal", "bold"),

        default="normal",

        help="Weight for axis labels, ticks, and legend. Default: normal",

    )

    parser.add_argument(

        "--axis-label-size",

        type=float,

        default=18,

        help="Axis-label font size. Default: 18",

    )

    parser.add_argument(

        "--tick-label-size",

        type=float,

        default=15,

        help="Tick-label font size. Default: 15",

    )

    parser.add_argument(

        "--title-size",

        type=float,

        default=18,

        help="Title font size. Default: 18",

    )

    parser.add_argument(

        "--title-weight",

        choices=("normal", "bold"),

        default="bold",

        help="Title font weight. Default: bold",

    )

    parser.add_argument(

        "--legend-size",

        type=float,

        default=12,

        help="Legend font size. Default: 12",

    )

    parser.add_argument(

        "--legend-loc",

        type=str,

        default="best",

        help='Matplotlib legend location, e.g. "best", "upper right". Default: best',

    )

    parser.add_argument(

        "--legend-ncol",

        type=int,

        default=1,

        help="Number of legend columns. Default: 1",

    )

    parser.add_argument(

        "--legend-frame",

        action="store_true",

        help="Draw a frame around the legend.",

    )

    parser.add_argument(

        "--legend-title",

        type=str,

        default=None,

        help="Optional legend title.",

    )

    parser.add_argument(

        "--legend-title-size",

        type=float,

        default=12,

        help="Legend-title font size. Default: 12",

    )

    parser.add_argument(

        "--reference-label",

        type=str,

        default="Reference",

        help='Legend label for the reference RDF. Default: "Reference"',

    )

    parser.add_argument(

        "--line-width",

        type=float,

        default=2.5,

        help="Line width of calculated RDF curves. Default: 2.5",

    )

    parser.add_argument(

        "--cn-line-width",

        type=float,

        default=2.0,

        help="Line width of CN curves. Default: 2.0",

    )

    parser.add_argument(

        "--cn-line-style",

        type=str,

        default="--",

        help='Line style of CN curves. Default: "--"',

    )

    parser.add_argument(

        "--cn-alpha",

        type=float,

        default=0.90,

        help="Opacity of CN curves. Default: 0.90",

    )

    parser.add_argument(

        "--reference-line-width",

        type=float,

        default=2.5,

        help="Line width of the reference RDF. Default: 2.5",

    )

    parser.add_argument(

        "--axis-line-width",

        type=float,

        default=1.4,

        help="Width of plot spines. Default: 1.4",

    )

    parser.add_argument(

        "--tick-width",

        type=float,

        default=1.2,

        help="Major tick width. Default: 1.2",

    )

    parser.add_argument(

        "--tick-length",

        type=float,

        default=5.0,

        help="Major tick length. Default: 5.0",

    )

    parser.add_argument(

        "--xtick-step",

        type=float,

        default=None,

        help="Major x-tick interval, e.g. --xtick-step 1.0",

    )

    parser.add_argument(

        "--ytick-step",

        type=float,

        default=None,

        help="Major y-tick interval, e.g. --ytick-step 5.0",

    )

    parser.add_argument(

        "--cn-tick-step",

        type=float,

        default=None,

        help="Major CN-axis tick interval, e.g. --cn-tick-step 1.0",

    )

    parser.add_argument(

        "--minor-ticks",

        action="store_true",

        help="Show minor ticks.",

    )

    parser.add_argument(

        "--no-grid",

        action="store_true",

        help="Disable the major grid.",

    )

    parser.add_argument(

        "--grid-alpha",

        type=float,

        default=0.25,

        help="Grid transparency. Default: 0.25",

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

    print("===== Multi-PDB RDF + coordination-number analysis without imolcraft (v1.1.1) =====")

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

            rdf_cn_txt_path = csvdir / f"rdf_cn_{label_safe}_{elem1}_{elem2}.txt"

            use_cached = csv_path.exists() and not args.force

            if use_cached:

                df_calc = pd.read_csv(csv_path)

                if args.with_cn and "cn" not in df_calc.columns:

                    print(f"[RECALC] CN column missing in old CSV: {csv_path}")

                    use_cached = False

            if use_cached:

                print(f"[LOAD CSV] {csv_path}")

            else:

                print(f"[CALC] {label}: {pdb_path}")

                u = load_pdb_universe(
                    pdb_path, topology_mode=args.pdb_topology_mode
                )

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

                np.savetxt(

                    rdf_cn_txt_path,

                    df_calc[["r", "g", "cn"]].values,

                    fmt="%.10e",

                    header="r_A  g_r  CN_r",

                )

                print(f"[SAVE CSV] {csv_path}")

                print(f"[SAVE TXT] {txt_path}")

                print(f"[SAVE TXT] {rdf_cn_txt_path}")

            curves.append((label, df_calc))

        plot_multi_rdf(

            curves=curves,

            df_ref=df_ref,

            elem1=elem1,

            elem2=elem2,

            outdir=outdir,

            xlim=args.xlim,

            ylim=args.ylim,

            with_cn=args.with_cn,

            cn_ylim=args.cn_ylim,

            figsize=args.figsize,

            title=args.title,

            no_title=args.no_title,

            xlabel=args.xlabel,

            ylabel=args.ylabel,

            cn_ylabel=args.cn_ylabel,

            font_family=args.font_family,

            font_weight=args.font_weight,

            axis_label_size=args.axis_label_size,

            tick_label_size=args.tick_label_size,

            title_size=args.title_size,

            title_weight=args.title_weight,

            legend_size=args.legend_size,

            legend_loc=args.legend_loc,

            legend_ncol=args.legend_ncol,

            legend_frame=args.legend_frame,

            legend_title=args.legend_title,

            legend_title_size=args.legend_title_size,

            line_width=args.line_width,

            cn_line_width=args.cn_line_width,

            cn_line_style=args.cn_line_style,

            cn_alpha=args.cn_alpha,

            reference_line_width=args.reference_line_width,

            reference_label=args.reference_label,

            axis_line_width=args.axis_line_width,

            tick_width=args.tick_width,

            tick_length=args.tick_length,

            xtick_step=args.xtick_step,

            ytick_step=args.ytick_step,

            cn_tick_step=args.cn_tick_step,

            show_grid=not args.no_grid,

            grid_alpha=args.grid_alpha,

            minor_ticks=args.minor_ticks,

            dpi=args.dpi,

        )

    print("")

    print("===== Done =====")



if __name__ == "__main__":

    main()
