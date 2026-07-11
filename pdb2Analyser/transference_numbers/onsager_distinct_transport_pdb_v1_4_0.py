#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onsager_distinct_transport_pdb_v1_2_0.py
========================================
Onsager/Einstein analysis of ionic-transport correlations from one or more PDB trajectories.

The script decomposes the conductivity into

  * self terms for each charged species,
  * same-species distinct terms (a-a, i != j),
  * unlike-species distinct terms (a-b), and
  * the total correlated conductivity.

For species a and b, define the collective displacement correlation

    C_ab(t) = < [sum_i in a Delta r_i(t)] . [sum_j in b Delta r_j(t)] >

For a = b,

    C_aa(t) = C_aa^self(t) + C_aa^distinct(t)

where C_aa^self is the sum of single-particle MSDs.  The conductivity
contribution is obtained from the long-time slope:

    sigma_ab = e^2 z_a z_b / (6 V k_B T) * d C_ab / dt

For unlike species, the total conductivity contains both ordered terms ab and
ba.  Therefore the decomposition plot and total use 2*sigma_ab for each
unordered pair a<b.

The formulation follows the Einstein representation of the Onsager matrix and
the self/distinct decomposition used in Sasaki et al., npj Comput. Mater. 9,
48 (2023), especially Eq. (1) and Supplementary Eqs. (17)-(24), (42)-(46).

Typical LiFSA/SN example
------------------------
python3 onsager_distinct_transport_pdb_v1_1_0.py \
  --pdb Liquid_nvt300K_90ns.pdb \
  --dt-ns 0.1 --temperature-k 300 \
  --track-element Li:Li:+1 \
  --track-residue-com FSA:FSA,FSI,TFSI,NFS:-1 \
  --fit-start-ns 5 --fit-end-ns 30 --max-lag-ns 30 \
  --remove-drift all \
  --outdir onsager_results

Atom-type proxy example
-----------------------
python3 onsager_distinct_transport_pdb_v1_1_0.py \
  --pdb NVT_T300K_all.pdb --dt-ns 0.1 --temperature-k 300 \
  --atom-type-field occupancy \
  --track-element Li:Li:+1 \
  --track-atom-type FSA_N:72:-1 \
  --fit-start-ns 5 --fit-end-ns 30 --max-lag-ns 30 \
  --outdir onsager_results

Notes
-----
* Molecular ions should preferably be represented by their molecular COM
  (--track-residue-com), not by a single atom proxy.
* Coordinates are unwrapped with an orthorhombic CRYST1 box.
* Translational drift removal is strongly recommended for collective terms.
* A negative distinct contribution is physically allowed and indicates
  anticorrelated charge transport for that contribution.
* In addition to conductivity decomposition, the script writes SI-like MSD plots:
  self terms are visualized as MSD_tr, while distinct and direct terms are
  visualized as MSD_sigma-style charge-weighted correlations.
"""

import argparse
import glob
import re
import csv
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

E_CHARGE_C = 1.602176634e-19
K_B_J_PER_K = 1.380649e-23

MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "S": 32.06, "Li": 6.94, "Na": 22.990,
    "K": 39.098, "Cl": 35.45, "P": 30.974, "Ge": 72.630,
    "La": 138.905, "Zr": 91.224,
}


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def safe_label(text):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_") or "species"


def infer_element(name, element_field=""):
    e = element_field.strip()
    if e:
        e = e[0].upper() + e[1:].lower()
        if e in MASS:
            return e
    letters = "".join(c for c in name if c.isalpha())
    if len(letters) >= 2 and letters[:2].title() in MASS:
        return letters[:2].title()
    if letters and letters[0].upper() in MASS:
        return letters[0].upper()
    return "C"


def _int_like(text, allow_small=False):
    s = str(text).strip()
    if not s:
        return None
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
        val = int(round(float(s)))
        if allow_small or abs(val) > 1:
            return val
    return None


def parse_atom_type(line, field="auto"):
    field = field.lower()
    if field == "none":
        return None

    def occ(allow=False):
        return _int_like(line[54:60] if len(line) >= 60 else "", allow)

    def bfac(allow=False):
        return _int_like(line[60:66] if len(line) >= 66 else "", allow)

    def tail():
        for tok in (line[66:].split() if len(line) > 66 else []):
            val = _int_like(tok, False)
            if val is not None:
                return val
        return None

    def last_int():
        vals = [_int_like(tok, False) for tok in line.split()]
        vals = [v for v in vals if v is not None]
        return vals[-1] if vals else None

    if field == "occupancy":
        return occ(True)
    if field == "bfactor":
        return bfac(True)
    if field == "resname":
        return _int_like(line[17:21] if len(line) >= 21 else "", allow_small=True)
    if field == "tail":
        return tail()
    if field == "last_int":
        return last_int()
    if field != "auto":
        raise ValueError(f"Unknown atom-type field: {field}")
    for fn in (tail, bfac, occ, last_int):
        val = fn()
        if val is not None:
            return val
    return None


def parse_cryst1(line):
    try:
        box = np.array([float(line[6:15]), float(line[15:24]), float(line[24:33])])
        ang = np.array([float(line[33:40]), float(line[40:47]), float(line[47:54])])
        return box, ang
    except Exception:
        return None, None


def read_topology(path, atom_type_field):
    atoms, first_box = [], None
    has_model = False
    with open_text(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                has_model = True
            elif line.startswith("CRYST1") and first_box is None:
                first_box, _ = parse_cryst1(line)
            elif line.startswith(("ATOM  ", "HETATM")):
                name = line[12:16].strip()
                atoms.append({
                    "ordinal": len(atoms),
                    "name": name,
                    "resname": line[17:21].strip(),
                    "chain": line[21:22].strip(),
                    "resid": line[22:27].strip(),
                    "element": infer_element(name, line[76:78] if len(line) >= 78 else ""),
                    "atom_type": parse_atom_type(line, atom_type_field),
                })
            elif line.startswith("ENDMDL") and atoms:
                break
    if not atoms:
        raise RuntimeError(f"No atoms found in {path}")
    return atoms, has_model, first_box


def iter_frames(path, natoms, selected, has_model, first_box):
    selected = sorted(selected)
    loc = {o: i for i, o in enumerate(selected)}
    coords = np.empty((len(selected), 3), np.float32)
    atom_i = found = 0
    box = first_box
    with open_text(path) as f:
        for line in f:
            if line.startswith("CRYST1"):
                b, _ = parse_cryst1(line)
                if b is not None:
                    box = b
            elif line.startswith("MODEL"):
                atom_i = found = 0
                coords = np.empty((len(selected), 3), np.float32)
            elif line.startswith(("ATOM  ", "HETATM")):
                if atom_i in loc:
                    j = loc[atom_i]
                    coords[j] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    found += 1
                atom_i += 1
                if not has_model and atom_i == natoms:
                    if found == len(selected):
                        yield coords.copy(), None if box is None else box.copy()
                    atom_i = found = 0
                    coords = np.empty((len(selected), 3), np.float32)
            elif line.startswith("ENDMDL"):
                if found == len(selected):
                    yield coords.copy(), None if box is None else box.copy()
                atom_i = found = 0
                coords = np.empty((len(selected), 3), np.float32)


def unwrap_orthorhombic(pos, boxes):
    out = pos.astype(np.float64, copy=True)
    if boxes is None or np.any(~np.isfinite(boxes)) or np.any(boxes <= 0):
        return out
    for t in range(1, len(out)):
        d = pos[t].astype(float) - pos[t - 1].astype(float)
        d -= boxes[t] * np.round(d / boxes[t])
        out[t] = out[t - 1] + d
    return out


def residue_key(a):
    return a["chain"], a["resid"], a["resname"]


def parse_triplet(spec, what):
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid {what}: {spec}; expected LABEL:SELECTOR:CHARGE")
    return safe_label(parts[0]), parts[1], float(parts[2])


def build_species(atoms, args):
    species = []

    for spec in args.track_element:
        label, elem, charge = parse_triplet(spec, "--track-element")
        ords = [a["ordinal"] for a in atoms if a["element"].lower() == elem.lower()]
        species.append({"label": label, "charge": charge, "kind": "atoms", "groups": [[o] for o in ords]})

    for spec in args.track_atom_type:
        label, text, charge = parse_triplet(spec, "--track-atom-type")
        types = {int(x) for x in re.split(r"[,;]", text) if x.strip()}
        ords = [a["ordinal"] for a in atoms if a["atom_type"] in types]
        species.append({"label": label, "charge": charge, "kind": "atoms", "groups": [[o] for o in ords]})

    residues = defaultdict(list)
    for a in atoms:
        residues[residue_key(a)].append(a["ordinal"])

    for spec in args.track_residue_com:
        label, text, charge = parse_triplet(spec, "--track-residue-com")
        names = {x.strip() for x in re.split(r"[,;]", text) if x.strip()}
        groups = [ords for (chain, resid, resname), ords in residues.items() if resname in names]
        species.append({"label": label, "charge": charge, "kind": "com", "groups": groups})

    labels = [s["label"] for s in species]
    if len(set(labels)) != len(labels):
        raise ValueError("Species labels must be unique")
    for s in species:
        if not s["groups"]:
            raise RuntimeError(f"No particles found for species {s['label']}")
    if len(species) < 2:
        raise RuntimeError("Define at least two charged species to evaluate unlike-species distinct terms")
    return species


def print_topology(atoms):
    print("[elements]", dict(Counter(a["element"] for a in atoms)))
    print("[resnames]", dict(Counter(a["resname"] or "<blank>" for a in atoms)))
    types = Counter(a["atom_type"] for a in atoms if a["atom_type"] is not None)
    print("[atom types]", dict(types) if types else "<none>")


def msd_fft_particles(pos, max_lag):
    """Mean MSD over particles; pos shape (T,N,3)."""
    pos = np.asarray(pos, float)
    T, N, dim = pos.shape
    max_lag = min(max_lag, T - 1)
    x = pos.reshape(T, N * dim)
    q = np.sum(pos * pos, axis=(1, 2))
    prefix = np.concatenate([[0.0], np.cumsum(q)])
    nfft = 1 << ((2 * T - 1).bit_length())
    f = np.fft.rfft(x, n=nfft, axis=0)
    ac = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=0)[:max_lag + 1].sum(axis=1)
    lag = np.arange(max_lag + 1)
    count = T - lag
    out = (prefix[T - lag] + (prefix[T] - prefix[lag]) - 2 * ac) / (count * N)
    out[0] = 0.0
    return out


def msd_fft_vector(vec, max_lag):
    return msd_fft_particles(np.asarray(vec, float)[:, None, :], max_lag)


def cross_displacement_fft(a, b, max_lag):
    """<Delta a . Delta b> by polarization identity."""
    return 0.5 * (msd_fft_vector(a + b, max_lag) - msd_fft_vector(a, max_lag) - msd_fft_vector(b, max_lag))


def fit_slope(time_ns, y, start_ns, end_ns):
    mask = (time_ns >= start_ns) & (time_ns <= end_ns) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(time_ns[mask], y[mask], 1)
    pred = slope * time_ns[mask] + intercept
    ss_res = np.sum((y[mask] - pred) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def expanding_slope(time_ns, y, start_ns, min_points=8):
    out = np.full_like(time_ns, np.nan, dtype=float)
    start_i = int(np.searchsorted(time_ns, start_ns, side="left"))
    for end_i in range(start_i + min_points - 1, len(time_ns)):
        x = time_ns[start_i:end_i + 1]
        yy = y[start_i:end_i + 1]
        m = np.isfinite(yy)
        if m.sum() >= min_points:
            out[end_i] = np.polyfit(x[m], yy[m], 1)[0]
    return out


def slope_to_sigma(slope_A2_ns, za, zb, volume_A3, temperature_k, dimension=3):
    if not np.isfinite(slope_A2_ns):
        return np.nan
    slope_m2_s = slope_A2_ns * 1.0e-11
    volume_m3 = volume_A3 * 1.0e-30
    return (E_CHARGE_C ** 2) * za * zb * slope_m2_s / ((2.0 * dimension) * volume_m3 * K_B_J_PER_K * temperature_k)


def setup_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": args.font_size,
        "axes.labelsize": args.axis_label_size,
        "xtick.labelsize": args.tick_label_size,
        "ytick.labelsize": args.tick_label_size,
        "legend.fontsize": args.legend_font_size,
        "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def term_label_for_plot(key):
    if key.startswith("self:"):
        lab = key.split(":", 1)[1]
        return f"{lab} self"
    if key.startswith("distinct:"):
        pair = key.split(":", 1)[1]
        a, b = pair.split("-", 1)
        return f"{a}-{b} distinct"
    if key == "total:charge":
        return "total charge"
    return key


def _error_from_blocks(arr, mode="sem"):
    """Return mean and uncertainty across independent contiguous blocks."""
    arr = np.asarray(arr, float)
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] < 2:
        return mean, np.full_like(mean, np.nan)
    sd = np.nanstd(arr, axis=0, ddof=1)
    if mode == "sd":
        err = sd
    elif mode == "ci95":
        err = 1.96 * sd / np.sqrt(arr.shape[0])
    else:
        err = sd / np.sqrt(arr.shape[0])
    return mean, err


def compute_block_msd_statistics(particle_pos, species, dt_eff, requested_max_lag, args):
    """Compute block-averaged MSD_tr and MSD_sigma curves.

    The trajectory is divided into non-overlapping contiguous blocks. Each
    block is treated as one statistical replicate. Error bars therefore
    represent between-block variability, not the strongly correlated spread
    over individual time origins.
    """
    nframes = next(iter(particle_pos.values())).shape[0]
    nblocks = int(args.msd_blocks)
    if nblocks < 2:
        return None
    block_len = nframes // nblocks
    if block_len < 4:
        print(f"[WARN] MSD block averaging disabled: block length is only {block_len} frames")
        return None
    block_max_lag = min(requested_max_lag, block_len - 1)
    if args.block_max_lag_ns is not None:
        block_max_lag = min(block_max_lag, int(round(args.block_max_lag_ns / dt_eff)))
    if block_max_lag < 1:
        print("[WARN] MSD block averaging disabled: block max lag is < 1 frame")
        return None

    self_samples = defaultdict(list)
    distinct_samples = defaultdict(list)
    direct_samples = []

    for ib in range(nblocks):
        i0 = ib * block_len
        i1 = i0 + block_len
        block_pos = {lab: p[i0:i1] for lab, p in particle_pos.items()}

        for sp in species:
            lab, z = sp["label"], sp["charge"]
            pp = block_pos[lab]
            n = pp.shape[1]
            tracer = msd_fft_particles(pp, block_max_lag)
            self_sum = tracer * n
            collective = msd_fft_vector(pp.sum(axis=1), block_max_lag)
            distinct = collective - self_sum
            self_samples[lab].append(tracer)
            distinct_samples[f"{lab}-{lab}"].append((z * z) * distinct)

        for ia in range(len(species)):
            for ib2 in range(ia + 1, len(species)):
                a, b = species[ia], species[ib2]
                va = block_pos[a["label"]].sum(axis=1)
                vb = block_pos[b["label"]].sum(axis=1)
                cross = cross_displacement_fft(va, vb, block_max_lag)
                distinct_samples[f"{a['label']}-{b['label']}"] .append(
                    2.0 * a["charge"] * b["charge"] * cross
                )

        qvec = np.zeros((block_len, 3), float)
        for sp in species:
            qvec += sp["charge"] * block_pos[sp["label"]].sum(axis=1)
        direct_samples.append(msd_fft_vector(qvec, block_max_lag))

    self_stats = {}
    for lab, samples in self_samples.items():
        mean, err = _error_from_blocks(np.stack(samples), args.msd_error_stat)
        self_stats[lab] = {"mean": mean, "err": err}
    distinct_stats = {}
    for lab, samples in distinct_samples.items():
        mean, err = _error_from_blocks(np.stack(samples), args.msd_error_stat)
        distinct_stats[lab] = {"mean": mean, "err": err}
    direct_mean, direct_err = _error_from_blocks(np.stack(direct_samples), args.msd_error_stat)

    return {
        "time_ns": np.arange(block_max_lag + 1, dtype=float) * dt_eff,
        "self": self_stats,
        "distinct": distinct_stats,
        "direct": {"mean": direct_mean, "err": direct_err},
        "n_blocks": nblocks,
        "block_frames": block_len,
        "block_duration_ns": block_len * dt_eff,
        "error_stat": args.msd_error_stat,
    }


def _species_tag(labels):
    return safe_label("_".join(labels))


def _plot_mean_error(plt, x, mean, err, label, args, loglog=False, abs_for_log=False):
    mean = np.asarray(mean, float)
    err = np.asarray(err, float)
    if loglog:
        yy = np.abs(mean) if abs_for_log else mean
        mask = (x > 0) & np.isfinite(yy) & (yy > 0)
        if not np.any(mask):
            return
        plt.loglog(x[mask], yy[mask], lw=args.line_width, label=label)
        lo = np.maximum(yy - np.nan_to_num(err, nan=0.0), np.finfo(float).tiny)
        hi = yy + np.nan_to_num(err, nan=0.0)
        plt.fill_between(x[mask], lo[mask], hi[mask], alpha=args.error_alpha)
    else:
        plt.plot(x, mean, lw=args.line_width, label=label)
        if np.any(np.isfinite(err)):
            plt.fill_between(x, mean - err, mean + err, alpha=args.error_alpha)


def write_grouped_msd_figures(outdir, stem, time_ns, self_curves, distinct_curves,
                              direct_curve, species_labels, args, fit_end_ns,
                              block_stats=None):
    """Write SI-like MSD figures with block-average uncertainty when available."""
    plt = setup_plot(args)
    species_tag = _species_tag(species_labels)
    distinct_tag = _species_tag(list(distinct_curves.keys()))
    direct_label = " + ".join(species_labels) + " total charge"

    if block_stats is not None:
        xerr = block_stats["time_ns"]
        self_plot = block_stats["self"]
        distinct_plot = block_stats["distinct"]
        direct_plot = block_stats["direct"]
    else:
        xerr = time_ns
        self_plot = {k: {"mean": v, "err": np.full_like(v, np.nan)} for k, v in self_curves.items()}
        distinct_plot = {k: {"mean": v, "err": np.full_like(v, np.nan)} for k, v in distinct_curves.items()}
        direct_plot = {"mean": direct_curve, "err": np.full_like(direct_curve, np.nan)}

    if self_plot:
        plt.figure(figsize=(args.fig_width, args.fig_height))
        for lab, stat in self_plot.items():
            _plot_mean_error(plt, xerr, stat["mean"], stat["err"], lab, args)
        plt.axvspan(args.fit_start_ns, min(fit_end_ns, xerr[-1]), alpha=0.12)
        plt.xlabel("Time / ns")
        plt.ylabel(r"MSD$_{tr}$ / $\AA^2$")
        plt.axhline(0, lw=0.8)
        plt.legend(frameon=False)
        plt.tight_layout()
        p = Path(outdir) / f"{stem}_onsager_self_MSDtr_{species_tag}.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")

        plt.figure(figsize=(args.fig_width, args.fig_height))
        for lab, stat in self_plot.items():
            _plot_mean_error(plt, xerr, stat["mean"], stat["err"], lab, args, loglog=True)
        plt.xlabel("Time / ns")
        plt.ylabel(r"MSD$_{tr}$ / $\AA^2$")
        plt.legend(frameon=False)
        plt.tight_layout()
        p = Path(outdir) / f"{stem}_onsager_self_MSDtr_{species_tag}_loglog.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")

    if distinct_plot:
        plt.figure(figsize=(args.fig_width, args.fig_height))
        for lab, stat in distinct_plot.items():
            _plot_mean_error(plt, xerr, stat["mean"], stat["err"], f"{lab} distinct", args)
        plt.axvspan(args.fit_start_ns, min(fit_end_ns, xerr[-1]), alpha=0.12)
        plt.xlabel("Time / ns")
        plt.ylabel(r"MSD$_{\sigma}$ / $\AA^2$")
        plt.axhline(0, lw=0.8)
        plt.legend(frameon=False)
        plt.tight_layout()
        p = Path(outdir) / f"{stem}_onsager_distinct_MSDsigma_{distinct_tag}.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")

        plt.figure(figsize=(args.fig_width, args.fig_height))
        for lab, stat in distinct_plot.items():
            _plot_mean_error(plt, xerr, stat["mean"], stat["err"], f"|{lab} distinct|", args,
                             loglog=True, abs_for_log=True)
        plt.xlabel("Time / ns")
        plt.ylabel(r"|MSD$_{\sigma}$| / $\AA^2$")
        plt.legend(frameon=False)
        plt.tight_layout()
        p = Path(outdir) / f"{stem}_onsager_distinct_MSDsigma_{distinct_tag}_abs_loglog.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")

    if direct_plot is not None:
        plt.figure(figsize=(args.fig_width, args.fig_height))
        _plot_mean_error(plt, xerr, direct_plot["mean"], direct_plot["err"], direct_label, args)
        plt.axvspan(args.fit_start_ns, min(fit_end_ns, xerr[-1]), alpha=0.12)
        plt.xlabel("Time / ns")
        plt.ylabel(r"MSD$_{\sigma}$ / $\AA^2$")
        plt.axhline(0, lw=0.8)
        plt.legend(frameon=False)
        plt.tight_layout()
        p = Path(outdir) / f"{stem}_onsager_direct_MSDsigma_{species_tag}.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")

        plt.figure(figsize=(args.fig_width, args.fig_height))
        _plot_mean_error(plt, xerr, direct_plot["mean"], direct_plot["err"], direct_label, args,
                         loglog=True)
        plt.xlabel("Time / ns")
        plt.ylabel(r"MSD$_{\sigma}$ / $\AA^2$")
        plt.legend(frameon=False)
        plt.tight_layout()
        p = Path(outdir) / f"{stem}_onsager_direct_MSDsigma_{species_tag}_loglog.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")


def make_common_axis_msd_figures(outdir, rows, args):
    """Add one self and one distinct figure per temperature with common y limits."""
    plot_rows = [r for r in rows if r.get("_plot_data") is not None]
    if len(plot_rows) < 2:
        return
    common_dir = Path(outdir) / "common_axis_multiT"
    common_dir.mkdir(parents=True, exist_ok=True)

    def extrema(section):
        lows, highs = [], []
        for r in plot_rows:
            pd = r["_plot_data"]
            for stat in pd[section].values():
                m, e = np.asarray(stat["mean"]), np.nan_to_num(np.asarray(stat["err"]), nan=0.0)
                lows.append(np.nanmin(m - e)); highs.append(np.nanmax(m + e))
        lo, hi = float(np.nanmin(lows)), float(np.nanmax(highs))
        pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
        return lo - pad, hi + pad

    self_ylim = extrema("self")
    distinct_ylim = extrema("distinct")
    plt = setup_plot(args)

    for r in sorted(plot_rows, key=lambda x: x["temperature_K"]):
        pd = r["_plot_data"]
        T = r["temperature_K"]
        species_tag = _species_tag(pd["species_labels"])
        distinct_tag = _species_tag(list(pd["distinct"].keys()))
        x = pd["time_ns"]

        plt.figure(figsize=(args.fig_width, args.fig_height))
        for lab, stat in pd["self"].items():
            _plot_mean_error(plt, x, stat["mean"], stat["err"], f"{lab}, {T:g} K", args)
        plt.xlabel("Time / ns"); plt.ylabel(r"MSD$_{tr}$ / $\AA^2$")
        plt.ylim(*self_ylim); plt.axhline(0, lw=0.8); plt.legend(frameon=False)
        plt.tight_layout()
        p = common_dir / f"{r['system']}_self_MSDtr_{species_tag}_commonY.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")

        plt.figure(figsize=(args.fig_width, args.fig_height))
        for lab, stat in pd["distinct"].items():
            _plot_mean_error(plt, x, stat["mean"], stat["err"], f"{lab} distinct, {T:g} K", args)
        plt.xlabel("Time / ns"); plt.ylabel(r"MSD$_{\sigma}$ / $\AA^2$")
        plt.ylim(*distinct_ylim); plt.axhline(0, lw=0.8); plt.legend(frameon=False)
        plt.tight_layout()
        p = common_dir / f"{r['system']}_distinct_MSDsigma_{distinct_tag}_commonY.png"
        plt.savefig(p, dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")


def analyze_one(path, temperature_k, args, outdir):
    path = Path(path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = path.name[:-7] if path.name.endswith(".pdb.gz") else path.stem

    atoms, has_model, first_box = read_topology(path, args.atom_type_field)
    if args.list_topology:
        print_topology(atoms)
        if args.only_list_topology:
            return None

    species = build_species(atoms, args)
    selected_analysis = sorted({o for s in species for g in s["groups"] for o in g})
    if args.remove_drift == "all":
        selected_read = list(range(len(atoms)))
        drift_ord = list(range(len(atoms)))
    elif args.remove_drift == "selected":
        selected_read = selected_analysis
        drift_ord = selected_analysis
    else:
        selected_read = selected_analysis
        drift_ord = []

    ord_to_local = {o: i for i, o in enumerate(selected_read)}
    frames, boxes = [], []
    for iframe, (coord, box) in enumerate(iter_frames(path, len(atoms), selected_read, has_model, first_box)):
        if iframe < args.discard_frames:
            continue
        if (iframe - args.discard_frames) % args.stride:
            continue
        frames.append(coord)
        boxes.append(np.array([np.nan, np.nan, np.nan]) if box is None else box)
    if len(frames) < 10:
        raise RuntimeError("Too few frames after discard/stride")

    pos = np.asarray(frames, np.float32)
    boxes = np.asarray(boxes, float)
    if args.no_unwrap:
        unwrapped = pos.astype(float)
    else:
        unwrapped = unwrap_orthorhombic(pos, boxes)

    valid_boxes = boxes[np.all(np.isfinite(boxes) & (boxes > 0), axis=1)]
    if not len(valid_boxes):
        raise RuntimeError("A valid CRYST1 box is required for conductivity")
    volume_A3 = float(np.mean(np.prod(valid_boxes, axis=1)))

    drift = None
    if drift_ord:
        idx = np.array([ord_to_local[o] for o in drift_ord], int)
        if args.drift_geometric:
            weights = np.ones(len(idx))
        else:
            weights = np.array([MASS.get(atoms[o]["element"], 12.011) for o in drift_ord], float)
        center = (unwrapped[:, idx, :] * weights[None, :, None]).sum(axis=1) / weights.sum()
        drift = center - center[0]

    particle_pos = {}
    for s in species:
        plist = []
        for group in s["groups"]:
            idx = np.array([ord_to_local[o] for o in group], int)
            if s["kind"] == "atoms":
                p = unwrapped[:, idx[0], :]
            else:
                masses = np.array([MASS.get(atoms[o]["element"], 12.011) for o in group], float)
                p = (unwrapped[:, idx, :] * masses[None, :, None]).sum(axis=1) / masses.sum()
            if drift is not None:
                p = p - drift
            plist.append(p)
        particle_pos[s["label"]] = np.stack(plist, axis=1)
        print(f"[species] {s['label']}: N={len(plist)}, z={s['charge']:+g}, kind={s['kind']}")

    dt_eff = args.dt_ns * args.stride
    max_lag = min(int(round(args.max_lag_ns / dt_eff)), len(frames) - 1)
    time_ns = np.arange(max_lag + 1) * dt_eff
    fit_end_ns = min(args.fit_end_ns, time_ns[-1])
    if fit_end_ns < args.fit_start_ns:
        raise RuntimeError(f"fit-end-ns ({args.fit_end_ns}) exceeds available lag range ({time_ns[-1]:.6g} ns) and becomes < fit-start-ns")
    ts_start = args.fit_start_ns if args.timeseries_start_ns is None else args.timeseries_start_ns

    curves = {}
    meta = {}
    vis_self_curves = {}
    vis_distinct_curves = {}

    # Same-species self and distinct terms.
    for s in species:
        lab, z = s["label"], s["charge"]
        p = particle_pos[lab]
        n = p.shape[1]
        tracer_msd = msd_fft_particles(p, max_lag)
        self_curve = tracer_msd * n
        collective_vec = p.sum(axis=1)
        collective_curve = msd_fft_vector(collective_vec, max_lag)
        distinct_curve = collective_curve - self_curve
        curves[f"self:{lab}"] = self_curve
        curves[f"distinct:{lab}-{lab}"] = distinct_curve
        vis_self_curves[f"{lab}"] = tracer_msd
        vis_distinct_curves[f"{lab}-{lab}"] = (z * z) * distinct_curve
        meta[f"self:{lab}"] = {"za": z, "zb": z, "mult": 1.0, "kind": "self"}
        meta[f"distinct:{lab}-{lab}"] = {"za": z, "zb": z, "mult": 1.0, "kind": "same distinct"}

    # Unlike-species cross terms, one unordered pair; total multiplicity = 2.
    for ia in range(len(species)):
        for ib in range(ia + 1, len(species)):
            a, b = species[ia], species[ib]
            va = particle_pos[a["label"]].sum(axis=1)
            vb = particle_pos[b["label"]].sum(axis=1)
            key = f"distinct:{a['label']}-{b['label']}"
            curves[key] = cross_displacement_fft(va, vb, max_lag)
            meta[key] = {"za": a["charge"], "zb": b["charge"], "mult": 2.0, "kind": "cross distinct"}
            vis_distinct_curves[f"{a['label']}-{b['label']}"] = 2.0 * a["charge"] * b["charge"] * curves[key]

    # Total charge collective correlation as a consistency check.
    qvec = np.zeros((len(frames), 3), float)
    for s in species:
        qvec += s["charge"] * particle_pos[s["label"]].sum(axis=1)
    curves["total:charge"] = msd_fft_vector(qvec, max_lag)
    meta["total:charge"] = {"za": 1.0, "zb": 1.0, "mult": 1.0, "kind": "total"}
    vis_direct_curve = curves["total:charge"]

    summary_rows = []
    timeseries = {"time_ns": time_ns}
    for key, y in curves.items():
        m = meta[key]
        slope, intercept, r2 = fit_slope(time_ns, y, args.fit_start_ns, fit_end_ns)
        sigma_pair = slope_to_sigma(slope, m["za"], m["zb"], volume_A3, temperature_k, args.dimension)
        sigma_total_contrib = sigma_pair * m["mult"]
        running_slope = expanding_slope(time_ns, y, ts_start, args.timeseries_min_points)
        running_sigma = np.array([
            slope_to_sigma(v, m["za"], m["zb"], volume_A3, temperature_k, args.dimension) * m["mult"]
            if np.isfinite(v) else np.nan for v in running_slope
        ])
        timeseries[f"C_A2__{key}"] = y
        timeseries[f"sigma_mS_cm__{key}"] = running_sigma * 10.0
        summary_rows.append({
            "term": key, "kind": m["kind"], "z_a": m["za"], "z_b": m["zb"],
            "multiplicity_in_total": m["mult"], "fit_start_ns": args.fit_start_ns,
            "fit_end_ns": fit_end_ns, "slope_A2_per_ns": slope, "fit_R2": r2,
            "sigma_pair_S_per_m": sigma_pair,
            "sigma_contribution_S_per_m": sigma_total_contrib,
            "sigma_contribution_mS_per_cm": sigma_total_contrib * 10.0,
        })

    # Additional visualization-oriented MSD series.
    for key, y in vis_self_curves.items():
        timeseries[f"MSDtr_A2__{key}"] = y
    for key, y in vis_distinct_curves.items():
        timeseries[f"MSDsigma_A2__{key}"] = y
    timeseries["MSDsigma_A2__direct_total"] = vis_direct_curve

    # Decomposed sum excluding direct total curve.
    decomposed = sum(r["sigma_contribution_S_per_m"] for r in summary_rows if r["term"] != "total:charge")
    direct_total = next(r["sigma_contribution_S_per_m"] for r in summary_rows if r["term"] == "total:charge")
    print(f"[total] decomposed = {decomposed*10:.8g} mS/cm")
    print(f"[total] direct     = {direct_total*10:.8g} mS/cm")
    print(f"[check] difference = {(decomposed-direct_total)*10:.8g} mS/cm")

    # Save CSVs.
    ts_path = outdir / f"{stem}_onsager_correlation_timeseries.csv"
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(timeseries.keys())
        writer.writerow(keys)
        for i in range(len(time_ns)):
            writer.writerow([timeseries[k][i] for k in keys])

    sum_path = outdir / f"{stem}_onsager_conductivity_decomposition.csv"
    with open(sum_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)

    info_path = outdir / f"{stem}_onsager_run_info.txt"
    with open(info_path, "w") as f:
        f.write(f"pdb = {path}\nframes = {len(frames)}\ndt_eff_ns = {dt_eff}\n")
        f.write(f"temperature_K = {temperature_k}\navg_volume_A3 = {volume_A3}\n")
        f.write(f"fit_start_ns = {args.fit_start_ns}\nfit_end_ns = {fit_end_ns}\n")
        f.write(f"remove_drift = {args.remove_drift}\nunwrap = {not args.no_unwrap}\ndimension = {args.dimension}\n")
        f.write(f"sigma_decomposed_mS_cm = {decomposed*10}\n")
        f.write(f"sigma_direct_total_mS_cm = {direct_total*10}\n")

    block_stats = compute_block_msd_statistics(particle_pos, species, dt_eff, max_lag, args)
    if block_stats is not None:
        print(f"[blocks] n={block_stats['n_blocks']}, frames/block={block_stats['block_frames']}, "
              f"duration/block={block_stats['block_duration_ns']:.6g} ns, error={block_stats['error_stat']}")
        block_csv = outdir / f"{stem}_onsager_block_MSD_statistics.csv"
        cols = {"time_ns": block_stats["time_ns"]}
        for lab, stat in block_stats["self"].items():
            cols[f"MSDtr_mean_A2__{lab}"] = stat["mean"]
            cols[f"MSDtr_error_A2__{lab}"] = stat["err"]
        for lab, stat in block_stats["distinct"].items():
            cols[f"MSDsigma_mean_A2__{lab}"] = stat["mean"]
            cols[f"MSDsigma_error_A2__{lab}"] = stat["err"]
        cols["MSDsigma_mean_A2__direct_total"] = block_stats["direct"]["mean"]
        cols["MSDsigma_error_A2__direct_total"] = block_stats["direct"]["err"]
        with open(block_csv, "w", newline="") as bf:
            bw = csv.writer(bf); names = list(cols.keys()); bw.writerow(names)
            for ii in range(len(block_stats["time_ns"])):
                bw.writerow([cols[k][ii] for k in names])
        print(f"[OK] {block_csv}")

    species_labels = [sp["label"] for sp in species]
    write_grouped_msd_figures(outdir, stem, time_ns, vis_self_curves, vis_distinct_curves,
                              vis_direct_curve, species_labels, args, fit_end_ns, block_stats)

    plt = setup_plot(args)

    # Correlation curves.
    plt.figure(figsize=(args.fig_width, args.fig_height))
    for key, y in curves.items():
        if key == "total:charge":
            continue
        plt.plot(time_ns, y, lw=args.line_width, label=term_label_for_plot(key))
    plt.axvspan(args.fit_start_ns, fit_end_ns, alpha=0.12)
    plt.xlabel("Time / ns")
    plt.ylabel(r"Displacement correlation / $\AA^2$")
    plt.axhline(0, lw=0.8)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    p = outdir / f"{stem}_onsager_correlation_terms.png"
    plt.savefig(p, dpi=args.dpi)
    if args.save_pdf: plt.savefig(p.with_suffix(".pdf"))
    plt.close()

    # Running conductivity contributions.
    plt.figure(figsize=(args.fig_width, args.fig_height))
    for key in curves:
        if key == "total:charge":
            continue
        plt.plot(time_ns, timeseries[f"sigma_mS_cm__{key}"], lw=args.line_width,
                 label=term_label_for_plot(key))
    plt.plot(time_ns, timeseries["sigma_mS_cm__total:charge"], "--", lw=args.line_width + 0.3, label="total charge")
    plt.xlabel("Upper fitting time / ns")
    plt.ylabel(r"Cumulative $\sigma$ contribution / mS cm$^{-1}$")
    plt.axhline(0, lw=0.8)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    p = outdir / f"{stem}_onsager_conductivity_timeseries.png"
    plt.savefig(p, dpi=args.dpi)
    if args.save_pdf: plt.savefig(p.with_suffix(".pdf"))
    plt.close()

    # Final decomposition bar chart.
    plot_rows = [r for r in summary_rows if r["term"] != "total:charge"]
    labels = [term_label_for_plot(r["term"]).replace(" ", "\n", 1) for r in plot_rows]
    vals = [r["sigma_contribution_mS_per_cm"] for r in plot_rows]
    plt.figure(figsize=(max(args.fig_width, 1.1 * len(labels)), args.fig_height))
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.axhline(0, lw=0.8)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(r"$\sigma$ contribution / mS cm$^{-1}$")
    plt.tight_layout()
    p = outdir / f"{stem}_onsager_conductivity_decomposition.png"
    plt.savefig(p, dpi=args.dpi)
    if args.save_pdf: plt.savefig(p.with_suffix(".pdf"))
    plt.close()

    print(f"[OK] {ts_path}")
    print(f"[OK] {sum_path}")
    print(f"[OK] {info_path}")
    plot_data = None
    if block_stats is not None:
        plot_data = {
            "time_ns": block_stats["time_ns"],
            "self": block_stats["self"],
            "distinct": block_stats["distinct"],
            "direct": block_stats["direct"],
            "species_labels": species_labels,
        }
    return {"pdb": str(path), "system": stem, "temperature_K": temperature_k,
            "avg_volume_A3": volume_A3, "sigma_decomposed_mS_cm": decomposed*10.0,
            "sigma_direct_total_mS_cm": direct_total*10.0, "summary_csv": str(sum_path),
            "timeseries_csv": str(ts_path), "_plot_data": plot_data}


def safe_stem(path):
    p = Path(path)
    return p.name[:-7] if p.name.endswith(".pdb.gz") else p.stem


def infer_temperature_from_name(path):
    text = str(path)
    patterns = [
        r"(?:^|[^A-Za-z0-9])T\s*([0-9]+(?:\.[0-9]+)?)\s*K(?:[^A-Za-z0-9]|$)",
        r"(?:nvt|npt|nve)\s*([0-9]+(?:\.[0-9]+)?)\s*K",
        r"([0-9]+(?:\.[0-9]+)?)\s*_?K(?:[^A-Za-z0-9]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_temperature_map(entries):
    mapping = {}
    for item in entries or []:
        if ":" in item:
            key, value = item.rsplit(":", 1)
        elif "=" in item:
            key, value = item.rsplit("=", 1)
        else:
            raise ValueError(f"Invalid --temperature-map entry: {item}; use file.pdb:300 or stem=300")
        mapping[key.strip()] = float(value)
    return mapping


def resolve_temperature(path, args, mapping):
    p = Path(path)
    for key in (str(p), p.name, safe_stem(p)):
        if key in mapping:
            return mapping[key], f"map:{key}"
    if not args.no_temperature_from_name:
        t = infer_temperature_from_name(p)
        if t is not None:
            return t, "filename"
    if args.temperature_k is not None:
        return args.temperature_k, "fallback"
    raise ValueError(f"Temperature not found for {path}. Add 300K to the filename or use --temperature-map {p.name}:300")


def expand_inputs(items):
    out=[]
    for item in items:
        matches=sorted(glob.glob(item)) if any(c in item for c in "*?[]") else []
        out.extend(matches or [item])
    seen=[]
    for x in out:
        if x not in seen: seen.append(x)
    return [Path(x) for x in seen]


def write_combined_summary(outdir, rows):
    path=Path(outdir)/"summary_all_onsager_conductivity.csv"
    if not rows: return
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    keys=list(clean_rows[0].keys())
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(clean_rows)
    print(f"[OK] {path}")


def make_temperature_summary_plots(outdir, rows, args):
    if len(rows) < 2: return
    clean=sorted([r for r in rows if np.isfinite(r["temperature_K"])], key=lambda r:r["temperature_K"])
    if len(clean)<2: return
    T=np.array([r["temperature_K"] for r in clean],float)
    sig=np.array([r["sigma_decomposed_mS_cm"] for r in clean],float)
    plt=setup_plot(args)
    plt.figure(figsize=(args.fig_width,args.fig_height))
    plt.plot(T,sig,"o-",lw=args.line_width)
    plt.xlabel("Temperature / K"); plt.ylabel(r"$\sigma_{Onsager}$ / mS cm$^{-1}$")
    plt.tight_layout(); p=Path(outdir)/"temperature_total_onsager_conductivity.png"; plt.savefig(p,dpi=args.dpi)
    if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
    plt.close(); print(f"[OK] {p}")
    mask=(T>0)&(sig>0)
    if mask.sum()>=2:
        x=1000.0/T[mask]; y=np.log10(sig[mask]*T[mask])
        slope,intercept=np.polyfit(x,y,1)
        Ea=-1000.0*np.log(10.0)*8.617333262145e-5*slope
        xx=np.linspace(x.min(),x.max(),100)
        plt.figure(figsize=(args.fig_width,args.fig_height))
        plt.plot(x,y,"o",label="data"); plt.plot(xx,intercept+slope*xx,"--",label=f"fit: Ea={Ea:.3f} eV")
        plt.xlabel(r"1000 / T / K$^{-1}$"); plt.ylabel(r"log$_{10}$($\sigma T$ / mS cm$^{-1}$ K)")
        plt.legend(frameon=False); plt.tight_layout(); p=Path(outdir)/"arrhenius_total_onsager_sigmaT.png"; plt.savefig(p,dpi=args.dpi)
        if args.save_pdf: plt.savefig(p.with_suffix('.pdf'))
        plt.close(); print(f"[OK] {p}")


def main():
    ap = argparse.ArgumentParser(description="Multi-temperature Onsager self/distinct conductivity decomposition from PDB trajectories")
    ap.add_argument("--pdb", nargs="+", required=True, help="One or more PDB trajectories; globs are accepted")
    ap.add_argument("--dt-ns", type=float, required=True, help="Saved-frame interval in ns")
    ap.add_argument("--temperature-k", type=float, default=None, help="Fallback temperature only when it cannot be inferred per PDB")
    ap.add_argument("--temperature-map", nargs="*", default=[], help="Per-PDB mapping, e.g. T300.pdb:300 system_stem=350")
    ap.add_argument("--no-temperature-from-name", action="store_true", help="Disable automatic temperature inference from path/name")
    ap.add_argument("--outdir", default="onsager_distinct_results")
    ap.add_argument("--discard-frames", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-lag-ns", type=float, required=True); ap.add_argument("--fit-start-ns", type=float, required=True); ap.add_argument("--fit-end-ns", type=float, required=True)
    ap.add_argument("--timeseries-start-ns", type=float, default=None); ap.add_argument("--timeseries-min-points", type=int, default=8)
    ap.add_argument("--track-element", action="append", default=[], metavar="LABEL:ELEMENT:CHARGE")
    ap.add_argument("--track-atom-type", action="append", default=[], metavar="LABEL:TYPES:CHARGE")
    ap.add_argument("--track-residue-com", action="append", default=[], metavar="LABEL:RESNAMES:CHARGE")
    ap.add_argument("--atom-type-field", default="auto", choices=["auto","occupancy","bfactor","resname","tail","last_int","none"], help="PDB field containing the Tinker atom type. Use resname when xyzpdb writes types such as 72 into columns 18-21.")
    ap.add_argument("--remove-drift", default="all", choices=["none","all","selected"]); ap.add_argument("--drift-geometric", action="store_true")
    ap.add_argument("--dimension", type=int, default=3, help="Diffusion dimensionality d used in sigma = ... / (2 d V k_B T). Default: 3")
    ap.add_argument("--msd-blocks", type=int, default=5, help="Number of contiguous non-overlapping blocks for MSD uncertainty. Default: 5")
    ap.add_argument("--msd-error-stat", choices=["sem", "sd", "ci95"], default="sem", help="Uncertainty shown around block-averaged MSD: sem, sd, or ci95. Default: sem")
    ap.add_argument("--block-max-lag-ns", type=float, default=None, help="Optional maximum lag used only for block-error MSD curves")
    ap.add_argument("--error-alpha", type=float, default=0.20, help="Opacity of the MSD uncertainty band")
    ap.add_argument("--no-unwrap", action="store_true"); ap.add_argument("--list-topology", action="store_true"); ap.add_argument("--only-list-topology", action="store_true")
    ap.add_argument("--font-size", type=float, default=14); ap.add_argument("--axis-label-size", type=float, default=18); ap.add_argument("--tick-label-size", type=float, default=15); ap.add_argument("--legend-font-size", type=float, default=12)
    ap.add_argument("--line-width", type=float, default=2.0); ap.add_argument("--fig-width", type=float, default=7.2); ap.add_argument("--fig-height", type=float, default=5.2); ap.add_argument("--dpi", type=int, default=300); ap.add_argument("--save-pdf", action="store_true")
    ap.add_argument("--show-term-titles", action="store_true", help="Add titles to the per-term MSD figures")
    ap.add_argument("--save-total-term-figure", action="store_true", help="Also save a separate MSD figure for the direct total-charge term")
    args=ap.parse_args()
    paths=expand_inputs(args.pdb)
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError("Missing PDB file(s): "+", ".join(missing))
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    mapping=parse_temperature_map(args.temperature_map)
    rows=[]
    for p in paths:
        temp,source=resolve_temperature(p,args,mapping)
        print("\n"+"="*80); print(f"[ANALYZE] {p} | T={temp:g} K ({source})")
        row=analyze_one(p,temp,args,outdir)
        if row is None:
            continue
        row["temperature_source"]=source
        rows.append(row)
    write_combined_summary(outdir,rows)
    make_temperature_summary_plots(outdir,rows,args)
    make_common_axis_msd_figures(outdir,rows,args)


if __name__ == "__main__":
    main()

