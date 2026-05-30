#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# Example: python3 fast_msd_multi_pdb_v1_1_0.py --pdb *.pdb --dt-ps 50.0 --outdir msd_results --fit-start-ps 1000 --max-lag-ps 20000
#
#
#
##

import argparse
import gzip
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "S": 32.06, "Li": 6.94, "Na": 22.990,
    "K": 39.098, "Cl": 35.45,
}


def open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def safe_stem(path):
    p = Path(path)
    name = p.name
    if name.endswith(".pdb.gz"):
        return name[:-7]
    return p.stem


def infer_element(name, element_field=""):
    e = element_field.strip()
    if e:
        e = e[0].upper() + e[1:].lower()
        if e in MASS:
            return e

    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        return "C"

    if len(letters) >= 2:
        e2 = letters[:2].title()
        if e2 in MASS:
            return e2

    e1 = letters[0].upper()
    return e1 if e1 in MASS else "C"


def parse_cryst1(line):
    try:
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
        alpha = float(line[33:40])
        beta = float(line[40:47])
        gamma = float(line[47:54])
        return np.array([a, b, c], dtype=np.float64), np.array([alpha, beta, gamma], dtype=np.float64)
    except Exception:
        return None, None


def read_topology_first_model(pdb_path):
    atoms = []
    has_model = False
    first_box = None
    first_angles = None

    with open_text(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                has_model = True
                continue

            if line.startswith("CRYST1"):
                box, angles = parse_cryst1(line)
                if box is not None and first_box is None:
                    first_box = box
                    first_angles = angles
                continue

            if line.startswith(("ATOM  ", "HETATM")):
                serial_text = line[6:11].strip()
                try:
                    serial = int(serial_text)
                except Exception:
                    serial = len(atoms) + 1

                name = line[12:16].strip()
                resname = line[17:21].strip()
                chain = line[21:22].strip()
                resid = line[22:27].strip()
                element_field = line[76:78].strip() if len(line) >= 78 else ""
                element = infer_element(name, element_field)

                atoms.append({
                    "ordinal": len(atoms),
                    "serial": serial,
                    "name": name,
                    "resname": resname,
                    "chain": chain,
                    "resid": resid,
                    "element": element,
                    "mass": MASS.get(element, 12.011),
                })

            if line.startswith("ENDMDL") and atoms:
                break

    if not atoms:
        raise RuntimeError(f"No ATOM/HETATM records found in {pdb_path}")

    return atoms, has_model, first_box, first_angles


def residue_key(atom):
    return (atom["chain"], atom["resid"], atom["resname"])


def print_topology_summary(pdb_path, atoms, first_box):
    print("\n" + "=" * 80)
    print(f"[TOPOLOGY] {pdb_path}")
    print(f"atoms = {len(atoms)}")
    if first_box is not None:
        print(f"box   = {first_box[0]:.6f} {first_box[1]:.6f} {first_box[2]:.6f} A")

    print("\n[atom names: top 50]")
    for k, v in Counter(a["name"] for a in atoms).most_common(50):
        print(f"  {k:>8s} : {v}")

    print("\n[elements]")
    for k, v in Counter(a["element"] for a in atoms).most_common():
        print(f"  {k:>8s} : {v}")

    groups = defaultdict(list)
    for a in atoms:
        groups[residue_key(a)].append(a)

    by_resname = defaultdict(list)
    for key, group in groups.items():
        by_resname[key[2]].append(group)

    print("\n[residue summary]")
    for resname, glist in sorted(by_resname.items(), key=lambda x: (-len(x[1]), x[0])):
        sizes = Counter(len(g) for g in glist)
        elem_sets = Counter("".join(sorted(set(a["element"] for a in g))) for g in glist)
        size_text = ", ".join(f"{n} atoms x {c}" for n, c in sorted(sizes.items()))
        elem_text = ", ".join(f"{e} x {c}" for e, c in elem_sets.most_common(5))
        shown = resname if resname else "<blank>"
        print(f"  {shown:>8s} : residues={len(glist):5d} | {size_text} | elements: {elem_text}")
    print("=" * 80 + "\n")


def iter_selected_frames(pdb_path, natoms, selected_ordinals, has_model):
    selected_ordinals = sorted(selected_ordinals)
    ordinal_to_local = {o: i for i, o in enumerate(selected_ordinals)}
    nsel = len(selected_ordinals)

    coords = np.empty((nsel, 3), dtype=np.float32)
    atom_i = 0
    found = 0
    current_box = None
    current_angles = None

    with open_text(pdb_path) as f:
        for line in f:
            if line.startswith("CRYST1"):
                box, angles = parse_cryst1(line)
                if box is not None:
                    current_box = box
                    current_angles = angles
                continue

            if line.startswith("MODEL"):
                atom_i = 0
                found = 0
                coords = np.empty((nsel, 3), dtype=np.float32)
                continue

            if line.startswith(("ATOM  ", "HETATM")):
                if atom_i in ordinal_to_local:
                    j = ordinal_to_local[atom_i]
                    coords[j, 0] = float(line[30:38])
                    coords[j, 1] = float(line[38:46])
                    coords[j, 2] = float(line[46:54])
                    found += 1

                atom_i += 1

                if (not has_model) and atom_i == natoms:
                    if found == nsel:
                        yield coords.copy(), current_box, current_angles
                    atom_i = 0
                    found = 0
                    coords = np.empty((nsel, 3), dtype=np.float32)

            if line.startswith("ENDMDL"):
                if found == nsel:
                    yield coords.copy(), current_box, current_angles
                atom_i = 0
                found = 0
                coords = np.empty((nsel, 3), dtype=np.float32)


def unwrap_orthorhombic(pos, boxes):
    """
    pos:   (T, N, 3), wrapped coordinates in Angstrom
    boxes: (T, 3), box lengths in Angstrom

    Orthorhombic minimum-image unwrapping.
    Assumes displacement per saved frame is less than L/2.
    """
    if boxes is None:
        return pos.astype(np.float64, copy=False)

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 3:
        return pos.astype(np.float64, copy=False)
    if np.any(~np.isfinite(boxes)) or np.any(boxes <= 0):
        return pos.astype(np.float64, copy=False)

    out = pos.astype(np.float64, copy=True)
    for t in range(1, out.shape[0]):
        box = boxes[t]
        d = pos[t].astype(np.float64) - pos[t - 1].astype(np.float64)
        d -= box * np.round(d / box)
        out[t] = out[t - 1] + d
    return out


def build_li_selection(atoms, li_names, li_resnames):
    li_names_upper = {x.upper() for x in li_names}
    li_resnames_set = set(li_resnames)
    li_ord = []
    for a in atoms:
        if (
            a["element"] == "Li"
            or a["name"].upper() in li_names_upper
            or a["resname"] in li_resnames_set
        ):
            li_ord.append(a["ordinal"])
    return li_ord


def build_fsa_groups(atoms, fsa_resnames, auto_fsa_by_residue=True):
    """
    Returns list of FSA groups as lists of atom ordinals.

    Priority:
      1. residues whose resname is in --fsa-resnames
      2. if none found and auto is enabled, residues containing S, F, and O.
         This catches FSA/FSI even if the resname is MOL or blank, as long as
         each molecule is separated by residue ID.
    """
    groups = defaultdict(list)
    for a in atoms:
        groups[residue_key(a)].append(a)

    out = []
    fsa_resnames_set = set(fsa_resnames)

    if fsa_resnames_set:
        for key, group in groups.items():
            if key[2] in fsa_resnames_set:
                out.append([a["ordinal"] for a in group])

    if not out and auto_fsa_by_residue:
        for key, group in groups.items():
            elems = {a["element"] for a in group}
            if {"S", "F", "O"}.issubset(elems):
                out.append([a["ordinal"] for a in group])

    return out


def compute_com_from_unwrapped(unwrapped_selected, groups_local, atoms, selected_ord_to_global_ord):
    T = unwrapped_selected.shape[0]
    G = len(groups_local)
    com = np.zeros((T, G, 3), dtype=np.float64)

    for gi, locs in enumerate(groups_local):
        locs = np.array(locs, dtype=int)
        masses = np.array([atoms[selected_ord_to_global_ord[j]]["mass"] for j in locs], dtype=np.float64)
        msum = masses.sum()
        com[:, gi, :] = (unwrapped_selected[:, locs, :] * masses[None, :, None]).sum(axis=1) / msum

    return com


def msd_fft_multi(pos, max_lag=None, batch_cols=256):
    """
    Exact window-averaged MSD using FFT.

    pos: (T, N, 3), unwrapped coordinates in Angstrom
    returns msd[lag] in Angstrom^2 averaged over particles and time origins.
    """
    pos = np.asarray(pos, dtype=np.float64)
    T, N, dim = pos.shape

    if max_lag is None:
        max_lag = T - 1
    max_lag = min(max_lag, T - 1)
    if max_lag < 1:
        raise RuntimeError("max_lag is too small. Need at least 2 frames.")

    x = pos.reshape(T, N * dim)
    q = np.sum(pos * pos, axis=(1, 2))
    prefix = np.concatenate([[0.0], np.cumsum(q)])

    nfft = 1 << ((2 * T - 1).bit_length())
    ac = np.zeros(max_lag + 1, dtype=np.float64)

    for start in range(0, x.shape[1], batch_cols):
        xb = x[:, start:start + batch_cols]
        f = np.fft.rfft(xb, n=nfft, axis=0)
        c = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=0)
        ac += c[:max_lag + 1].sum(axis=1)

    lag = np.arange(max_lag + 1)
    count = T - lag

    sum_q_0 = prefix[T - lag] - prefix[0]
    sum_q_lag = prefix[T] - prefix[lag]

    msd = (sum_q_0 + sum_q_lag - 2.0 * ac) / (count * N)
    msd[0] = 0.0
    msd[msd < 0.0] = 0.0
    return msd


def fit_diffusion(time_ps, msd_A2, fit_start_ps=None, fit_end_ps=None):
    if fit_start_ps is None:
        fit_start_ps = time_ps[int(0.2 * len(time_ps))]
    if fit_end_ps is None:
        fit_end_ps = time_ps[int(0.8 * len(time_ps))]

    mask = (time_ps >= fit_start_ps) & (time_ps <= fit_end_ps) & (time_ps > 0)
    if mask.sum() < 3:
        raise RuntimeError(
            f"Too few points in fitting window: {fit_start_ps} - {fit_end_ps} ps. "
            f"Try smaller --fit-start-ps or larger --fit-end-ps."
        )

    slope, intercept = np.polyfit(time_ps[mask], msd_A2[mask], 1)
    pred = slope * time_ps[mask] + intercept
    ss_res = np.sum((msd_A2[mask] - pred) ** 2)
    ss_tot = np.sum((msd_A2[mask] - np.mean(msd_A2[mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    D_A2_ps = slope / 6.0
    D_cm2_s = D_A2_ps * 1.0e-4
    D_1e5_cm2_s = D_A2_ps * 10.0

    return {
        "fit_start_ps": float(fit_start_ps),
        "fit_end_ps": float(fit_end_ps),
        "slope_A2_ps": float(slope),
        "intercept_A2": float(intercept),
        "D_A2_ps": float(D_A2_ps),
        "D_cm2_s": float(D_cm2_s),
        "D_1e5_cm2_s": float(D_1e5_cm2_s),
        "r2": float(r2),
    }


def save_species_result(outdir, pdb_stem, species, time_ps, msd_A2, fit, make_png=True):
    outdir = Path(outdir)
    csv_path = outdir / f"{pdb_stem}_{species}_msd.csv"
    summary_path = outdir / f"{pdb_stem}_{species}_summary.txt"
    png_path = outdir / f"{pdb_stem}_{species}_msd.png"

    D_from_origin = np.full_like(time_ps, np.nan, dtype=np.float64)
    valid = time_ps > 0
    D_from_origin[valid] = msd_A2[valid] / (6.0 * time_ps[valid])

    arr = np.column_stack([
        time_ps,
        msd_A2,
        D_from_origin,
        D_from_origin * 10.0,
    ])
    np.savetxt(
        csv_path,
        arr,
        delimiter=",",
        header="time_ps,MSD_A2,D_from_origin_A2_per_ps,D_from_origin_1e-5_cm2_per_s",
        comments="",
    )

    with open(summary_path, "w") as f:
        f.write(f"pdb = {pdb_stem}\n")
        f.write(f"species = {species}\n")
        f.write(f"fit_start_ps = {fit['fit_start_ps']:.10g}\n")
        f.write(f"fit_end_ps = {fit['fit_end_ps']:.10g}\n")
        f.write(f"slope_A2_per_ps = {fit['slope_A2_ps']:.12g}\n")
        f.write(f"D_A2_per_ps = {fit['D_A2_ps']:.12g}\n")
        f.write(f"D_cm2_per_s = {fit['D_cm2_s']:.12g}\n")
        f.write(f"D_1e-5_cm2_per_s = {fit['D_1e5_cm2_s']:.12g}\n")
        f.write(f"R2 = {fit['r2']:.8g}\n")

    if make_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.0, 4.2))
        plt.plot(time_ps, msd_A2, lw=1.8, label="MSD")

        mask = (time_ps >= fit["fit_start_ps"]) & (time_ps <= fit["fit_end_ps"])
        yfit = fit["slope_A2_ps"] * time_ps[mask] + fit["intercept_A2"]
        plt.plot(time_ps[mask], yfit, "--", lw=1.5, label="linear fit")

        plt.xlabel("Time lag / ps")
        plt.ylabel(r"MSD / $\AA^2$")
        plt.title(f"{pdb_stem} | {species}\nD = {fit['D_1e5_cm2_s']:.4g} × 10$^{{-5}}$ cm$^2$/s, R$^2$ = {fit['r2']:.4f}")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()

    return csv_path, summary_path, png_path if make_png else None


def analyze_one_pdb(pdb_path, args):
    pdb_path = Path(pdb_path)
    pdb_stem = safe_stem(pdb_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atoms, has_model, first_box, first_angles = read_topology_first_model(pdb_path)

    if args.list_topology:
        print_topology_summary(pdb_path, atoms, first_box)
        if args.only_list_topology:
            return []

    li_ord = build_li_selection(atoms, args.li_names, args.li_resnames)
    fsa_groups_ord = build_fsa_groups(
        atoms,
        args.fsa_resnames,
        auto_fsa_by_residue=(not args.no_auto_fsa),
    )

    selected = sorted(set(li_ord) | set(o for g in fsa_groups_ord for o in g))

    print("\n" + "-" * 80)
    print(f"[ANALYZE] {pdb_path}")
    print(f"  atoms          = {len(atoms)}")
    print(f"  has MODEL      = {has_model}")
    if first_box is not None:
        print(f"  first box      = {first_box[0]:.6f} {first_box[1]:.6f} {first_box[2]:.6f} A")
    print(f"  Li atoms       = {len(li_ord)}")
    print(f"  FSA molecules  = {len(fsa_groups_ord)}")

    if len(fsa_groups_ord) == 0:
        print("  [WARN] FSA molecules were not detected.")
        print("         Try --list-topology and then specify --fsa-resnames <resname>.")

    if not selected:
        print("  [SKIP] No selected atoms. Check Li/FSA names and residue names.")
        return []

    ordinal_to_selected_local = {o: i for i, o in enumerate(selected)}
    selected_local_to_ordinal = list(selected)

    li_local = np.array([ordinal_to_selected_local[o] for o in li_ord], dtype=int)
    fsa_groups_local = [
        [ordinal_to_selected_local[o] for o in group]
        for group in fsa_groups_ord
    ]

    frames = []
    boxes = []
    n_total_frames = 0

    for iframe, (coords_sel, box, angles) in enumerate(
        iter_selected_frames(pdb_path, len(atoms), selected, has_model)
    ):
        n_total_frames += 1
        if iframe < args.discard_frames:
            continue
        if (iframe - args.discard_frames) % args.stride != 0:
            continue
        frames.append(coords_sel)
        if box is not None:
            boxes.append(box.astype(np.float64))
        elif first_box is not None:
            boxes.append(first_box.astype(np.float64))
        else:
            boxes.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))

    if len(frames) < 5:
        print("  [SKIP] Too few frames after discard/stride.")
        return []

    dt_eff_ps = args.dt_ps * args.stride
    max_lag = None
    if args.max_lag_ps is not None:
        max_lag = int(round(args.max_lag_ps / dt_eff_ps))

    pos = np.asarray(frames, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float64)

    if args.no_unwrap:
        unwrapped = pos.astype(np.float64, copy=False)
        unwrap_status = "OFF"
    else:
        unwrapped = unwrap_orthorhombic(pos, boxes)
        unwrap_status = "ON if CRYST1/box is valid"

    print(f"  total frames   = {n_total_frames}")
    print(f"  used frames    = {len(frames)}")
    print(f"  effective dt   = {dt_eff_ps} ps")
    print(f"  unwrap         = {unwrap_status}")

    rows = []

    if len(li_local) > 0:
        li_pos = unwrapped[:, li_local, :]
        msd = msd_fft_multi(li_pos, max_lag=max_lag, batch_cols=args.batch_cols)
        time_ps = np.arange(len(msd), dtype=np.float64) * dt_eff_ps
        fit = fit_diffusion(time_ps, msd, args.fit_start_ps, args.fit_end_ps)
        csv_path, summary_path, png_path = save_species_result(
            outdir, pdb_stem, "Li", time_ps, msd, fit, make_png=(not args.no_png)
        )
        print(f"  [OK] Li: D = {fit['D_1e5_cm2_s']:.6g} x 10^-5 cm^2/s, R2 = {fit['r2']:.5f}")
        rows.append({
            "pdb": str(pdb_path), "system": pdb_stem, "species": "Li",
            "n_particles": len(li_local), "n_frames": len(frames), "dt_ps": dt_eff_ps,
            "fit_start_ps": fit["fit_start_ps"], "fit_end_ps": fit["fit_end_ps"],
            "D_A2_per_ps": fit["D_A2_ps"], "D_cm2_per_s": fit["D_cm2_s"],
            "D_1e-5_cm2_per_s": fit["D_1e5_cm2_s"], "R2": fit["r2"],
            "csv": str(csv_path), "png": str(png_path) if png_path else "",
        })

    if len(fsa_groups_local) > 0:
        fsa_pos = compute_com_from_unwrapped(
            unwrapped, fsa_groups_local, atoms, selected_local_to_ordinal
        )
        msd = msd_fft_multi(fsa_pos, max_lag=max_lag, batch_cols=args.batch_cols)
        time_ps = np.arange(len(msd), dtype=np.float64) * dt_eff_ps
        fit = fit_diffusion(time_ps, msd, args.fit_start_ps, args.fit_end_ps)
        csv_path, summary_path, png_path = save_species_result(
            outdir, pdb_stem, "FSA_COM", time_ps, msd, fit, make_png=(not args.no_png)
        )
        print(f"  [OK] FSA_COM: D = {fit['D_1e5_cm2_s']:.6g} x 10^-5 cm^2/s, R2 = {fit['r2']:.5f}")
        rows.append({
            "pdb": str(pdb_path), "system": pdb_stem, "species": "FSA_COM",
            "n_particles": len(fsa_groups_local), "n_frames": len(frames), "dt_ps": dt_eff_ps,
            "fit_start_ps": fit["fit_start_ps"], "fit_end_ps": fit["fit_end_ps"],
            "D_A2_per_ps": fit["D_A2_ps"], "D_cm2_per_s": fit["D_cm2_s"],
            "D_1e-5_cm2_per_s": fit["D_1e5_cm2_s"], "R2": fit["r2"],
            "csv": str(csv_path), "png": str(png_path) if png_path else "",
        })

    return rows


def write_summary_all(outdir, rows):
    outdir = Path(outdir)
    summary_csv = outdir / "summary_all_diffusion.csv"
    if not rows:
        print("[WARN] No rows to summarize.")
        return None

    cols = [
        "pdb", "system", "species", "n_particles", "n_frames", "dt_ps",
        "fit_start_ps", "fit_end_ps", "D_A2_per_ps", "D_cm2_per_s",
        "D_1e-5_cm2_per_s", "R2", "csv", "png",
    ]
    with open(summary_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            values = []
            for c in cols:
                v = r.get(c, "")
                s = str(v)
                if "," in s or " " in s:
                    s = '"' + s.replace('"', '""') + '"'
                values.append(s)
            f.write(",".join(values) + "\n")

    print("\n" + "=" * 80)
    print(f"[OK] Wrote summary: {summary_csv}")
    print("=" * 80)
    return summary_csv


def make_overlay_plots(outdir, rows):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    by_species = defaultdict(list)
    for r in rows:
        by_species[r["species"]].append(r)

    for species, items in by_species.items():
        plt.figure(figsize=(6.4, 4.4))
        for r in items:
            data = np.loadtxt(r["csv"], delimiter=",", skiprows=1)
            plt.plot(data[:, 0], data[:, 1], lw=1.5, label=r["system"])
        plt.xlabel("Time lag / ps")
        plt.ylabel(r"MSD / $\AA^2$")
        plt.title(f"MSD overlay: {species}")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        out = outdir / f"overlay_{species}_MSD.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[OK] Wrote overlay: {out}")


def expand_pdb_inputs(pdb_args):
    paths = []
    for item in pdb_args:
        matches = sorted(Path().glob(item)) if any(ch in item for ch in "*?[]") else []
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(item))
    seen = set()
    out = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Fast MSD and self-diffusion analysis from one or multiple PDB trajectories."
    )
    ap.add_argument("--pdb", nargs="+", required=True,
                    help="Input PDB trajectory files. Globs are OK, e.g. '*.pdb'.")
    ap.add_argument("--dt-ps", type=float, required=True,
                    help="Time interval between saved PDB frames in ps.")
    ap.add_argument("--outdir", default="msd_pdb_results",
                    help="Output directory. Created automatically.")

    ap.add_argument("--discard-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-lag-ps", type=float, default=None,
                    help="Maximum lag time for MSD. Recommended: total trajectory length / 3 to 1/2.")
    ap.add_argument("--fit-start-ps", type=float, default=None)
    ap.add_argument("--fit-end-ps", type=float, default=None)

    ap.add_argument("--li-names", nargs="*", default=["Li", "LI", "Li+"],
                    help="Atom names treated as Li.")
    ap.add_argument("--li-resnames", nargs="*", default=[],
                    help="Residue names treated as Li.")

    ap.add_argument("--fsa-resnames", nargs="*", default=["FSA", "FSI", "TFSI", "NFS"],
                    help="Residue names treated as FSA/FSI molecules.")
    ap.add_argument("--no-auto-fsa", action="store_true",
                    help="Disable auto-detection of FSA-like residues containing S, F, and O.")

    ap.add_argument("--no-unwrap", action="store_true",
                    help="Do not unwrap coordinates using CRYST1 box.")
    ap.add_argument("--batch-cols", type=int, default=256,
                    help="FFT batch size. Lower this if memory is tight.")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")

    ap.add_argument("--list-topology", action="store_true",
                    help="Print atom/residue summary before analysis.")
    ap.add_argument("--only-list-topology", action="store_true",
                    help="Only print topology summary and exit.")

    args = ap.parse_args()

    pdb_paths = expand_pdb_inputs(args.pdb)
    missing = [str(p) for p in pdb_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing PDB file(s): " + ", ".join(missing))

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    all_rows = []
    for pdb_path in pdb_paths:
        rows = analyze_one_pdb(pdb_path, args)
        all_rows.extend(rows)

    write_summary_all(args.outdir, all_rows)
    if (not args.no_overlay) and (not args.no_png):
        make_overlay_plots(args.outdir, all_rows)


if __name__ == "__main__":
    main()

