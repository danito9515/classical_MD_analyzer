#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fast_msd_multiT_pdb_ns_conductivity_v1_0_0.py
=================================================

Purpose
-------
Fast analysis of mean squared displacement (MSD), self-diffusion coefficients,
and approximate ionic conductivity from one or multiple PDB trajectory files.

This version supports multi-temperature datasets. For each PDB, the temperature
can be inferred from the filename, e.g. Liquid_nvt300K_120ns.pdb,
Liquid_1Vac_nvt350K.pdb, or T400K_run.pdb. A manual temperature map can also
be supplied when filenames do not contain a usable temperature tag.

This script was designed for Tinker/Tinker-GPU PDB trajectories such as LiFSA
systems, but it should also work for general MODEL/ENDMDL-style PDB trajectories.
It uses FFT-based window-averaged MSD evaluation, so it is much faster than a
naive double loop over all time origins and lag times.

Main outputs
------------
For each input PDB and each detected species:
  - *_msd.csv
      time_ns, MSD_A2, instantaneous Einstein D estimate, and related columns.
  - *_summary.txt
      fitted self-diffusion coefficient and Nernst-Einstein conductivity estimate.
  - *_msd_loglog.png
      log-log MSD plot. A slope close to 1 in the fitting region indicates
      approximately normal diffusion.

For all systems together:
  - summary_all_diffusion.csv
      one row per species per PDB trajectory.
  - summary_system_conductivity_NE.csv
      total Nernst-Einstein conductivity summed over detected charged species
      for each PDB trajectory.
  - overlay_<species>_MSD_loglog.png
      overlay log-log MSD plots across PDB trajectories.
  - temperature_<species>_D.png
      diffusion coefficient versus temperature.
  - arrhenius_<species>_D.png
      log10(D) versus 1000/T diagnostic plot.
  - temperature_<species>_sigma_NE.png
      Nernst-Einstein conductivity versus temperature.

Units
-----
Time is handled in ns throughout the user interface and output files.
Coordinates are in Angstrom, so MSD is in Angstrom^2.
The fitted diffusion coefficient is reported as:
  - A^2/ns
  - m^2/s
  - cm^2/s
  - 10^-5 cm^2/s

Conversion:
  1 A^2/ns = 1.0e-11 m^2/s = 1.0e-7 cm^2/s

Diffusion model
---------------
The self-diffusion coefficient is obtained from the 3D Einstein relation:

  D = (1/6) d<|r(t+tau)-r(t)|^2>/d tau

where tau is the lag time. The fit is done in linear MSD-vs-time space over the
specified fitting window. The PNG is shown as log-log only for diagnosis and
visualization.

Conductivity model
------------------
The conductivity is estimated using the ideal Nernst-Einstein approximation:

  sigma_NE = sum_i n_i z_i^2 e^2 D_i / (k_B T)

where n_i is the number density of mobile species i, z_i is the ionic charge,
e is the elementary charge, k_B is the Boltzmann constant, and T is temperature.
This neglects ion-ion velocity correlations and should be interpreted as an
upper-bound-like ideal estimate, not a rigorous Green-Kubo conductivity.

For PDB trajectories with CRYST1 records, the average box volume is used to
estimate number density. If valid CRYST1 box lengths are absent, conductivity is
reported as NaN while diffusion is still computed.

Important assumptions and cautions
----------------------------------
1. Coordinates are assumed to be wrapped into an orthorhombic box. The script
   unwraps trajectories using CRYST1 box lengths and a minimum-image convention.
   This assumes displacement between saved frames is less than half the box.
2. FSA/FSI-like anions are detected by residue names or, if needed, by residues
   containing S, F, and O atoms. The FSA result is for molecular COM diffusion.
3. If FSA molecules are split across periodic boundaries in the PDB, COM can be
   noisy. In that case, use a whole-molecule trajectory or analyze Tinker arc/xyz.
4. For interfacial solid/liquid systems, 3D diffusion coefficients are effective
   values. Check the MSD shape carefully.

Typical usage
-------------
Recommended ns-based multi-temperature usage:

  python3 fast_msd_multiT_pdb_ns_conductivity_v1_0_0.py \
    --pdb 'Liquid*nvt*K*.pdb' \
    --dt-ns 0.05 \
    --outdir msd_multiT_results_ns \
    --fit-start-ns 1.0 \
    --fit-end-ns 30.0 \
    --max-lag-ns 30.0 \
    --list-topology

Temperature is inferred from each filename by default. For example,
Liquid_nvt300K_120ns.pdb -> 300 K, Liquid_1Vac_nvt350K.pdb -> 350 K,
and T400K/run.pdb -> 400 K. If a file name does not contain a temperature,
--temperature-k is used as a fallback. Manual mapping is also accepted:

  python3 fast_msd_multiT_pdb_ns_conductivity_v1_0_0.py \
    --pdb fileA.pdb fileB.pdb \
    --temperature-map fileA.pdb:300 fileB.pdb:400 \
    --dt-ps 50.0

Backward-compatible ps-based input is also accepted and converted internally.
"""

import argparse
import gzip
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "S": 32.06, "Li": 6.94, "Na": 22.990,
    "K": 39.098, "Cl": 35.45,
}

# SI constants for Nernst-Einstein conductivity.
E_CHARGE_C = 1.602176634e-19
K_B_J_PER_K = 1.380649e-23
N_A_PER_MOL = 6.02214076e23


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


def infer_temperature_from_name(path):
    """Infer temperature in K from filename/path.

    Recognized examples:
      Liquid_nvt300K_120ns.pdb -> 300
      Liquid_1Vac_nvt350K.pdb  -> 350
      T400K/run.pdb            -> 400
      temp_300_K.pdb           -> 300
    """
    text = str(path)
    patterns = [
        r"(?:^|[^A-Za-z0-9])T\s*([0-9]+(?:\.[0-9]+)?)\s*K(?:[^A-Za-z0-9]|$)",
        r"(?:nvt|npt|nve)\s*([0-9]+(?:\.[0-9]+)?)\s*K",
        r"([0-9]+(?:\.[0-9]+)?)\s*_?K(?:[^A-Za-z0-9]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def parse_temperature_map(entries):
    """Parse entries like file.pdb:300 or stem=300."""
    mapping = {}
    for item in entries or []:
        if ":" in item:
            key, value = item.rsplit(":", 1)
        elif "=" in item:
            key, value = item.rsplit("=", 1)
        else:
            raise ValueError(
                f"Invalid --temperature-map entry: {item}. "
                "Use file.pdb:300 or system_stem=300."
            )
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in --temperature-map entry: {item}")
        mapping[key] = float(value)
    return mapping


def resolve_temperature_for_pdb(pdb_path, args):
    """Resolve per-PDB temperature and source label."""
    p = Path(pdb_path)
    candidates = [str(p), p.name, safe_stem(p)]
    temp_map = getattr(args, "temperature_map_dict", {})
    for key in candidates:
        if key in temp_map:
            return float(temp_map[key]), f"map:{key}"

    if getattr(args, "temperature_from_name", True):
        t = infer_temperature_from_name(p)
        if t is not None:
            return float(t), "filename"

    if args.temperature_k is not None:
        return float(args.temperature_k), "fallback"

    raise ValueError(
        f"Could not determine temperature for {pdb_path}. "
        "Add a temperature tag such as 300K to the filename, "
        "or use --temperature-map file.pdb:300, or --temperature-k 300."
    )


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


def orthorhombic_volume_A3(box):
    if box is None:
        return np.nan
    box = np.asarray(box, dtype=np.float64)
    if box.size != 3 or np.any(~np.isfinite(box)) or np.any(box <= 0):
        return np.nan
    return float(np.prod(box))


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
        print(f"vol   = {orthorhombic_volume_A3(first_box):.6f} A^3")

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
    Returns list of FSA/FSI-like groups as lists of atom ordinals.

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


def fit_diffusion(time_ns, msd_A2, fit_start_ns=None, fit_end_ns=None):
    if fit_start_ns is None:
        fit_start_ns = time_ns[int(0.2 * len(time_ns))]
    if fit_end_ns is None:
        fit_end_ns = time_ns[int(0.8 * len(time_ns))]

    mask = (time_ns >= fit_start_ns) & (time_ns <= fit_end_ns) & (time_ns > 0)
    if mask.sum() < 3:
        raise RuntimeError(
            f"Too few points in fitting window: {fit_start_ns} - {fit_end_ns} ns. "
            f"Try smaller --fit-start-ns or larger --fit-end-ns."
        )

    slope, intercept = np.polyfit(time_ns[mask], msd_A2[mask], 1)
    pred = slope * time_ns[mask] + intercept
    ss_res = np.sum((msd_A2[mask] - pred) ** 2)
    ss_tot = np.sum((msd_A2[mask] - np.mean(msd_A2[mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Optional log-log exponent in the same fit window.
    log_mask = mask & (msd_A2 > 0) & (time_ns > 0)
    if log_mask.sum() >= 3:
        alpha, log_intercept = np.polyfit(np.log10(time_ns[log_mask]), np.log10(msd_A2[log_mask]), 1)
    else:
        alpha, log_intercept = np.nan, np.nan

    D_A2_ns = slope / 6.0
    D_m2_s = D_A2_ns * 1.0e-11
    D_cm2_s = D_A2_ns * 1.0e-7
    D_1e5_cm2_s = D_cm2_s / 1.0e-5

    return {
        "fit_start_ns": float(fit_start_ns),
        "fit_end_ns": float(fit_end_ns),
        "slope_A2_ns": float(slope),
        "intercept_A2": float(intercept),
        "D_A2_ns": float(D_A2_ns),
        "D_m2_s": float(D_m2_s),
        "D_cm2_s": float(D_cm2_s),
        "D_1e5_cm2_s": float(D_1e5_cm2_s),
        "loglog_alpha": float(alpha),
        "r2": float(r2),
    }


def compute_ne_conductivity(n_particles, volume_A3, D_m2_s, charge_number, temperature_k):
    if (
        n_particles <= 0
        or not np.isfinite(volume_A3)
        or volume_A3 <= 0
        or not np.isfinite(D_m2_s)
        or not np.isfinite(temperature_k)
        or temperature_k <= 0
    ):
        return {
            "number_density_m3": np.nan,
            "concentration_mol_L": np.nan,
            "sigma_NE_S_m": np.nan,
            "sigma_NE_mS_cm": np.nan,
        }

    volume_m3 = volume_A3 * 1.0e-30
    number_density_m3 = n_particles / volume_m3
    concentration_mol_L = number_density_m3 / (N_A_PER_MOL * 1000.0)
    sigma_S_m = number_density_m3 * (charge_number ** 2) * (E_CHARGE_C ** 2) * D_m2_s / (K_B_J_PER_K * temperature_k)
    sigma_mS_cm = sigma_S_m * 10.0

    return {
        "number_density_m3": float(number_density_m3),
        "concentration_mol_L": float(concentration_mol_L),
        "sigma_NE_S_m": float(sigma_S_m),
        "sigma_NE_mS_cm": float(sigma_mS_cm),
    }


def save_species_result(outdir, pdb_stem, species, time_ns, msd_A2, fit, conductivity, temperature_k=None, make_png=True):
    outdir = Path(outdir)
    csv_path = outdir / f"{pdb_stem}_{species}_msd.csv"
    summary_path = outdir / f"{pdb_stem}_{species}_summary.txt"
    png_path = outdir / f"{pdb_stem}_{species}_msd_loglog.png"

    D_from_origin_A2_ns = np.full_like(time_ns, np.nan, dtype=np.float64)
    valid = time_ns > 0
    D_from_origin_A2_ns[valid] = msd_A2[valid] / (6.0 * time_ns[valid])
    D_from_origin_m2_s = D_from_origin_A2_ns * 1.0e-11
    D_from_origin_cm2_s = D_from_origin_A2_ns * 1.0e-7

    arr = np.column_stack([
        time_ns,
        msd_A2,
        D_from_origin_A2_ns,
        D_from_origin_m2_s,
        D_from_origin_cm2_s,
        D_from_origin_cm2_s / 1.0e-5,
    ])
    np.savetxt(
        csv_path,
        arr,
        delimiter=",",
        header="time_ns,MSD_A2,D_from_origin_A2_per_ns,D_from_origin_m2_per_s,D_from_origin_cm2_per_s,D_from_origin_1e-5_cm2_per_s",
        comments="",
    )

    with open(summary_path, "w") as f:
        f.write(f"pdb = {pdb_stem}\n")
        f.write(f"species = {species}\n")
        f.write(f"fit_start_ns = {fit['fit_start_ns']:.10g}\n")
        f.write(f"fit_end_ns = {fit['fit_end_ns']:.10g}\n")
        f.write(f"slope_A2_per_ns = {fit['slope_A2_ns']:.12g}\n")
        f.write(f"D_A2_per_ns = {fit['D_A2_ns']:.12g}\n")
        f.write(f"D_m2_per_s = {fit['D_m2_s']:.12g}\n")
        f.write(f"D_cm2_per_s = {fit['D_cm2_s']:.12g}\n")
        f.write(f"D_1e-5_cm2_per_s = {fit['D_1e5_cm2_s']:.12g}\n")
        f.write(f"MSD_loglog_alpha_in_fit_window = {fit['loglog_alpha']:.8g}\n")
        f.write(f"linear_fit_R2 = {fit['r2']:.8g}\n")
        f.write("\n[Nernst-Einstein conductivity estimate]\n")
        f.write(f"number_density_m^-3 = {conductivity['number_density_m3']:.12g}\n")
        f.write(f"concentration_mol_L = {conductivity['concentration_mol_L']:.12g}\n")
        f.write(f"sigma_NE_S_m = {conductivity['sigma_NE_S_m']:.12g}\n")
        f.write(f"sigma_NE_mS_cm = {conductivity['sigma_NE_mS_cm']:.12g}\n")

    if make_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid_plot = (time_ns > 0) & (msd_A2 > 0)
        fit_mask = (
            valid_plot
            & (time_ns >= fit["fit_start_ns"])
            & (time_ns <= fit["fit_end_ns"])
        )

        plt.figure(figsize=(6.2, 4.5))
        plt.loglog(time_ns[valid_plot], msd_A2[valid_plot], lw=1.8, label="MSD")

        if fit_mask.sum() >= 2:
            yfit = fit["slope_A2_ns"] * time_ns[fit_mask] + fit["intercept_A2"]
            yfit_mask = yfit > 0
            if np.any(yfit_mask):
                plt.loglog(time_ns[fit_mask][yfit_mask], yfit[yfit_mask], "--", lw=1.5, label="linear fit")

        plt.xlabel("Time lag / ns")
        plt.ylabel(r"MSD / $\AA^2$")
        temp_text = f" | {temperature_k:g} K" if temperature_k is not None else ""
        plt.title(
            f"{pdb_stem} | {species}{temp_text}\n"
            f"D = {fit['D_1e5_cm2_s']:.4g} × 10$^{{-5}}$ cm$^2$/s, "
            f"σ_NE = {conductivity['sigma_NE_mS_cm']:.4g} mS/cm"
        )
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()

    return csv_path, summary_path, png_path if make_png else None


def resolve_time_options(args):
    if args.dt_ns is None and args.dt_ps is None:
        raise ValueError("Please specify --dt-ns, e.g. --dt-ns 0.05. Legacy --dt-ps is also accepted.")

    dt_ns = args.dt_ns if args.dt_ns is not None else args.dt_ps / 1000.0

    fit_start_ns = args.fit_start_ns
    fit_end_ns = args.fit_end_ns
    max_lag_ns = args.max_lag_ns

    if fit_start_ns is None and args.fit_start_ps is not None:
        fit_start_ns = args.fit_start_ps / 1000.0
    if fit_end_ns is None and args.fit_end_ps is not None:
        fit_end_ns = args.fit_end_ps / 1000.0
    if max_lag_ns is None and args.max_lag_ps is not None:
        max_lag_ns = args.max_lag_ps / 1000.0

    return dt_ns, fit_start_ns, fit_end_ns, max_lag_ns


def analyze_one_pdb(pdb_path, args):
    dt_ns, fit_start_ns, fit_end_ns, max_lag_ns = resolve_time_options(args)

    pdb_path = Path(pdb_path)
    pdb_stem = safe_stem(pdb_path)
    temperature_k, temperature_source = resolve_temperature_for_pdb(pdb_path, args)
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

    dt_eff_ns = dt_ns * args.stride
    max_lag = None
    if max_lag_ns is not None:
        max_lag = int(round(max_lag_ns / dt_eff_ns))

    pos = np.asarray(frames, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float64)
    volumes_A3 = np.array([orthorhombic_volume_A3(b) for b in boxes], dtype=np.float64)
    valid_volumes = volumes_A3[np.isfinite(volumes_A3) & (volumes_A3 > 0)]
    avg_volume_A3 = float(np.mean(valid_volumes)) if valid_volumes.size else np.nan

    if args.no_unwrap:
        unwrapped = pos.astype(np.float64, copy=False)
        unwrap_status = "OFF"
    else:
        unwrapped = unwrap_orthorhombic(pos, boxes)
        unwrap_status = "ON if CRYST1/box is valid"

    print(f"  total frames   = {n_total_frames}")
    print(f"  used frames    = {len(frames)}")
    print(f"  effective dt   = {dt_eff_ns:.10g} ns")
    print(f"  avg volume     = {avg_volume_A3:.6g} A^3")
    print(f"  temperature    = {temperature_k:.6g} K ({temperature_source})")
    print(f"  unwrap         = {unwrap_status}")

    rows = []

    if len(li_local) > 0:
        li_pos = unwrapped[:, li_local, :]
        msd = msd_fft_multi(li_pos, max_lag=max_lag, batch_cols=args.batch_cols)
        time_ns = np.arange(len(msd), dtype=np.float64) * dt_eff_ns
        fit = fit_diffusion(time_ns, msd, fit_start_ns, fit_end_ns)
        cond = compute_ne_conductivity(
            n_particles=len(li_local),
            volume_A3=avg_volume_A3,
            D_m2_s=fit["D_m2_s"],
            charge_number=args.li_charge,
            temperature_k=temperature_k,
        )
        csv_path, summary_path, png_path = save_species_result(
            outdir, pdb_stem, "Li", time_ns, msd, fit, cond, temperature_k=temperature_k, make_png=(not args.no_png)
        )
        print(
            f"  [OK] Li: D = {fit['D_1e5_cm2_s']:.6g} x 10^-5 cm^2/s, "
            f"sigma_NE = {cond['sigma_NE_mS_cm']:.6g} mS/cm, R2 = {fit['r2']:.5f}"
        )
        rows.append({
            "pdb": str(pdb_path), "system": pdb_stem, "species": "Li",
            "charge_number": args.li_charge,
            "n_particles": len(li_local), "n_frames": len(frames), "dt_ns": dt_eff_ns,
            "avg_volume_A3": avg_volume_A3, "temperature_K": temperature_k, "temperature_source": temperature_source,
            "fit_start_ns": fit["fit_start_ns"], "fit_end_ns": fit["fit_end_ns"],
            "slope_A2_per_ns": fit["slope_A2_ns"], "D_A2_per_ns": fit["D_A2_ns"],
            "D_m2_per_s": fit["D_m2_s"], "D_cm2_per_s": fit["D_cm2_s"],
            "D_1e-5_cm2_per_s": fit["D_1e5_cm2_s"],
            "MSD_loglog_alpha": fit["loglog_alpha"], "R2": fit["r2"],
            "number_density_m^-3": cond["number_density_m3"],
            "concentration_mol_L": cond["concentration_mol_L"],
            "sigma_NE_S_m": cond["sigma_NE_S_m"],
            "sigma_NE_mS_cm": cond["sigma_NE_mS_cm"],
            "csv": str(csv_path), "png": str(png_path) if png_path else "",
        })

    if len(fsa_groups_local) > 0:
        fsa_pos = compute_com_from_unwrapped(
            unwrapped, fsa_groups_local, atoms, selected_local_to_ordinal
        )
        msd = msd_fft_multi(fsa_pos, max_lag=max_lag, batch_cols=args.batch_cols)
        time_ns = np.arange(len(msd), dtype=np.float64) * dt_eff_ns
        fit = fit_diffusion(time_ns, msd, fit_start_ns, fit_end_ns)
        cond = compute_ne_conductivity(
            n_particles=len(fsa_groups_local),
            volume_A3=avg_volume_A3,
            D_m2_s=fit["D_m2_s"],
            charge_number=args.fsa_charge,
            temperature_k=temperature_k,
        )
        csv_path, summary_path, png_path = save_species_result(
            outdir, pdb_stem, "FSA_COM", time_ns, msd, fit, cond, temperature_k=temperature_k, make_png=(not args.no_png)
        )
        print(
            f"  [OK] FSA_COM: D = {fit['D_1e5_cm2_s']:.6g} x 10^-5 cm^2/s, "
            f"sigma_NE = {cond['sigma_NE_mS_cm']:.6g} mS/cm, R2 = {fit['r2']:.5f}"
        )
        rows.append({
            "pdb": str(pdb_path), "system": pdb_stem, "species": "FSA_COM",
            "charge_number": args.fsa_charge,
            "n_particles": len(fsa_groups_local), "n_frames": len(frames), "dt_ns": dt_eff_ns,
            "avg_volume_A3": avg_volume_A3, "temperature_K": temperature_k, "temperature_source": temperature_source,
            "fit_start_ns": fit["fit_start_ns"], "fit_end_ns": fit["fit_end_ns"],
            "slope_A2_per_ns": fit["slope_A2_ns"], "D_A2_per_ns": fit["D_A2_ns"],
            "D_m2_per_s": fit["D_m2_s"], "D_cm2_per_s": fit["D_cm2_s"],
            "D_1e-5_cm2_per_s": fit["D_1e5_cm2_s"],
            "MSD_loglog_alpha": fit["loglog_alpha"], "R2": fit["r2"],
            "number_density_m^-3": cond["number_density_m3"],
            "concentration_mol_L": cond["concentration_mol_L"],
            "sigma_NE_S_m": cond["sigma_NE_S_m"],
            "sigma_NE_mS_cm": cond["sigma_NE_mS_cm"],
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
        "pdb", "system", "species", "charge_number", "n_particles", "n_frames", "dt_ns",
        "avg_volume_A3", "temperature_K", "temperature_source", "fit_start_ns", "fit_end_ns",
        "slope_A2_per_ns", "D_A2_per_ns", "D_m2_per_s", "D_cm2_per_s",
        "D_1e-5_cm2_per_s", "MSD_loglog_alpha", "R2",
        "number_density_m^-3", "concentration_mol_L",
        "sigma_NE_S_m", "sigma_NE_mS_cm", "csv", "png",
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


def write_system_conductivity_summary(outdir, rows):
    outdir = Path(outdir)
    out_csv = outdir / "summary_system_conductivity_NE.csv"
    if not rows:
        return None

    by_system = defaultdict(list)
    for r in rows:
        by_system[(r["pdb"], r["system"])].append(r)

    cols = [
        "pdb", "system", "temperature_K", "avg_volume_A3",
        "sigma_total_NE_S_m", "sigma_total_NE_mS_cm", "species_included",
    ]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for (pdb, system), items in sorted(by_system.items(), key=lambda x: x[0][1]):
            sigma_total_S_m = np.nansum([r["sigma_NE_S_m"] for r in items])
            sigma_total_mS_cm = sigma_total_S_m * 10.0
            temp = items[0].get("temperature_K", np.nan)
            vol = items[0].get("avg_volume_A3", np.nan)
            species = "+".join(r["species"] for r in items)
            row = {
                "pdb": pdb,
                "system": system,
                "temperature_K": temp,
                "avg_volume_A3": vol,
                "sigma_total_NE_S_m": sigma_total_S_m,
                "sigma_total_NE_mS_cm": sigma_total_mS_cm,
                "species_included": species,
            }
            values = []
            for c in cols:
                s = str(row[c])
                if "," in s or " " in s:
                    s = '"' + s.replace('"', '""') + '"'
                values.append(s)
            f.write(",".join(values) + "\n")

    print(f"[OK] Wrote system conductivity summary: {out_csv}")
    return out_csv


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
        items = sorted(items, key=lambda r: (float(r.get("temperature_K", np.nan)), r.get("system", "")))
        plt.figure(figsize=(6.4, 4.6))
        for r in items:
            data = np.loadtxt(r["csv"], delimiter=",", skiprows=1)
            time_ns = data[:, 0]
            msd = data[:, 1]
            valid = (time_ns > 0) & (msd > 0)
            if np.any(valid):
                label = f"{r['system']} ({float(r.get('temperature_K', np.nan)):g} K)"
                plt.loglog(time_ns[valid], msd[valid], lw=1.5, label=label)
        plt.xlabel("Time lag / ns")
        plt.ylabel(r"MSD / $\AA^2$")
        plt.title(f"MSD overlay: {species}")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        out = outdir / f"overlay_{species}_MSD_loglog.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[OK] Wrote overlay: {out}")


def make_temperature_plots(outdir, rows):
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
        clean = []
        for r in items:
            try:
                T = float(r.get("temperature_K", np.nan))
                D = float(r.get("D_1e-5_cm2_per_s", np.nan))
                sigma = float(r.get("sigma_NE_mS_cm", np.nan))
            except Exception:
                continue
            if np.isfinite(T):
                clean.append((T, D, sigma, r.get("system", "")))

        if not clean:
            continue

        clean.sort(key=lambda x: (x[0], x[3]))
        T = np.array([x[0] for x in clean], dtype=float)
        D = np.array([x[1] for x in clean], dtype=float)
        sigma = np.array([x[2] for x in clean], dtype=float)
        labels = [x[3] for x in clean]

        # Diffusion coefficient versus temperature.
        valid_D = np.isfinite(T) & np.isfinite(D)
        if np.any(valid_D):
            plt.figure(figsize=(6.0, 4.4))
            plt.plot(T[valid_D], D[valid_D], "o-", lw=1.5)
            for x, y, lab in zip(T[valid_D], D[valid_D], np.array(labels, dtype=object)[valid_D]):
                plt.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
            plt.xlabel("Temperature / K")
            plt.ylabel(r"D / $10^{-5}$ cm$^2$ s$^{-1}$")
            plt.title(f"Diffusion coefficient vs temperature: {species}")
            plt.tight_layout()
            out = outdir / f"temperature_{species}_D.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"[OK] Wrote temperature plot: {out}")

        # Arrhenius diagnostic: log10(D) versus 1000/T.
        valid_A = np.isfinite(T) & np.isfinite(D) & (T > 0) & (D > 0)
        if np.any(valid_A):
            plt.figure(figsize=(6.0, 4.4))
            x = 1000.0 / T[valid_A]
            y = np.log10(D[valid_A])
            plt.plot(x, y, "o-", lw=1.5)
            for xi, yi, lab in zip(x, y, np.array(labels, dtype=object)[valid_A]):
                plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
            plt.xlabel(r"1000 / T / K$^{-1}$")
            plt.ylabel(r"log$_{10}$(D / $10^{-5}$ cm$^2$ s$^{-1}$)")
            plt.title(f"Arrhenius diagnostic: {species}")
            plt.tight_layout()
            out = outdir / f"arrhenius_{species}_D.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"[OK] Wrote Arrhenius plot: {out}")

        # Nernst-Einstein conductivity versus temperature.
        valid_s = np.isfinite(T) & np.isfinite(sigma)
        if np.any(valid_s):
            plt.figure(figsize=(6.0, 4.4))
            plt.plot(T[valid_s], sigma[valid_s], "o-", lw=1.5)
            for x, y, lab in zip(T[valid_s], sigma[valid_s], np.array(labels, dtype=object)[valid_s]):
                plt.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
            plt.xlabel("Temperature / K")
            plt.ylabel(r"$\sigma_{NE}$ / mS cm$^{-1}$")
            plt.title(f"Nernst-Einstein conductivity vs temperature: {species}")
            plt.tight_layout()
            out = outdir / f"temperature_{species}_sigma_NE.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"[OK] Wrote conductivity plot: {out}")


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
        description="Fast MSD, self-diffusion, and Nernst-Einstein conductivity analysis from one or multiple PDB trajectories."
    )
    ap.add_argument("--pdb", nargs="+", required=True,
                    help="Input PDB trajectory files. Globs are OK, e.g. '*.pdb'.")
    ap.add_argument("--dt-ns", type=float, default=None,
                    help="Time interval between saved PDB frames in ns. Recommended new option, e.g. 0.05 for 50 ps.")
    ap.add_argument("--dt-ps", type=float, default=None,
                    help="Legacy option: time interval between saved PDB frames in ps. Converted to ns internally.")
    ap.add_argument("--outdir", default="msd_pdb_results_ns",
                    help="Output directory. Created automatically.")

    ap.add_argument("--discard-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-lag-ns", type=float, default=None,
                    help="Maximum lag time for MSD in ns. Recommended: total trajectory length / 3 to 1/2.")
    ap.add_argument("--fit-start-ns", type=float, default=None)
    ap.add_argument("--fit-end-ns", type=float, default=None)

    # Backward-compatible ps options. Output still uses ns.
    ap.add_argument("--max-lag-ps", type=float, default=None,
                    help="Legacy option. Converted to ns internally.")
    ap.add_argument("--fit-start-ps", type=float, default=None,
                    help="Legacy option. Converted to ns internally.")
    ap.add_argument("--fit-end-ps", type=float, default=None,
                    help="Legacy option. Converted to ns internally.")

    ap.add_argument("--temperature-k", type=float, default=300.0,
                    help="Fallback temperature for Nernst-Einstein conductivity in K when per-file temperature cannot be inferred. Default: 300 K.")
    ap.add_argument("--temperature-map", nargs="*", default=[],
                    help="Optional per-file temperature map, e.g. Liquid300.pdb:300 Liquid400.pdb:400 or system_stem=350.")
    ap.add_argument("--no-temperature-from-name", dest="temperature_from_name", action="store_false",
                    help="Disable automatic temperature inference from filename/path. Default: enabled.")
    ap.set_defaults(temperature_from_name=True)
    ap.add_argument("--li-charge", type=float, default=1.0,
                    help="Charge number z for Li. Default: +1.")
    ap.add_argument("--fsa-charge", type=float, default=-1.0,
                    help="Charge number z for FSA/FSI. Default: -1. Conductivity uses z^2.")

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
    ap.add_argument("--no-temperature-plots", action="store_true",
                    help="Do not make D-vs-T, Arrhenius, and sigma-vs-T summary plots.")

    ap.add_argument("--list-topology", action="store_true",
                    help="Print atom/residue summary before analysis.")
    ap.add_argument("--only-list-topology", action="store_true",
                    help="Only print topology summary and exit.")

    args = ap.parse_args()

    # Validate time options early.
    resolve_time_options(args)
    args.temperature_map_dict = parse_temperature_map(args.temperature_map)

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
    write_system_conductivity_summary(args.outdir, all_rows)
    if (not args.no_overlay) and (not args.no_png):
        make_overlay_plots(args.outdir, all_rows)
    if (not args.no_temperature_plots) and (not args.no_png):
        make_temperature_plots(args.outdir, all_rows)


if __name__ == "__main__":
    main()
