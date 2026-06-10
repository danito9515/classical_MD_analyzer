#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example Usage: python3 ../../analyze_TinkerGpu_npt_joinruns_v1.1.0.py --root ./ --temps T300K --force

Analyze Tinker-GPU / Tinker-HP NPT results for multiple temperature directories.
Continuation directories such as T300K_run_002 are automatically stitched into T300K.

Expected directory structure:

    .
    ├── T250K/
    │   ├── *_log_*.out
    │   ├── *.001
    │   ├── *.002
    │   └── ...
    ├── T300K/
    ├── T350K/
    └── analysis_result/

This script extracts:
    - potential energy
    - kinetic energy
    - total energy
    - temperature
    - pressure
    - lattice constants a, b, c, alpha, beta, gamma
    - cell volume
    - cumulative averages of all numeric observables

Outputs:
    analysis_result/T250K_timeseries.csv
    analysis_result/T300K_timeseries.csv
    analysis_result/T350K_timeseries.csv
    analysis_result/all_temperatures_timeseries.csv
    analysis_result/summary.csv
    analysis_result/png/*.png
"""

import argparse
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shlex

# ============================================================
# Basic utilities
# ============================================================

FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)
AVOGADRO = 6.02214076e23
ANG3_TO_CM3 = 1.0e-24



def to_float(x):
    """Convert string to float, accepting Fortran D exponent."""
    return float(x.replace("D", "E").replace("d", "e"))


def extract_floats(line):
    return [to_float(x) for x in FLOAT_RE.findall(line)]


def infer_temp_from_path(path: Path):
    """
    Infer temperature from directory/file name like T250K.
    """
    m = re.search(r"T(\d+(?:\.\d+)?)K", str(path))
    if m:
        return float(m.group(1))
    return np.nan



def temperature_series_label(path: Path):
    """
    Return the base temperature label.

    Examples:
        T300K           -> T300K
        T300K_run_002   -> T300K
        T300K_restart03 -> T300K

    This lets NPT continuation directories be treated as one trajectory.
    """
    m = re.search(r"(T\d+(?:\.\d+)?K)", path.name)
    if m:
        return m.group(1)
    return path.name


def run_number_from_path(path: Path):
    """
    Infer continuation run number from directory name.

    Examples:
        T300K         -> 1
        T300K_run_002 -> 2
    """
    m = re.search(r"_run_(\d+)", path.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1


def sort_key_for_run_dir(path: Path):
    """
    Sort by temperature first and continuation index second.
    """
    return (
        infer_temp_from_path(path),
        temperature_series_label(path),
        run_number_from_path(path),
        path.name,
    )


def estimate_time_step_ps(time_values, fallback=50.0):
    """
    Estimate the output interval from existing time values.
    Used when stitching restart runs whose local time resets.
    """
    try:
        vals = pd.to_numeric(pd.Series(time_values), errors="coerce").dropna().unique()
        vals = np.sort(vals)
        diffs = np.diff(vals)
        diffs = diffs[diffs > 1.0e-9]
        if len(diffs) > 0:
            return float(np.median(diffs))
    except Exception:
        pass
    return float(fallback)


def discover_temperature_groups(root: Path, temps=None, join_continuations=True):
    """
    Discover temperature directories and optionally group continuation runs.

    With join_continuations=True:
        T300K + T300K_run_002 + T300K_run_003 -> one group labeled T300K

    If --temps T300K is given, sibling continuation directories are also included.
    If --temps T300K_run_002 is given, all siblings with the same T300K label are included.
    """

    def is_temperature_dir(p: Path):
        return p.is_dir() and re.search(r"T\d+(?:\.\d+)?K", p.name)

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
            # Also keep explicitly requested paths even if they do not match the strict pattern.
            candidate_dirs.extend(requested)
            # unique while preserving sorted order
            candidate_dirs = sorted(set(candidate_dirs), key=sort_key_for_run_dir)
        else:
            candidate_dirs = requested

    if not join_continuations:
        return [(p.name, [p]) for p in sorted(candidate_dirs, key=sort_key_for_run_dir)]

    groups = {}
    for d in candidate_dirs:
        label = temperature_series_label(d)
        groups.setdefault(label, [])
        if d not in groups[label]:
            groups[label].append(d)

    grouped = []
    for label, dirs in groups.items():
        dirs = sorted(dirs, key=sort_key_for_run_dir)
        grouped.append((label, dirs))

    grouped.sort(key=lambda x: (infer_temp_from_path(Path(x[0])), x[0]))
    return grouped


def cell_volume(a, b, c, alpha, beta, gamma):
    """
    Compute triclinic cell volume from lengths and angles.
    Angles are in degrees.
    """
    try:
        ar = math.radians(alpha)
        br = math.radians(beta)
        gr = math.radians(gamma)
        factor = (
            1.0
            - math.cos(ar) ** 2
            - math.cos(br) ** 2
            - math.cos(gr) ** 2
            + 2.0 * math.cos(ar) * math.cos(br) * math.cos(gr)
        )
        if factor < 0:
            return np.nan
        return a * b * c * math.sqrt(factor)
    except Exception:
        return np.nan

def parse_atom_masses_from_prm(prm_file: Path):
    """
    Parse Tinker prm atom mass table.

    Tinker atom line is typically:
        atom  type  class  name  "description"  atomic_number  mass  valence

    Returns:
        dict: atom_type -> atomic_mass_g_mol
    """

    masses = {}

    with prm_file.open("r", errors="ignore") as f:
        for line in f:
            line_strip = line.strip()

            if not line_strip:
                continue

            if line_strip.startswith("#"):
                continue

            if not re.match(r"^\s*atom\s+", line, flags=re.IGNORECASE):
                continue

            try:
                tokens = shlex.split(line_strip)
            except Exception:
                tokens = line_strip.split()

            if len(tokens) < 6:
                continue

            try:
                atom_type = int(tokens[1])
            except Exception:
                continue

            numeric_tail = []
            for tok in tokens[4:]:
                try:
                    numeric_tail.append(to_float(tok))
                except Exception:
                    pass

            # Usually numeric_tail = [atomic_number, mass, valence]
            # If valence is missing, numeric_tail = [atomic_number, mass]
            if len(numeric_tail) >= 3:
                mass = numeric_tail[-2]
            elif len(numeric_tail) >= 2:
                mass = numeric_tail[-1]
            else:
                continue

            if mass > 0:
                masses[atom_type] = mass

    return masses


def compute_system_mass_from_tinker_xyz_prm(xyz_file: Path, prm_file: Path):
    """
    Compute total mass [g/mol] of one simulation cell from Tinker xyz and prm.

    Tinker xyz atom line is usually:
        atom_id  atom_name  x  y  z  atom_type  bonded_atoms...
    """

    atom_masses = parse_atom_masses_from_prm(prm_file)

    if not atom_masses:
        raise RuntimeError(f"No atom masses were parsed from prm file: {prm_file}")

    total_mass = 0.0
    n_atoms_counted = 0
    missing_types = set()

    with xyz_file.open("r", errors="ignore") as f:
        lines = f.readlines()

    for line in lines[1:]:
        tokens = line.split()

        if len(tokens) < 6:
            continue

        # Skip periodic box line:
        # e.g. 48.376 48.376 30.162 90.0 90.0 90.0
        try:
            atom_id = int(tokens[0])
        except Exception:
            continue

        try:
            atom_type = int(tokens[5])
        except Exception:
            continue

        if atom_type in atom_masses:
            total_mass += atom_masses[atom_type]
            n_atoms_counted += 1
        else:
            missing_types.add(atom_type)

    if missing_types:
        print(
            "[WARNING] Missing masses for atom types: "
            + ", ".join(map(str, sorted(missing_types)))
        )

    if n_atoms_counted == 0:
        raise RuntimeError(f"No atoms were counted from xyz file: {xyz_file}")

    return total_mass


def find_reference_file(root: Path, temp_dirs, suffix: str):
    """
    Find reference file such as .xyz or .prm from root or temperature directories.
    """

    candidates = []

    candidates.extend(sorted(root.glob(f"*{suffix}")))

    for d in temp_dirs:
        candidates.extend(sorted(d.glob(f"*{suffix}")))

    if not candidates:
        return None

    # Prefer files directly under root
    for p in candidates:
        if p.parent == root:
            return p

    return candidates[0]


def add_density(df, system_mass_g_mol=None):
    """
    Add density [g/cm3] from system mass [g/mol] and volume [A^3].
    """

    df = df.copy()

    if system_mass_g_mol is None:
        return df

    if "volume_A3" not in df.columns:
        return df

    df["system_mass_g_mol"] = system_mass_g_mol
    df["density_g_cm3"] = (
        system_mass_g_mol
        / (AVOGADRO * df["volume_A3"] * ANG3_TO_CM3)
    )

    return df

# ============================================================
# Parse Tinker log file
# ============================================================

def parse_tinker_log(log_file: Path):
    """
    Parse Tinker dynamic output file.

    The parser is intentionally tolerant because Tinker/Tinker-HP/Tinker-GPU
    output formats can differ slightly.

    It looks for lines containing keywords such as:
        Current Time
        Current Potential
        Current Kinetic
        Total Energy
        Current Temperature
        Current Pressure
        Lattice Lengths
        Lattice Angles
        Periodic Box Dimensions
    """

    rows = []
    current = {}

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

        # --------------------------
        # Time
        # --------------------------
        if "current time" in low and nums:
            # New MD record usually starts here
            flush_current()
            current["time_ps"] = nums[0]
            current["log_file"] = str(log_file)

        # --------------------------
        # Energies
        # --------------------------
        elif "current potential" in low and nums:
            current["potential_kcal_mol"] = nums[0]

        elif "current kinetic" in low and nums:
            current["kinetic_kcal_mol"] = nums[0]

        elif "total energy" in low and nums:
            current["total_energy_kcal_mol"] = nums[0]

        # Some Tinker builds may print only "Potential Energy" etc.
        elif "potential energy" in low and nums:
            current.setdefault("potential_kcal_mol", nums[0])

        elif "kinetic energy" in low and nums:
            current.setdefault("kinetic_kcal_mol", nums[0])

        # --------------------------
        # Temperature and pressure
        # --------------------------
        elif "current temperature" in low and nums:
            current["temperature_K"] = nums[0]

        elif "temperature" in low and "current" not in low and nums:
            # avoid overwriting if a cleaner value already exists
            current.setdefault("temperature_K", nums[0])

        elif "current pressure" in low and nums:
            current["pressure_atm"] = nums[0]

        elif "pressure" in low and "current" not in low and nums:
            current.setdefault("pressure_atm", nums[0])

        # --------------------------
        # Lattice constants from log
        # --------------------------
        elif ("lattice lengths" in low or "cell lengths" in low) and len(nums) >= 3:
            current["a_A"] = nums[0]
            current["b_A"] = nums[1]
            current["c_A"] = nums[2]

        elif ("lattice angles" in low or "cell angles" in low) and len(nums) >= 3:
            current["alpha_deg"] = nums[0]
            current["beta_deg"] = nums[1]
            current["gamma_deg"] = nums[2]

        elif "periodic box dimensions" in low or "box dimensions" in low:
            # Try to read next few lines and find six numbers:
            # a b c alpha beta gamma
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

    # Remove rows without any useful observable
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


# ============================================================
# Parse Tinker snapshot files: *.001, *.002, ...
# ============================================================

def parse_tinker_snapshot_cell(snapshot_file: Path):
    """
    Parse lattice constants from a Tinker snapshot/archive-like file.

    Many Tinker periodic XYZ/archive snapshots have:
        line 1: number of atoms
        line 2: a b c alpha beta gamma

    Returns dict or None.
    """

    try:
        with snapshot_file.open("r", errors="ignore") as f:
            first = f.readline()
            second = f.readline()
    except Exception:
        return None

    nums = extract_floats(second)

    if len(nums) < 6:
        return None

    a, b, c, alpha, beta, gamma = nums[:6]

    # basic sanity check
    if not (a > 0 and b > 0 and c > 0):
        return None

    return {
        "a_A": a,
        "b_A": b,
        "c_A": c,
        "alpha_deg": alpha,
        "beta_deg": beta,
        "gamma_deg": gamma,
    }


def parse_snapshot_index(snapshot_file: Path):
    """
    Extract frame index from extension:
        file.001 -> 1
        file.400 -> 400
    """
    suffix = snapshot_file.suffix
    if re.fullmatch(r"\.\d{3,}", suffix):
        return int(suffix[1:])
    return None


def parse_snapshot_series(temp_dir: Path, save_interval_ps=50.0):
    """
    Parse all snapshot files in a temperature directory.
    """

    rows = []

    candidates = []
    for p in temp_dir.iterdir():
        if not p.is_file():
            continue
        idx = parse_snapshot_index(p)
        if idx is not None:
            candidates.append((idx, p))

    candidates.sort(key=lambda x: x[0])

    for idx, p in candidates:
        cell = parse_tinker_snapshot_cell(p)
        if cell is None:
            continue

        row = {
            "frame": idx,
            "time_ps": idx * save_interval_ps,
            "snapshot_file": str(p),
        }
        row.update(cell)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Merge log and snapshot information
# ============================================================

def merge_log_and_snapshot(log_df, snap_df):
    """
    Merge log-derived data and snapshot-derived lattice constants.

    Important:
    pandas.merge_asof() cannot handle NaN in merge keys.
    Therefore, rows without time_ps are removed before merging.
    """

    if log_df is None or log_df.empty:
        return snap_df.copy() if snap_df is not None else pd.DataFrame()

    if snap_df is None or snap_df.empty:
        return log_df.copy()

    log_df = log_df.copy()
    snap_df = snap_df.copy()

    # If time_ps does not exist, simple concat fallback
    if "time_ps" not in log_df.columns or "time_ps" not in snap_df.columns:
        return pd.concat(
            [
                log_df.reset_index(drop=True),
                snap_df.reset_index(drop=True),
            ],
            axis=1,
        )

    # Convert time_ps to numeric
    log_df["time_ps"] = pd.to_numeric(log_df["time_ps"], errors="coerce")
    snap_df["time_ps"] = pd.to_numeric(snap_df["time_ps"], errors="coerce")

    # Critical fix:
    # merge_asof cannot accept NaN in merge keys.
    n_log_before = len(log_df)
    n_snap_before = len(snap_df)

    log_df = log_df.dropna(subset=["time_ps"])
    snap_df = snap_df.dropna(subset=["time_ps"])

    n_log_after = len(log_df)
    n_snap_after = len(snap_df)

    if n_log_after < n_log_before:
        print(
            f"[WARNING] Removed {n_log_before - n_log_after} log rows "
            f"because time_ps was missing."
        )

    if n_snap_after < n_snap_before:
        print(
            f"[WARNING] Removed {n_snap_before - n_snap_after} snapshot rows "
            f"because time_ps was missing."
        )

    # If one side becomes empty, return the other
    if log_df.empty:
        return snap_df.copy()

    if snap_df.empty:
        return log_df.copy()

    log_df = log_df.sort_values("time_ps")
    snap_df = snap_df.sort_values("time_ps")

    merged = pd.merge_asof(
        log_df,
        snap_df,
        on="time_ps",
        direction="nearest",
        tolerance=None,
        suffixes=("", "_snap"),
    )

    # Fill lattice columns from snapshot if log lacks them
    for col in [
        "a_A",
        "b_A",
        "c_A",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "frame",
        "snapshot_file",
    ]:
        snap_col = f"{col}_snap"

        if snap_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[snap_col])
            else:
                merged[col] = merged[snap_col]

            merged = merged.drop(columns=[snap_col])

    return merged


# ============================================================
# Add cumulative averages
# ============================================================

def add_cumulative_averages(df):
    """
    Add cumulative average columns for selected numeric observables.
    """

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

    for col in target_cols:
        if col in df.columns:
            df[f"{col}_cumavg"] = df[col].expanding(min_periods=1).mean()

    return df


def add_volume(df):
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


# ============================================================
# Analyze one temperature series, including continuation runs
# ============================================================

def extract_temperature_dir(
        temp_dir: Path,
        save_interval_ps=50.0,
        ):
    """
    Extract raw data from one physical run directory.
    This does not write CSV and does not add cumulative averages.
    """

    print(f"[ANALYZE RUN] {temp_dir}")

    # Log files
    log_files = sorted(temp_dir.glob("*.out"))

    # Prefer dynamic log over scheduler .o/.e files
    preferred_logs = [
        p for p in log_files
        if ("log" in p.name.lower() or "dynamic" in p.name.lower())
    ]

    if preferred_logs:
        log_files = preferred_logs

    log_dfs = []
    for log_file in log_files:
        df_log = parse_tinker_log(log_file)
        if not df_log.empty:
            log_dfs.append(df_log)

    if log_dfs:
        log_df = pd.concat(log_dfs, ignore_index=True)
        if "time_ps" in log_df.columns:
            log_df = log_df.sort_values("time_ps")
    else:
        log_df = pd.DataFrame()

    # Snapshot files
    snap_df = parse_snapshot_series(temp_dir, save_interval_ps=save_interval_ps)

    # Merge
    df = merge_log_and_snapshot(log_df, snap_df)

    if df.empty:
        print(f"[WARNING] No analyzable data found in {temp_dir}")
        return df

    if "time_ps" in df.columns:
        df["time_ps"] = pd.to_numeric(df["time_ps"], errors="coerce")
        df = df.dropna(subset=["time_ps"]).sort_values("time_ps")

    df["run_dir"] = temp_dir.name
    df["source_dir"] = str(temp_dir)
    df["run_index"] = run_number_from_path(temp_dir)

    return df


def analyze_temperature_series(
        temp_label: str,
        run_dirs,
        out_dir: Path,
        save_interval_ps=50.0,
        force=False,
        system_mass_g_mol=None,
        join_time=True,
        ):
    """
    Analyze one temperature as a single trajectory.

    run_dirs can include:
        T300K, T300K_run_002, T300K_run_003, ...

    The local time in each continuation directory is stored as time_ps_local.
    The stitched continuous time is stored as time_ps.
    """

    temp_K = infer_temp_from_path(Path(temp_label))
    csv_file = out_dir / f"{temp_label}_timeseries.csv"

    if csv_file.exists() and not force:
        print(f"[READ CSV] {csv_file}")
        df = pd.read_csv(csv_file)
        return df

    print(f"[ANALYZE SERIES] {temp_label}")
    print("  run directories:")
    for d in run_dirs:
        print(f"    - {d}")

    dfs = []
    current_max_time = None

    for irun, run_dir in enumerate(sorted(run_dirs, key=sort_key_for_run_dir), start=1):
        if not run_dir.exists():
            print(f"[WARNING] Directory not found: {run_dir}")
            continue

        df = extract_temperature_dir(
            temp_dir=run_dir,
            save_interval_ps=save_interval_ps,
        )

        if df.empty:
            continue

        df = df.copy()
        df["temperature_dir"] = temp_label
        df["target_temperature_K"] = temp_K
        df["continuation_order"] = irun

        if "time_ps" in df.columns:
            df = df.sort_values("time_ps")
            df["time_ps_local"] = df["time_ps"]

            if join_time:
                local_min = df["time_ps_local"].min()

                if current_max_time is None:
                    offset = 0.0
                else:
                    local_dt = estimate_time_step_ps(
                        df["time_ps_local"],
                        fallback=save_interval_ps,
                    )

                    # Case 1: restart time reset or overlaps with previous run.
                    # Shift the first local frame just after the previous maximum time.
                    if local_min <= current_max_time + 1.0e-9:
                        offset = current_max_time + local_dt - local_min

                    # Case 2: Tinker log already kept global time.
                    # Do not shift.
                    else:
                        offset = 0.0

                df["time_offset_ps"] = offset
                df["time_ps"] = df["time_ps_local"] + offset
                current_max_time = df["time_ps"].max()
            else:
                df["time_offset_ps"] = 0.0
                current_max_time = df["time_ps"].max()

        dfs.append(df)

    if not dfs:
        print(f"[WARNING] No analyzable data found in series {temp_label}")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    if "time_ps" in df.columns:
        df = df.sort_values(["time_ps", "continuation_order"])

    df = add_volume(df)
    df = add_density(df, system_mass_g_mol=system_mass_g_mol)
    df = add_cumulative_averages(df)

    # Put important columns first
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
        "source_dir",
    ]

    cols = [c for c in first_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    df.to_csv(csv_file, index=False)
    print(f"[WRITE] {csv_file}")

    return df


# Backward-compatible wrapper for scripts that used analyze_temperature_dir()
def analyze_temperature_dir(
        temp_dir: Path,
        out_dir: Path,
        save_interval_ps=50.0,
        force=False,
        system_mass_g_mol=None,
        ):
    return analyze_temperature_series(
        temp_label=temp_dir.name,
        run_dirs=[temp_dir],
        out_dir=out_dir,
        save_interval_ps=save_interval_ps,
        force=force,
        system_mass_g_mol=system_mass_g_mol,
        join_time=False,
    )


# ============================================================
# Plotting
# ============================================================

def plot_property(all_df, out_dir: Path, ycol: str, ylabel: str):
    if ycol not in all_df.columns:
        return

    plot_df = all_df.dropna(subset=[ycol])
    if plot_df.empty:
        return

    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    # Raw time series
    plt.figure(figsize=(7, 4.5))

    for label, sub in plot_df.groupby("temperature_dir"):
        sub = sub.sort_values("time_ps")
        if "time_ps" in sub.columns:
            plt.plot(sub["time_ps"], sub[ycol], label=label)
        else:
            plt.plot(np.arange(len(sub)), sub[ycol], label=label)

    plt.xlabel("Time / ps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    fig_file = png_dir / f"{ycol}.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"[WRITE] {fig_file}")

    # Cumulative average
    cum_col = f"{ycol}_cumavg"
    if cum_col not in all_df.columns:
        return

    cum_df = all_df.dropna(subset=[cum_col])
    if cum_df.empty:
        return

    plt.figure(figsize=(7, 4.5))

    for label, sub in cum_df.groupby("temperature_dir"):
        sub = sub.sort_values("time_ps")
        if "time_ps" in sub.columns:
            plt.plot(sub["time_ps"], sub[cum_col], label=label)
        else:
            plt.plot(np.arange(len(sub)), sub[cum_col], label=label)

    plt.xlabel("Time / ps")
    plt.ylabel(f"Cumulative average of {ylabel}")
    plt.legend()
    plt.tight_layout()

    fig_file = png_dir / f"{cum_col}.png"
    plt.savefig(fig_file, dpi=300)
    plt.close()
    print(f"[WRITE] {fig_file}")


def make_plots(all_df, out_dir: Path):
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

    for col, label in plot_targets.items():
        plot_property(all_df, out_dir, col, label)


# ============================================================
# Summary
# ============================================================

def make_summary(all_df):
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

    rows = []

    for temp_label, sub in all_df.groupby("temperature_dir"):
        row = {
            "temperature_dir": temp_label,
            "target_temperature_K": sub["target_temperature_K"].iloc[0]
            if "target_temperature_K" in sub.columns else np.nan,
            "n_points": len(sub),
            "time_min_ps": sub["time_ps"].min() if "time_ps" in sub.columns else np.nan,
            "time_max_ps": sub["time_ps"].max() if "time_ps" in sub.columns else np.nan,
        }

        for col in summary_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_std"] = sub[col].std()
                row[f"{col}_last_cumavg"] = (
                    sub[f"{col}_cumavg"].dropna().iloc[-1]
                    if f"{col}_cumavg" in sub.columns and not sub[f"{col}_cumavg"].dropna().empty
                    else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("target_temperature_K")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Tinker-GPU/Tinker-HP NPT outputs in TxxxK directories, including continuation runs."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing T250K, T300K, T300K_run_002, ...",
    )

    parser.add_argument(
        "--temps",
        nargs="*",
        default=None,
        help=(
            "Temperature labels/directories to analyze, e.g. T250K T300K. "
            "With continuation joining enabled, T300K also includes T300K_run_002, T300K_run_003, ..."
        ),
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="analysis_result",
        help="Output directory.",
    )

    parser.add_argument(
        "--save-interval-ps",
        type=float,
        default=50.0,
        help="Time interval between Tinker snapshot files *.001, *.002, ... in ps. Default: 50 ps.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis even if CSV files already exist.",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate PNG plots.",
    )

    parser.add_argument(
        "--mass-g-mol",
        type=float,
        default=None,
        help="Total mass of one simulation cell in g/mol. If omitted, the script tries to compute it from xyz and prm.",
    )

    parser.add_argument(
        "--xyz-for-mass",
        type=str,
        default=None,
        help="Reference Tinker xyz file used to compute total system mass.",
    )

    parser.add_argument(
        "--prm-for-mass",
        type=str,
        default=None,
        help="Reference Tinker prm file used to compute total system mass.",
    )

    parser.add_argument(
        "--no-join-continuations",
        action="store_true",
        help=(
            "Do not stitch continuation directories. "
            "By default, T300K, T300K_run_002, ... are treated as one T300K trajectory."
        ),
    )

    parser.add_argument(
        "--no-time-stitch",
        action="store_true",
        help=(
            "Do not shift local time axes of continuation runs. "
            "Useful only if you want to inspect raw restart times."
        ),
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    join_continuations = not args.no_join_continuations
    join_time = not args.no_time_stitch

    temp_groups = discover_temperature_groups(
        root=root,
        temps=args.temps,
        join_continuations=join_continuations,
    )

    if not temp_groups:
        raise RuntimeError(f"No temperature directories found in {root}")

    # Flatten directories for mass-file discovery.
    temp_dirs_flat = []
    for _, dirs in temp_groups:
        temp_dirs_flat.extend(dirs)

    # ========================================================
    # Determine total system mass for density calculation
    # ========================================================

    system_mass_g_mol = args.mass_g_mol

    if system_mass_g_mol is not None:
        print(f"[INFO] Use manually specified system mass: {system_mass_g_mol:.8f} g/mol")
    else:
        if args.xyz_for_mass is not None:
            xyz_for_mass = Path(args.xyz_for_mass).resolve()
        else:
            xyz_for_mass = find_reference_file(root, temp_dirs_flat, ".xyz")

        if args.prm_for_mass is not None:
            prm_for_mass = Path(args.prm_for_mass).resolve()
        else:
            prm_for_mass = find_reference_file(root, temp_dirs_flat, ".prm")

        if xyz_for_mass is not None and prm_for_mass is not None:
            try:
                system_mass_g_mol = compute_system_mass_from_tinker_xyz_prm(
                    xyz_for_mass,
                    prm_for_mass,
                )
                print(f"[INFO] Mass reference xyz: {xyz_for_mass}")
                print(f"[INFO] Mass reference prm: {prm_for_mass}")
                print(f"[INFO] Computed system mass: {system_mass_g_mol:.8f} g/mol")
            except Exception as e:
                print("[WARNING] Failed to compute system mass from xyz/prm.")
                print(f"[WARNING] {e}")
                print("[WARNING] Density will not be calculated.")
                system_mass_g_mol = None
        else:
            print("[WARNING] xyz/prm reference files were not found.")
            print("[WARNING] Density will not be calculated.")

    print("Temperature series:")
    for label, dirs in temp_groups:
        joined = " + ".join(d.name for d in dirs)
        print(f"  - {label}: {joined}")

    dfs = []

    for temp_label, run_dirs in temp_groups:
        df = analyze_temperature_series(
            temp_label=temp_label,
            run_dirs=run_dirs,
            out_dir=out_dir,
            save_interval_ps=args.save_interval_ps,
            force=args.force,
            system_mass_g_mol=system_mass_g_mol,
            join_time=join_time,
        )

        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No data were extracted.")

    all_df = pd.concat(dfs, ignore_index=True)

    if "target_temperature_K" in all_df.columns and "time_ps" in all_df.columns:
        all_df = all_df.sort_values(["target_temperature_K", "time_ps"])

    all_csv = out_dir / "all_temperatures_timeseries.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"[WRITE] {all_csv}")

    summary_df = make_summary(all_df)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[WRITE] {summary_csv}")

    if not args.no_plot:
        make_plots(all_df, out_dir)

    print("")
    print("Done.")
    print(f"Results are saved in: {out_dir}")


if __name__ == "__main__":
    main()
