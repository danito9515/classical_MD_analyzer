#!/usr/bin/env python3
import re
import argparse
import pandas as pd

R = 0.00198720425864083  # kcal/mol/K

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--natoms", type=int, required=True)
    parser.add_argument("--dof", type=int, default=None)
    parser.add_argument("--out", default="temperature_from_tinker.csv")
    args = parser.parse_args()

    dof = args.dof if args.dof is not None else 3 * args.natoms - 3

    time_re = re.compile(r"Current Time\s+([0-9Ee+\-.]+)\s+Picosecond")
    kin_re  = re.compile(r"Current Kinetic\s+([0-9Ee+\-.]+)\s+Kcal/mole")

    rows = []
    current_time = None

    with open(args.logfile) as f:
        for line in f:
            m = time_re.search(line)
            if m:
                current_time = float(m.group(1))

            m = kin_re.search(line)
            if m:
                kinetic = float(m.group(1))
                temp = 2.0 * kinetic / (dof * R)
                rows.append({
                    "time_ps": current_time,
                    "kinetic_kcal_mol": kinetic,
                    "temperature_K": temp,
                    "dof_used": dof,
                })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out}")
    print(df.head())

if __name__ == "__main__":
    main()
