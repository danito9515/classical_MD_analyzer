import pandas as pd
import numpy as np

csv_file = "1_10_nvt300K_transference_block_estimates_solvent_fixed.csv"

df = pd.read_csv(csv_file)

columns = [
    "PFG_t_app",
    "eNMR_t0",
    "Bruce_Vincent_tss",
]

for col in columns:
    values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

    n = len(values)
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    sem = sd / np.sqrt(n)
    ci95 = 1.96 * sem

    print(f"{col}")
    print(f"  N       = {n}")
    print(f"  mean    = {mean:.6f}")
    print(f"  SD      = {sd:.6f}")
    print(f"  SEM     = {sem:.6f}")
    print(f"  95% CI  = ±{ci95:.6f}")
    print(f"  result  = {mean:.6f} ± {ci95:.6f}")
