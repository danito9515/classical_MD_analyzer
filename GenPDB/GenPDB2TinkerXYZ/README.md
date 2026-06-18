# LiFSA/SN liquid initial structures for Tinker GPU using Packmol

This directory contains a reproducible protocol for constructing LiFSA/succinonitrile (SN) liquid structures matching the experimental stoichiometry, density, and LiFSA concentration reported in Table 2 of Ugata et al., *Phys. Chem. Chem. Phys.* 2019 for LiFSA/SN mixtures.

The key lesson from testing is simple: **use Packmol for the initial molecular packing**. Direct deletion/addition or home-made rigid repacking easily creates hidden molecular overlaps or broken periodic molecules, which produces huge Tinker van der Waals or bonded energies during `testgrad`/`minimize`. Packmol gives much more robust initial configurations.

---

## 1. Target systems

The default small systems are chosen to keep the atom count below about 10,000 while matching the Table 2 density and LiFSA concentration up to integer molecule-count rounding.

| system key | ratio [LiFSA]/[SN] | Li | FSA | SN | atoms | density / g cm⁻³ | cLi / M | cubic L / Å |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1_6_N142` | 1/6 | 142 | 142 | 852 | 9940 | 1.186 | 1.776 | 51.010 |
| `1_10_N90` | 1/10 | 90 | 90 | 900 | 9900 | 1.121 | 1.135 | 50.880 |
| `1_0p8_N191` | 1/0.8 | 191 | 191 | 153 | 3440 | 1.627 | 6.476 | 34.305 |

The formula used for the box size is:

```text
mass_g = (N_LiFSA * M_LiFSA + N_SN * M_SN) / N_A
volume_A3 = mass_g / rho * 1.0e24
L_A = volume_A3 ** (1/3)
```

with:

```text
M_LiFSA = 187.07 g/mol
M_SN    = 80.09  g/mol
```

---

## 2. Directory contents

```text
LiFSA_SN_packmol_Tinker_protocol/
├── README.md
├── run_example_1_6.sh
├── scripts/
│   ├── make_packmol_lifsa_sn_table2.py
│   ├── retag_lifsa_sn_xyz_to_LiFSAC2_prm.py
│   └── check_tinker_xyz_overlaps.py
├── templates/
│   ├── FSA_final.xyz
│   ├── SN_final.xyz
│   ├── FSA_final.key
│   └── SN_final.key
└── params/
    └── LiFSAC2_poltype2.prm
```

### Main scripts

`make_packmol_lifsa_sn_table2.py`

Creates Packmol template PDBs from the single-molecule Tinker xyz files, writes Packmol input files, and converts Packmol PDB output back to Tinker xyz while preserving connectivity.

`retag_lifsa_sn_xyz_to_LiFSAC2_prm.py`

Retags atom types in the generated Tinker xyz so they match `LiFSAC2_poltype2.prm`.

`check_tinker_xyz_overlaps.py`

Checks short nonbonded contacts before Tinker minimization.

---

## 3. Recommended workflow

Copy the protocol directory to the working directory.

```bash
cp -r LiFSA_SN_packmol_Tinker_protocol work_lifsa_sn
cd work_lifsa_sn
```

### Step 1: Create Packmol inputs

For the default 1/6 and 1/10 systems:

```bash
python3 scripts/make_packmol_lifsa_sn_table2.py setup \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2 \
  --systems 1_6_N142 1_10_N90
```

To also create the 1/0.8 system:

```bash
python3 scripts/make_packmol_lifsa_sn_table2.py setup \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2 \
  --systems 1_0p8_N191 1_6_N142 1_10_N90
```

This creates directories such as:

```text
packmol_table2/LiFSA_SN_1_6_N142/packmol.inp
packmol_table2/LiFSA_SN_1_10_N90/packmol.inp
```

and template PDB files under:

```text
packmol_table2/templates/
```

### Step 2: Run Packmol

For 1/6:

```bash
cd packmol_table2/LiFSA_SN_1_6_N142
packmol < packmol.inp
cd ../..
```

For 1/10:

```bash
cd packmol_table2/LiFSA_SN_1_10_N90
packmol < packmol.inp
cd ../..
```

Packmol writes:

```text
packmol_output.pdb
```

### Step 3: Convert Packmol PDB to Tinker xyz

For 1/6:

```bash
python3 scripts/make_packmol_lifsa_sn_table2.py convert \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2 \
  --system 1_6_N142
```

For 1/10:

```bash
python3 scripts/make_packmol_lifsa_sn_table2.py convert \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2 \
  --system 1_10_N90
```

This gives:

```text
packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho.xyz
packmol_table2/LiFSA_SN_1_10_N90/LiFSA_SN_1_10_N90_Table2rho.xyz
```

### Step 4: Retag atom types for the current prm

The Packmol-converted xyz preserves connectivity, but it may inherit template atom types that do not match the current `LiFSAC2_poltype2.prm`. Therefore, retag before Tinker.

For 1/6:

```bash
python3 scripts/retag_lifsa_sn_xyz_to_LiFSAC2_prm.py \
  packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho.xyz \
  packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho_retyped.xyz
```

For 1/10:

```bash
python3 scripts/retag_lifsa_sn_xyz_to_LiFSAC2_prm.py \
  packmol_table2/LiFSA_SN_1_10_N90/LiFSA_SN_1_10_N90_Table2rho.xyz \
  packmol_table2/LiFSA_SN_1_10_N90/LiFSA_SN_1_10_N90_Table2rho_retyped.xyz
```

The retagging rules are:

```text
Li+ : 6

FSA:
  O : 81
  S : 164
  F : 91
  N : 72

SN:
  nitrile N : 71
  nitrile C : 65
  CH2 C     : 66
  H         : 13
```

### Step 5: Check overlaps

For 1/6:

```bash
python3 scripts/check_tinker_xyz_overlaps.py \
  packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho_retyped.xyz \
  --box 51.010376 \
  --cutoff 1.5 \
  --top 50
```

For 1/10:

```bash
python3 scripts/check_tinker_xyz_overlaps.py \
  packmol_table2/LiFSA_SN_1_10_N90/LiFSA_SN_1_10_N90_Table2rho_retyped.xyz \
  --box 50.8799 \
  --cutoff 1.5 \
  --top 50
```

A good initial structure should have very few or zero suspicious contacts below 1.5 Å.

### Step 6: Tinker check and pre-equilibration

Example:

```bash
testgrad packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho_retyped.xyz \
  -k nvt_optFF_gpu.key
```

Recommended MD preparation:

```text
1. testgrad
2. minimize / minimize9
3. short NVT with small timestep or conservative settings
4. NPT to relax density
5. production NVT/NVE
```

For AMOEBA/polarizable systems, if minimization still fails, use a looser Packmol box first and then compress by NPT.

---

## 4. Optional: loose initial box

You can generate a lower-density initial box to avoid minimization trouble:

```bash
python3 scripts/make_packmol_lifsa_sn_table2.py setup \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2_loose \
  --systems 1_6_N142 \
  --loose-factor 1.05
```

After Packmol, convert using the loose box length:

```bash
python3 scripts/make_packmol_lifsa_sn_table2.py convert \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2_loose \
  --system 1_6_N142 \
  --use-packmol-box
```

Then retag and run short NVT/NPT. This is safer if direct Table 2 density still gives large initial energy.

---

## 5. What each script does internally

### `setup` mode

`make_packmol_lifsa_sn_table2.py setup` does four things:

1. Reads `FSA_final.xyz` and `SN_final.xyz`.
2. Centers each molecule and writes Packmol-compatible PDB templates.
3. Computes the cubic box length from the target density and molecule counts.
4. Writes `packmol.inp` for each selected system.

### `convert` mode

`make_packmol_lifsa_sn_table2.py convert` does four things:

1. Reads `packmol_output.pdb` coordinates.
2. Assumes the Packmol order is Li → FSA → SN.
3. Rebuilds a Tinker xyz using atom names, atom types, and bonds from `FSA_final.xyz` and `SN_final.xyz`.
4. Writes the Tinker periodic cell line.

### `retag` script

`retag_lifsa_sn_xyz_to_LiFSAC2_prm.py` identifies connected components:

```text
Li+ = 1 isolated Li atom
FSA = 9-atom molecule: S2 N1 O4 F2
SN  = 10-atom molecule: C4 H4 N2
```

Then it overwrites atom types to match `LiFSAC2_poltype2.prm`.

### `check_tinker_xyz_overlaps.py`

This script reads the Tinker xyz, builds molecular components from connectivity, and reports short **intermolecular** contacts under a cutoff. It is mainly a pre-minimization sanity check.

---

## 6. Notes and pitfalls

- Packmol output is only coordinates. The Tinker xyz needs connectivity and atom types, so direct PDB → xyz conversion is not enough.
- Always retag atom types if the prm has changed.
- The original failed structures had very large van der Waals energies because of hidden short contacts.
- The repacking scripts developed earlier are not recommended as the primary protocol. They are useful for debugging, but Packmol is more reliable.
- Keep the final `*.xyz`, `*.key`, and `LiFSAC2_poltype2.prm` together in the same run directory.

---

## 7. Minimal one-system example for 1/6

```bash
python3 scripts/make_packmol_lifsa_sn_table2.py setup \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2 \
  --systems 1_6_N142

cd packmol_table2/LiFSA_SN_1_6_N142
packmol < packmol.inp
cd ../..

python3 scripts/make_packmol_lifsa_sn_table2.py convert \
  --fsa-xyz templates/FSA_final.xyz \
  --sn-xyz templates/SN_final.xyz \
  --outroot packmol_table2 \
  --system 1_6_N142

python3 scripts/retag_lifsa_sn_xyz_to_LiFSAC2_prm.py \
  packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho.xyz \
  packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho_retyped.xyz

python3 scripts/check_tinker_xyz_overlaps.py \
  packmol_table2/LiFSA_SN_1_6_N142/LiFSA_SN_1_6_N142_Table2rho_retyped.xyz \
  --box 51.010376 \
  --cutoff 1.5 \
  --top 50
```
