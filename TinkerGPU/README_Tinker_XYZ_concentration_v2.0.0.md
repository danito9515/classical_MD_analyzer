# Tinker XYZを主体としたNPT密度・濃度解析スクリプト v2.0.0

## 1. 目的

`analyze_TinkerGpu_npt_joinruns_xyz_concentration_v2.0.0.py`は、Tinker-GPU/Tinker-HPのNPT計算で生成されたTinker XYZ snapshotを主データとして、以下を同じ時間軸で解析します。

- potential / kinetic / total energy
- temperature
- pressure
- 格子定数とセル体積
- density
- 指定したatom typeの濃度
- 指定したatom typeを含む分子の濃度
- 上記の各時系列と累積平均
- `T300K + T300K_run_002 + ...`の継続run結合

PDBファイルは不要です。

通常のTinker snapshot、すなわち

```text
system.001
system.002
system.003
...
```

を完全なTinker XYZ frameとして読み取ります。multi-frameの`.arc`にも対応します。

---

## 2. Tinker XYZから読み取る情報

Tinker XYZの原子行は、通常、次の形式です。

```text
atom_id  atom_name  x  y  z  atom_type  bonded_atom_ids...
```

スクリプトは次を使用します。

- 6列目：Tinker atom type
- 7列目以降：結合相手のatom ID
- atom count行の次の行：`a b c alpha beta gamma`

分子は、7列目以降の結合情報から構築した結合グラフの**連結成分**として定義します。

したがって、FSA中に同じtypeのS原子が2個あっても、`molecule mode`ではFSA分子を1個として数えます。

---

## 3. 濃度の定義

各frameの濃度は、XYZ snapshotに記録された瞬間セル体積を用いて

```text
c(t) = N_entity / [N_A × V(t) × 10^-27]
```

として計算します。

- `N_entity`：選択された原子数または分子数
- `V(t)`：セル体積 / Å³
- `c(t)`：mol dm⁻³

通常の固定組成NPT-MDでは粒子数は一定です。そのため濃度の時間変化は、NPTセル体積の時間変化から生じます。

---

## 4. 推奨実行例：LiFSA/SN系

```bash
python3 ../../analyze_TinkerGpu_npt_joinruns_xyz_concentration_v2.0.0.py \
  --root ./ \
  --temps T300K T330K T350K \
  --save-interval-ps 50 \
  --conc-atom-type LiFSA:6 \
  --conc-molecule-type FSA:164 \
  --conc-molecule-type SN:71 \
  --list-xyz-types \
  --force
```

### 選択の意味

```bash
--conc-atom-type LiFSA:6
```

atom type 6のLiを直接数えます。LiFSA formula unitあたりLiが1個なら、Li数はLiFSA formula-unit数と一致します。

```bash
--conc-molecule-type FSA:164
```

atom type 164を含む結合連結成分を数えます。FSA中にtype 164が複数存在しても、1分子を1回だけ数えます。

```bash
--conc-molecule-type SN:71
```

atom type 71を含むSN分子の連結成分を数えます。SN中にtype 71が2個あっても二重計数しません。

---

## 5. snapshot basenameを指定する場合

同じ温度dir内に複数の`.001/.002/...`系列がある場合は、使用するbasenameを明示してください。

```bash
--snapshot-prefix LiFSA_SN_1_10_N90_Table2rho_retyped_Eq_T330K_np1
```

例えば、対象ファイルが

```text
LiFSA_SN_1_10_N90_Table2rho_retyped_Eq_T330K_np1.001
LiFSA_SN_1_10_N90_Table2rho_retyped_Eq_T330K_np1.002
```

なら、`.001`より前の部分を指定します。

未指定時に複数系列が見つかった場合、frame数が最も多い系列を自動選択し、warningを表示します。

---

## 6. 結合情報を持つ初期XYZを指定する場合

一部のtrajectory XYZ/ARCが結合列を省略している場合は、完全なTinker XYZをtopologyとして指定します。

```bash
--topology-xyz LiFSA_SN_1_10_N90_Table2rho_retyped.xyz
```

分子濃度を求める場合、結合情報が必要です。結合が記録されていないframeを使う場合は、このoptionを使用してください。

各frameのatom ID、atom type、bond listまで厳密に検査するには、次も追加します。

```bash
--strict-topology
```

大規模trajectoryでは検査コストが増えるため、通常はatom数のみ検査し、必要な場合に有効化します。

---

## 7. multi-frame ARCを解析する場合

TxxxK dirに`.001/.002/...`がなく、`.arc`がある場合は自動的に`.arc`を解析します。

明示する場合：

```bash
python3 ../../analyze_TinkerGpu_npt_joinruns_xyz_concentration_v2.0.0.py \
  --root ./ \
  --temps T330K \
  --xyz-pattern '*.arc' \
  --save-interval-ps 50 \
  --conc-atom-type LiFSA:6 \
  --conc-molecule-type FSA:164 \
  --conc-molecule-type SN:71 \
  --force
```

multi-frameの`.xyz`なら、次のように指定します。

```bash
--xyz-pattern '*.xyz'
```

`--xyz-pattern`は、numbered snapshotが見つからない場合だけ使用されます。

---

## 8. frame時刻

### Numbered snapshot

通常、`.001`を1回目の保存frameとして扱い、

```text
.001 → 50 ps
.002 → 100 ps
.003 → 150 ps
```

のように`--save-interval-ps`から時刻を割り当てます。

`.001`を0 psとしたい場合：

```bash
--first-frame-time-ps 0
```

### Continuation run

```text
T300K/
T300K_run_002/
T300K_run_003/
```

は自動的に1本の時間軸へ結合されます。各restartで`.001`に戻っていても、前runの最終時刻の次に配置します。

結合したくない場合：

```bash
--no-join-continuations
```

時刻shiftだけ止める場合：

```bash
--no-time-stitch
```

---

## 9. 密度計算

密度は

```text
rho(t) = M_cell / [N_A × V(t) × 10^-24]
```

で計算します。

- `M_cell`：simulation cell 1個分の質量 / g mol⁻¹
- `V(t)`：Å³
- `rho(t)`：g cm⁻³

スクリプトはrootまたは温度dirから`.xyz`と`.prm`を探し、atom typeとPRM massからcell massを求めます。

明示する場合：

```bash
--xyz-for-mass LiFSA_SN_1_10_N90_Table2rho_retyped.xyz \
--prm-for-mass LiFSAC2_poltype2.prm
```

cell massを直接指定することもできます。

```bash
--mass-g-mol 12345.6789
```

---

## 10. 主な出力

```text
analysis_result/
├── T300K_timeseries.csv
├── T300K_concentration_timeseries.csv
├── T300K_xyz_atom_type_inventory.csv
├── all_temperatures_timeseries.csv
├── all_concentrations_timeseries.csv
├── all_xyz_atom_type_inventory.csv
├── concentration_selection_metadata.csv
├── summary.csv
└── png/
    ├── density_g_cm3.png
    ├── density_g_cm3_cumavg.png
    ├── concentration_LiFSA_mol_dm3.png
    ├── concentration_LiFSA_mol_dm3_cumavg.png
    ├── concentration_FSA_mol_dm3.png
    ├── concentration_FSA_mol_dm3_cumavg.png
    ├── concentration_SN_mol_dm3.png
    └── concentration_SN_mol_dm3_cumavg.png
```

`T300K_timeseries.csv`はXYZ snapshot 1 frameにつき1行です。logのenergy、temperature、pressureは最近傍時刻から付加されます。

XYZとlogの許容時刻差は、デフォルトで`save interval / 2`です。変更する場合：

```bash
--merge-tolerance-ps 30
```

---

## 11. atom typeの確認

```bash
--list-xyz-types
```

を付けると、first frameから次を表示・保存します。

- atom type
- atom count
- atom name
- そのtypeを含むconnected component数

これにより、`--conc-atom-type`または`--conc-molecule-type`へ渡すtypeを確認できます。

---

## 12. 必要なPython package

```bash
python3 -m pip install numpy pandas matplotlib
```

---

## 13. 実装上の重要点

1. PDBのresidue IDには依存しません。
2. Tinker XYZのatom typeとbond connectivityを直接使用します。
3. 密度と濃度は同一のXYZ frame体積から計算します。
4. molecule modeでは同一分子中の同じatom typeを二重計数しません。
5. continuation run間で選択entity数が変わった場合はerrorにします。
6. molecule modeで結合情報が全くない場合は、誤計数を避けるためデフォルトで停止します。
