# PDBトラジェクトリーを用いたOnsager輸送解析スクリプト 使用説明書

対象スクリプト：

1. `onsager_distinct_transport_pdb_v1_4_0.py`  
   Onsager式に基づく自己項・distinct項・全伝導度の分解解析
2. `onsager_transference_compare_pdb_v1_5_2.py`  
   Onsager式に基づくPFG-NMR型、eNMR型、Bruce–Vincent型カチオン輸率の比較解析

本説明書は、上記2本の**現在の実装内容**に基づく。一般的な輸送理論の全定義を網羅するものではなく、各スクリプトが実際に計算・出力する量を再現可能な形で説明する。

---

## 1. スクリプトの役割

### 1.1 `onsager_distinct_transport_pdb_v1_4_0.py`

このスクリプトは、PDB形式のMDトラジェクトリーから、イオン伝導度を次の項に分解する。

- 各荷電種のself項
- 同種粒子間のdistinct項
- 異種粒子間のdistinct項
- 全電荷変位から直接計算したtotal項

主な用途は以下である。

- Li–Li、FSA–FSA、Li–FSA相関の分離
- Nernst–Einstein近似からのずれの起源の解析
- distinct相関が伝導度を増加・減少させるかの判定
- 複数温度の伝導度比較
- `log10(σT)`–`1000/T` Arrhenius解析
- block averagingによるMSD曲線の統計誤差表示

### 1.2 `onsager_transference_compare_pdb_v1_5_2.py`

このスクリプトは、上記のOnsager伝導度分解に加えて、二元1:1電解質を対象として次のカチオン輸率を比較する。

- PFG-NMR型 apparent transference number
- eNMR型の電流分率
- Bruce–Vincent型 steady-state transference number

さらに、次の基準系を比較できる。

- `barycentric`：推奨設定では全系の質量中心ドリフトを除去した基準系
- `solvent_fixed`：中性溶媒分子の平均COMを固定した基準系

中性溶媒については、分子自己MSDと全溶媒平均COM-MSDを同時に解析し、溶媒基準系の安定性を診断する。

---

## 2. 必要な環境

### 2.1 Pythonパッケージ

必須：

```bash
python3
numpy
matplotlib
```

確認例：

```bash
python3 -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"
```

### 2.2 構文確認

```bash
python3 -m py_compile onsager_distinct_transport_pdb_v1_4_0.py
python3 -m py_compile onsager_transference_compare_pdb_v1_5_2.py
```

---

## 3. 入力PDBトラジェクトリーの前提

### 3.1 座標とセル

- 原子座標の単位は Å とする。
- 各フレームに有効な `CRYST1` 情報が必要である。
- 伝導度計算にはセル体積が必要なため、`CRYST1` がない場合は停止する。
- 現在のunwrap処理はセル長 `a, b, c` のみを使用する。
- **実質的に直方体・直交セルを対象とする。**
- triclinicセルの角度を考慮したminimum-image unwrapは実装されていない。

### 3.2 フレーム形式

以下のいずれかを読み込める。

1. `MODEL` / `ENDMDL` を含むmulti-model PDB
2. 同一原子数のPDBフレームを連結した形式

全フレームで以下が一定である必要がある。

- 原子数
- 原子順序
- 分子・residueの対応

### 3.3 周期境界条件のunwrap

デフォルトでは各原子のフレーム間変位にminimum-image処理を適用してunwrapする。

```text
Δr ← Δr − L round(Δr/L)
```

`--no-unwrap` を指定するとunwrapを無効化する。通常の拡散・伝導度解析では無効化しない。

> **注意**  
> 現在の実装は固定セルNVTトラジェクトリーに最も適している。NPTでセルが大きく変形する場合、fractional-coordinateベースの厳密な補正ではないため、直接適用には注意が必要である。

### 3.4 Tinker `xyzpdb` のatom type

Tinkerの変換PDBでは、通常のresidue name欄にatom type番号が書かれる場合がある。本解析例では以下を用いる。

```bash
--atom-type-field resname
```

例：

- FSAのN marker atom type：72
- SNのN marker atom type：71

利用可能な読取欄：

```text
auto
occupancy
bfactor
resname
tail
last_int
none
```

まずトポロジーを確認することを推奨する。

```bash
python3 onsager_distinct_transport_pdb_v1_4_0.py \
  --pdb 1_10_nvt300K.pdb \
  --dt-ns 0.2 \
  --fit-start-ns 1 \
  --fit-end-ns 10 \
  --max-lag-ns 10 \
  --atom-type-field resname \
  --track-element Li:Li:+1 \
  --track-atom-type FSA_N:72:-1 \
  --list-topology \
  --only-list-topology
```

出力される主な情報：

```text
[elements] {...}
[resnames] {...}
[atom types] {...}
```

---

## 4. Onsager–Einstein表現とself/distinct分解

荷電種を `a`, `b` とし、粒子変位を

```math
\Delta \mathbf r_{a,i}(t)
```

とする。集合変位相関を

```math
C_{ab}(t)
=
\left\langle
\left[\sum_{i\in a}\Delta\mathbf r_{a,i}(t)\right]
\cdot
\left[\sum_{j\in b}\Delta\mathbf r_{b,j}(t)\right]
\right\rangle
```

と定義する。

### 4.1 同種粒子

```math
C_{aa}(t)=C_{aa}^{self}(t)+C_{aa}^{distinct}(t)
```

self項：

```math
C_{aa}^{self}(t)
=
\sum_i\left\langle|\Delta\mathbf r_{a,i}(t)|^2\right\rangle
=
N_a\,MSD_{tr,a}(t)
```

distinct項：

```math
C_{aa}^{distinct}(t)
=
\sum_{i\ne j}
\left\langle
\Delta\mathbf r_{a,i}(t)\cdot\Delta\mathbf r_{a,j}(t)
\right\rangle
```

### 4.2 異種粒子

```math
C_{ab}(t)
=
\sum_{i\in a}\sum_{j\in b}
\left\langle
\Delta\mathbf r_{a,i}(t)\cdot\Delta\mathbf r_{b,j}(t)
\right\rangle
```

異種項は全伝導度中で `ab` と `ba` の2項を含むため、スクリプトではunordered pairに対してmultiplicity 2を適用する。

### 4.3 伝導度への変換

空間次元を `d` とすると、各相関の長時間傾きから

```math
\sigma_{ab}
=
\frac{e^2 z_a z_b}{2dVk_BT}
\frac{dC_{ab}(t)}{dt}
```

を計算する。

3次元では `2d = 6` である。

### 4.4 直接total項

全電荷変位を

```math
\Delta\mathbf Q(t)
=
\sum_a z_a\sum_{i\in a}\Delta\mathbf r_{a,i}(t)
```

として、

```math
C_Q(t)=\langle|\Delta\mathbf Q(t)|^2\rangle
```

からtotal conductivityを直接計算する。

スクリプトは次の一致を確認する。

```text
[total] decomposed = ...
[total] direct     = ...
[check] difference = ...
```

`difference` が丸め誤差程度であれば、self/distinct分解と直接total項の内部整合性が確認できる。

---

# Part I. distinct term解析

## 5. `onsager_distinct_transport_pdb_v1_4_0.py`

### 5.1 今回の実行例

```bash
python3 onsager_distinct_transport_pdb_v1_4_0.py \
  --pdb *pdb \
  --dt-ns 0.2 \
  --atom-type-field resname \
  --track-element Li:Li:+1 \
  --track-atom-type FSA_N:72:-1 \
  --fit-start-ns 1.0 \
  --fit-end-ns 250.0 \
  --timeseries-start-ns 1.0 \
  --max-lag-ns 250.0 \
  --remove-drift all \
  --dimension 3 \
  --msd-blocks 5 \
  --msd-error-stat sem \
  --block-max-lag-ns 250 \
  --outdir 1_10_onsager_multiT_v3
```

### 5.2 このコマンドが行う解析

- `*pdb` に一致する複数温度PDBを順番に解析
- ファイル名から `240K`, `270K`, `300K` などを自動認識
- Liを元素名で選択
- atom type 72のN原子をFSAの代表点として選択
- 全原子の質量中心ドリフトを除去
- 最大lag 250 nsまでfull-trajectory相関を計算
- 1–250 nsを線形fit範囲として伝導度を計算
- 軌跡を5個の非重複blockに分割してMSD誤差を評価
- 複数温度の伝導度、Arrhenius plot、共通y軸図を作成

### 5.3 FSA_NはFSA分子COMではない

```bash
--track-atom-type FSA_N:72:-1
```

は、atom type 72の各原子を独立粒子として追跡する。

したがって、この例では

- FSA分子のN原子位置をFSA運動のproxyとして使用
- FSA全原子から計算した質量中心ではない

という意味になる。

FSA分子COMを用いるには、通常のPDB residue nameが保持されている場合、例えば次を使用する。

```bash
--track-residue-com FSA:FSA:-1
```

複数のresidue nameを許す例：

```bash
--track-residue-com FSA:FSA,FSI,TFSI,NFS:-1
```

> **現在のv1.4.0の制約**  
> `onsager_distinct_transport_pdb_v1_4_0.py` には、atom type markerから分子全体を選んでCOMを作る `--track-molecule-atom-type-com` は実装されていない。Tinker `xyzpdb` でresidue name欄がatom type番号に置換されている場合、FSA_N proxyを使うか、PDB側で分子residue情報を保持する必要がある。

---

## 6. distinct解析の主要オプション

### 6.1 入力と温度

| オプション | 意味 |
|---|---|
| `--pdb FILE [FILE ...]` | 1つ以上のPDB。glob使用可 |
| `--dt-ns FLOAT` | 保存フレーム間隔。単位ns |
| `--temperature-k FLOAT` | ファイル名から温度を取得できない場合のfallback |
| `--temperature-map FILE:K ...` | ファイルごとの温度指定 |
| `--no-temperature-from-name` | ファイル名からの温度自動認識を停止 |
| `--outdir DIR` | 出力ディレクトリー |

温度決定の優先順位：

1. `--temperature-map`
2. ファイル名
3. `--temperature-k`

認識例：

```text
T300K
nvt300K
300K
```

今回の例では `--temperature-k 330` を指定していても、`1_0p8_nvt330K.pdb` から温度を認識するため、ログは

```text
T=330 K (filename)
```

となる。

### 6.2 フレーム選択

| オプション | 意味 |
|---|---|
| `--discard-frames N` | 冒頭Nフレームを除外 |
| `--stride N` | Nフレームごとに解析 |

有効時間間隔：

```math
\Delta t_{eff}=\texttt{dt-ns}\times\texttt{stride}
```

### 6.3 荷電種の選択

すべて次の形式を使う。

```text
LABEL:SELECTOR:CHARGE
```

例：

```bash
--track-element Li:Li:+1
--track-atom-type FSA_N:72:-1
--track-residue-com FSA:FSA:-1
```

| オプション | 選択方法 |
|---|---|
| `--track-element` | 元素記号で原子を選択 |
| `--track-atom-type` | Tinker atom typeで原子を選択 |
| `--track-residue-com` | residue単位の質量中心を追跡 |

- LABELは出力ファイル名・legendに使われる。
- LABELは重複不可。
- 少なくとも2種類の荷電種が必要。
- 電気的中性や粒子数の整合性は自動検査されない。

### 6.4 fitとrunning estimate

| オプション | 意味 |
|---|---|
| `--max-lag-ns` | full-trajectory相関の最大lag |
| `--fit-start-ns` | 最終伝導度fitの開始lag |
| `--fit-end-ns` | 最終伝導度fitの終了lag |
| `--timeseries-start-ns` | running fitの固定開始lag |
| `--timeseries-min-points` | running fitに必要な最小点数 |

running conductivityは、開始時刻を固定し、fit終了時刻を順次伸ばした推定値である。

```text
fit range = [timeseries-start-ns, current upper fitting time]
```

### 6.5 ドリフト除去

| 指定 | 動作 |
|---|---|
| `--remove-drift all` | 全原子の質量中心ドリフトを除去。推奨 |
| `--remove-drift selected` | 選択粒子のみの中心移動を除去 |
| `--remove-drift none` | ドリフト除去なし |
| `--drift-geometric` | 質量中心ではなく幾何中心を使用 |

collective displacementは一様並進ドリフトの影響を強く受けるため、通常は

```bash
--remove-drift all
```

を使用する。

### 6.6 次元

```bash
--dimension 3
```

- bulk 3D輸送：3
- 面内2D輸送として解釈する場合：2
- 1D輸送として解釈する場合：1

ただし、現在のスクリプトは座標成分を明示的に選別せず、3成分の内積を計算する。`dimension` は分母の `2d` にのみ入るため、異方的解析では別途座標成分選択の実装が必要である。

---

## 7. block averagingの意味

### 7.1 block分割

```bash
--msd-blocks 5
```

は、解析後の全フレームを5つの**連続・非重複block**に分ける。

```math
N_{block\,frames}
=
\left\lfloor
\frac{N_{frames}}{N_{blocks}}
\right\rfloor
```

余った末尾フレームはblock統計では使われない。full-trajectory解析では全フレームが使われる。

### 7.2 block図の最大lag

block統計の最大lagは

```math
\min(
\texttt{max-lag-ns},
\texttt{block-max-lag-ns},
(T_{block}-\Delta t_{eff})
)
```

に制限される。

今回のログ：

```text
[blocks] n=5, frames/block=291, duration/block=58.2 ns
```

では、1 blockの長さが58.2 nsなので、

```bash
--block-max-lag-ns 250
```

と指定しても、block-average MSD図は約58 nsまでしか表示できない。

**5 blockを1 MD stepとして数えているわけではない。**  
1 block内で計算できるlagがblock長を超えられないためである。

### 7.3 誤差の定義

```bash
--msd-error-stat sem
--msd-error-stat sd
--msd-error-stat ci95
```

- `sd`：block間標準偏差
- `sem`：`SD / sqrt(Nblock)`
- `ci95`：`1.96 × SD / sqrt(Nblock)`

`ci95` は正規近似であり、Student-t分布やbootstrapではない。

### 7.4 block数の選び方

block数を増やすと：

- 独立replicate数は増える
- 1 blockは短くなる
- 表示可能lagとfit可能範囲は短くなる

block数を減らすと：

- 長いlagまで見られる
- 誤差推定のsample数が減る

実務上は、次を同時に満たす必要がある。

1. block長が拡散領域まで到達する
2. block数が最低5程度、可能なら10以上ある
3. 各blockの推定値が極端に不安定でない

---

## 8. distinct解析の出力

### 8.1 各温度・各PDBのCSV

#### `*_onsager_correlation_timeseries.csv`

含まれる主な列：

```text
time_ns
C_A2__self:Li
C_A2__distinct:Li-Li
C_A2__distinct:Li-FSA_N
sigma_mS_cm__self:Li
sigma_mS_cm__distinct:Li-Li
MSDtr_A2__Li
MSDsigma_A2__Li-Li
MSDsigma_A2__direct_total
```

- `C_A2__...`：未荷重または各項の基礎相関
- `sigma_mS_cm__...`：running fitから得た伝導度寄与
- `MSDtr`：1粒子平均のtracer MSD
- `MSDsigma`：charge-weighted collective相関の可視化量

#### `*_onsager_conductivity_decomposition.csv`

各self/distinct項の最終fit結果：

```text
term
kind
z_a
z_b
multiplicity_in_total
fit_start_ns
fit_end_ns
slope_A2_per_ns
fit_R2
sigma_pair_S_per_m
sigma_contribution_S_per_m
sigma_contribution_mS_per_cm
```

#### `*_onsager_block_MSD_statistics.csv`

block平均MSDと誤差：

```text
MSDtr_mean_A2__Li
MSDtr_error_A2__Li
MSDsigma_mean_A2__Li-Li
MSDsigma_error_A2__Li-Li
MSDsigma_mean_A2__direct_total
```

#### `*_onsager_run_info.txt`

- frame数
- 有効dt
- 温度
- 平均体積
- fit範囲
- unwrap設定
- drift設定
- decomposed total
- direct total

### 8.2 各温度の図

```text
*_onsager_self_MSDtr_*.png
*_onsager_self_MSDtr_*_loglog.png
*_onsager_distinct_MSDsigma_*.png
*_onsager_distinct_MSDsigma_*_abs_loglog.png
*_onsager_direct_MSDsigma_*.png
*_onsager_direct_MSDsigma_*_loglog.png
*_onsager_correlation_terms.png
*_onsager_conductivity_timeseries.png
*_onsager_conductivity_decomposition.png
```

#### self MSD図

1粒子平均tracer MSDを表示する。

```math
MSD_{tr,a}=\frac{1}{N_a}\sum_i\langle|\Delta r_i|^2\rangle
```

伝導度self項の内部では `N_a × MSDtr` を使用する。

#### distinct MSDσ図

chargeとmultiplicityを含む可視化量を表示する。

異符号イオン間では `z_a z_b < 0` なので、同方向に協調移動する正の位置相関が、伝導度寄与として負になる場合がある。

negative distinct term自体はエラーではない。

### 8.3 複数温度の出力

```text
summary_all_onsager_conductivity.csv
temperature_total_onsager_conductivity.png
arrhenius_total_onsager_sigmaT.png
common_axis_multiT/
```

Arrhenius図は

```math
\log_{10}(\sigma T) \quad vs \quad 1000/T
```

をfitし、活性化エネルギーをeVで表示する。

- `σ` はmS cm⁻¹
- `σ > 0` の点のみ使用
- 現在の実装では伝導度誤差をfitに使用しない

---

# Part II. transport number比較

## 9. `onsager_transference_compare_pdb_v1_5_2.py`

### 9.1 今回の実行例

```bash
python3 onsager_transference_compare_pdb_v1_5_2.py \
  --pdb 1_0p8_nvt330K.pdb \
  --dt-ns 0.2 \
  --temperature-k 330 \
  --atom-type-field resname \
  --track-element Li:Li:+1 \
  --track-molecule-atom-type-com FSA:72:-1 \
  --track-solvent-atom-type-com SN:71 \
  --transference-cation Li \
  --transference-anion FSA \
  --fit-start-ns 1.0 \
  --fit-end-ns 300.0 \
  --timeseries-start-ns 1.0 \
  --max-lag-ns 300.0 \
  --remove-drift all \
  --dimension 3 \
  --msd-blocks 30 \
  --msd-error-stat ci95 \
  --block-max-lag-ns 300 \
  --tn-blocks 30 \
  --tn-error-stat ci95 \
  --tn-block-max-lag-ns 10 \
  --tn-ylim 0 1.2 \
  --require-solvent-fixed \
  --outdir transference_T330K_v2
```

### 9.2 分子選択

#### Li

```bash
--track-element Li:Li:+1
```

各Li原子を1粒子として追跡する。

#### FSA分子COM

```bash
--track-molecule-atom-type-com FSA:72:-1
```

処理手順：

1. atom type 72を含むresidueを検索
2. 同一 `(chain, resid)` に属する全原子を1分子としてgroup化
3. 全原子の質量加重COMを計算
4. 各FSA分子を電荷−1の粒子として追跡

今回のログ：

```text
[species] FSA: N=191, z=-1, kind=com
```

は191個のFSA分子COMを追跡したことを表す。

#### SN分子COM

```bash
--track-solvent-atom-type-com SN:71
```

処理手順：

1. atom type 71を含むresidueを検索
2. residue内の全原子からSN分子COMを計算
3. 全SN分子COMの平均位置をsolvent-fixed基準として使用

今回のログ：

```text
[solvent reference] SN: N=153, selector=atom_type=[71], atoms/molecule={10: 153}
```

は、

- SN分子数：153
- 各SN分子の原子数：10

を意味する。

> marker atom typeは各分子に1個だけである必要はないが、意図しない分子に同じatom typeが含まれないことを確認する。

---

## 10. 実装されている3種類の輸率

この節の `Dplus`, `Dminus`, `Lpp`, `Lmm`, `Lpm` は、共通のEinstein/Onsager prefactorを除いた**変位相関の傾き**である。比を取るため共通係数は相殺される。

### 10.1 PFG-NMR型 apparent transference number

```math
t_{+,app}^{PFG}
=
\frac{D_+}{D_+ + D_-}
```

スクリプトでは、カチオン・アニオンのtracer MSDの傾きを用いる。

特徴：

- self diffusionのみを使用
- distinct correlationを含まない
- Nernst–Einstein型のapparent quantity

### 10.2 eNMR型の電流分率

一般電荷では

```math
t_+^{eNMR}
=
\frac{z_+\left(z_+L_{++}+z_-L_{+-}\right)}
{z_+^2L_{++}+z_-^2L_{--}+2z_+z_-L_{+-}}
```

1:1電解質 `z+=+1`, `z−=−1` では

```math
t_+^{eNMR}
=
\frac{L_{++}-L_{+-}}
{L_{++}+L_{--}-2L_{+-}}
```

特徴：

- same-species・cross correlationを含む
- 基準系に依存する
- スクリプトではbarycentricとsolvent-fixedを比較できる

### 10.3 Bruce–Vincent型 steady-state transference number

現在の実装は、二元単価電解質に対して

```math
t_{+,ss}^{BV}
=
\frac{L_{++}-L_{+-}^2/L_{--}}
{L_{++}+L_{--}-2L_{+-}}
```

を用いる。

特徴：

- binary monovalent electrolyte用
- `z+ != +1` または `z− != −1` の場合は `NaN`
- MDのOnsager係数から得る理論的steady-state quantity
- 実験の電流・抵抗補正式を直接simulationしているわけではない

> **重要**  
> このスクリプトのBruce–Vincent値は、EISやDC polarizationの生データを読み込んで算出するものではない。Onsager係数から対応するsteady-state表式を評価した値である。

---

## 11. 基準系

### 11.1 barycentric

出力名 `barycentric` は、解析対象イオンに対して追加の溶媒COM subtractionを行っていない位置系列を表す。

推奨設定

```bash
--remove-drift all
```

では、先に全原子の質量中心ドリフトを除去するため、全系のmass-fixed / barycentric frameに対応する。

`--remove-drift none` や `selected` を使った場合でも出力名は `barycentric` のままであるため、厳密な物理的意味は設定に依存する。

### 11.2 solvent_fixed

SN分子COMを `R_SN,m(t)` とすると、

```math
\bar{R}_{SN}(t)
=
\frac{1}{N_{SN}}\sum_m R_{SN,m}(t)
```

を計算し、各イオン位置から引く。

```math
r_i^{solvent-fixed}(t)
=
r_i(t)-\bar{R}_{SN}(t)
```

これにより中性溶媒の平均移動に固定した輸率を評価する。

### 11.3 solvent-fixedを必須にする

```bash
--require-solvent-fixed
```

を指定すると、溶媒referenceを定義できなかった場合に停止する。溶媒選択ミスを見逃さないため、solvent-fixed輸率を主目的とする場合は指定を推奨する。

---

## 12. 溶媒診断

SNを指定すると、次を自動計算する。

### 12.1 SN self-MSD

```math
MSD_{SN}^{self}(t)
=
\frac{1}{N_{SN}}
\sum_m
\langle|\Delta R_{SN,m}(t)|^2\rangle
```

これはSN分子の自己拡散に対応する。

### 12.2 SN mean-COM MSD

```math
MSD_{SN}^{meanCOM}(t)
=
\left\langle
|\Delta\bar{R}_{SN}(t)|^2
\right\rangle
```

これは**SN分子自己拡散係数ではない**。溶媒集団の平均位置がどれだけ動くかを調べる基準系診断である。

### 12.3 `N × mean-COM MSD`

独立な同一分子の運動では、おおよそ

```math
N_{SN}MSD_{SN}^{meanCOM}
\sim
MSD_{SN}^{self}
```

が期待される。両者の比較により、溶媒集団のcollective motionや基準系変動の大きさを確認する。

### 12.4 出力例

```text
*_SN_self_and_meanCOM_MSD.csv
*_SN_diffusion_diagnostics.csv
*_SN_block_MSD_diagnostics.csv
*_SN_self_vs_meanCOM_MSD_loglog.png
*_SN_self_vs_scaled_meanCOM_MSD.png
*_SN_running_diffusion_diagnostics.png
```

ログ：

```text
[solvent diffusion] SN self D = ... cm^2/s
[solvent reference] mean-COM diagnostic D = ...
N*D_COM = ...
```

`SN self D` は分子自己拡散係数、`mean-COM diagnostic D` は基準系診断量である。

---

## 13. transport number block統計

### 13.1 MSD blockとTN blockは独立

```bash
--msd-blocks 30
--tn-blocks 30
```

- `msd-blocks`：self/distinct/direct MSD曲線の誤差
- `tn-blocks`：3種類の輸率の誤差

を制御する。

### 13.2 今回の30 block例

ログ：

```text
[blocks] n=30, frames/block=71, duration/block=14.2 ns
```

有効dtが0.2 nsなので、1 blockは14.2 nsである。

#### MSD block

```bash
--block-max-lag-ns 300
```

としても、MSD block図の最大lagは約14 nsに制限される。

#### TN block

```bash
--tn-block-max-lag-ns 10
```

なので、各block内のrunning輸率は最大10 nsまで計算される。

最終block estimateのfit範囲は、今回の設定では

```text
1.0–10.0 ns
```

となる。

一方、full-trajectoryのOnsager伝導度fitは

```text
1.0–300.0 ns
```

である。

したがって、**block輸率の統計とfull-trajectory伝導度は同一fit範囲ではない**。

### 13.3 TN誤差

```bash
--tn-error-stat ci95
```

は

```math
1.96\times SD/\sqrt{N_{block}}
```

を表示する。

### 13.4 block数選択の推奨

輸率は複数のcollective slopeの比であり、MSDより不安定になりやすい。

推奨手順：

1. まず `tn-blocks=5–10` で長いblockを確保
2. 各blockの `Lpp`, `Lmm`, `Lpm` が十分線形か確認
3. block間ばらつきと時間収束を確認
4. 軌跡が十分長い場合のみblock数を増やす

30 blockはCI sample数を増やす一方、14.2 ns/blockしかないため、遅い輸送系では拡散領域に到達しない可能性がある。

---

## 14. transport number解析の追加オプション

| オプション | 意味 |
|---|---|
| `--track-molecule-atom-type-com` | marker atom typeを含む荷電分子の全原子COM |
| `--track-solvent-residue-com` | residue nameで中性溶媒COMを指定 |
| `--track-solvent-atom-type-com` | marker atom typeで中性溶媒全分子COMを指定 |
| `--transference-cation` | カチオンとして使うspecies label |
| `--transference-anion` | アニオンとして使うspecies label |
| `--require-solvent-fixed` | solvent-fixedが作れなければ停止 |
| `--no-solvent-diagnostics` | 溶媒MSD診断を省略 |
| `--tn-blocks` | TN統計用block数 |
| `--tn-error-stat` | `sd`, `sem`, `ci95` |
| `--tn-block-max-lag-ns` | TN block内の最大lag |
| `--tn-ylim YMIN YMAX` | TN図のy範囲 |

### 14.1 binary electrolyteとして使用する

伝導度分解部分は複数荷電種を扱えるが、輸率比較は指定したcation/anionの2種から計算する。

追加の荷電種を同時に定義すると、

- total conductivity分解には追加種が入る
- transference numberの式には指定cation/anionしか入らない

ため、解釈が不整合になりうる。輸率解析では原則としてbinary electrolyteの2荷電種のみを定義する。

---

## 15. transport number解析の出力

### 15.1 輸率時系列

```text
*_transference_running.csv
```

主な列：

```text
time_ns
PFG_t_app__barycentric
eNMR_t0__barycentric
Bruce_Vincent_tss__barycentric
PFG_t_app__solvent_fixed
eNMR_t0__solvent_fixed
Bruce_Vincent_tss__solvent_fixed
```

### 15.2 方法比較図

```text
*_transference_methods_timeseries_solvent_fixed.png
```

solvent-fixedが存在する場合は、PFG-NMR、eNMR、Bruce–Vincentのrunning estimateをsolvent-fixed frameで比較する。

### 15.3 基準系比較図

```text
*_transference_reference_frame_comparison.png
```

eNMR型輸率について

- barycentric
- solvent-fixed

を比較する。

### 15.4 block estimate

```text
*_transference_block_estimates_barycentric.csv
*_transference_block_estimates_solvent_fixed.csv
*_transference_block_summary_barycentric.png
*_transference_block_summary_solvent_fixed.png
```

block CSVには各blockの

```text
PFG_t_app
eNMR_t0
Bruce_Vincent_tss
sigma_reduced
Dplus
Dminus
Lpp
Lmm
Lpm
block
```

が保存される。

### 15.5 Onsager伝導度出力

transport numberスクリプトは、distinct解析スクリプトと同様に以下も出力する。

```text
*_onsager_correlation_timeseries.csv
*_onsager_conductivity_decomposition.csv
*_onsager_block_MSD_statistics.csv
*_onsager_self_MSDtr_*.png
*_onsager_distinct_MSDsigma_*.png
*_onsager_direct_MSDsigma_*.png
summary_all_onsager_conductivity.csv
```

複数温度入力時には温度依存図とArrhenius図も作成する。

---

## 16. RuntimeWarningについて

今回の実行では次のwarningが出ている。

```text
RuntimeWarning: Mean of empty slice
RuntimeWarning: Degrees of freedom <= 0 for slice
```

主な原因は、blockごとのrunning輸率において、初期時刻ではfit点数が

```bash
--timeseries-min-points
```

に達しておらず、全blockで値が `NaN` になる列が存在するためである。

`_error_from_blocks()` がその全NaN列に対して `nanmean` / `nanstd` を実行するとwarningが出る。

### 16.1 通常問題にならないケース

- warningが初期running-time領域だけに対応
- 後半の輸率曲線に有限値がある
- fixed-window block CSVに有限値がある
- summary図が正常に作られる

この場合、warningは初期未定義領域に由来し、最終fit結果そのものの失敗を意味しない。

### 16.2 確認すべきケース

- block CSVがほぼNaN
- `tn-block-max-lag-ns < fit-start-ns`
- 1 blockのframe数が少ない
- `Lmm` またはtotal reduced conductivityが0付近
- Bruce–Vincent値が極端に発散

### 16.3 回避方法

- `tn-blocks` を減らして1 blockを長くする
- `tn-block-max-lag-ns` を長くする
- `timeseries-min-points` を適切に調整する
- warningを出さないよう、全NaN列を明示処理するコード修正を行う

---

## 17. 推奨解析手順

### Step 1. トポロジー確認

```bash
--list-topology --only-list-topology
```

確認項目：

- Li数
- FSA marker数
- SN marker数
- atom typeの読取欄

### Step 2. 短いlagで試験実行

```bash
--max-lag-ns 10
--fit-start-ns 1
--fit-end-ns 10
```

確認項目：

- species数が化学組成と一致
- `decomposed` と `direct` が一致
- unwrap異常がない
- 出力CSVと図が作成される

### Step 3. full-trajectory解析

- log–log MSDでballistic/subdiffusive/diffusive領域を確認
- fit範囲を設定
- running conductivityのplateauを確認
- `fit_R2` だけでなく時間窓依存性を見る

### Step 4. block統計

- block長が拡散領域を含むようにblock数を決める
- mean ± error bandを確認
- blockごとの輸率値にoutlierがないか確認

### Step 5. reference-frame比較

- barycentricとsolvent-fixedの差を確認
- SN self-MSDとmean-COM MSDを確認
- solvent-fixed referenceが極端に移動していないか確認

### Step 6. 物理解釈

- PFGとeNMRの差：distinct correlationの寄与
- eNMRとBruce–Vincentの差：steady-state constraintの寄与
- barycentricとsolvent-fixedの差：reference frameの寄与
- total conductivityとself項和の差：ionic correlationの寄与

---

## 18. 推奨パラメータの考え方

### 18.1 `fit-start-ns`

短時間のballistic/cage領域を除外し、MSDまたはcollective correlationが線形になる時刻より後に設定する。

### 18.2 `fit-end-ns`

- 時間原点数が少なくなりnoiseが増える領域を避ける
- running estimateがplateauになる範囲を選ぶ
- 最大lagを軌跡全長近くまで使うことが必ずしも最善ではない

### 18.3 `max-lag-ns`

full trajectory長の1/3–1/2程度から試し、収束性を見る。非常に長いlagでは利用可能なtime origin数が減少する。

### 18.4 `msd-blocks`, `tn-blocks`

「多いほど良い」ではない。最低限必要なblock長から逆算する。

例：

- 拡散領域が10 ns以降
- fitを10–40 nsで行いたい

なら、1 blockは少なくとも40–50 ns以上必要である。

---

## 19. 結果のチェックリスト

### 入力

- [ ] PDBの原子順序は全フレームで一定
- [ ] `CRYST1` が有効
- [ ] 直交セル
- [ ] `dt-ns` は実際の保存間隔
- [ ] atom type fieldが正しい

### species選択

- [ ] Li数が期待値と一致
- [ ] FSA数が期待値と一致
- [ ] SN数が期待値と一致
- [ ] FSA_N proxyかFSA COMかを区別している
- [ ] cation/anion labelが正しい

### 解析

- [ ] unwrapを有効化
- [ ] `remove-drift all` を使用
- [ ] linear fit範囲が拡散領域
- [ ] `decomposed ≈ direct`
- [ ] block長がfit範囲より長い
- [ ] running estimateが収束

### 輸率

- [ ] binary 1:1 electrolyteとして定義
- [ ] Bruce–Vincentの分母が0付近でない
- [ ] barycentricとsolvent-fixedを区別
- [ ] SN mean-COMを自己拡散係数として解釈していない

---

## 20. よくある問題

### 20.1 `No particles found for species`

原因：

- atom type fieldが違う
- element名がPDBから正しく推定されていない
- residue nameが想定と異なる

対策：

```bash
--list-topology --only-list-topology
```

で確認する。

### 20.2 `A valid CRYST1 box is required`

PDBにセル情報がない、または数値が不正。

### 20.3 `Too few frames after discard/stride`

`discard-frames` または `stride` が大きすぎる。

### 20.4 block図が指定したmax lagまで伸びない

正常動作である。block図の最大lagは1 block長を超えられない。

### 20.5 conductivityが負になる

collective correlationのnoise、fit範囲不良、有限サンプリング、強いanticorrelationが考えられる。単に絶対値を取らず、running slope・block依存性・直接total項を確認する。

### 20.6 複数温度の伝導度が単調でない

- サンプリング不足
- fit窓の不整合
- 温度ごとの軌跡長差
- phase/stateの違い
- collective termの大きな統計誤差

を確認する。Arrhenius fitを行う前に各温度のMSDとrunning conductivityを検証する。

---

## 21. 現在の実装上の注意

1. `onsager_distinct_transport_pdb_v1_4_0.py` のファイル冒頭docstringには旧version名が残っているが、本説明書では実ファイル名v1.4.0を使用する。
2. `onsager_transference_compare_pdb_v1_5_2.py` の冒頭例にも旧script名が残っている。
3. 両スクリプトで次のCLI optionは定義されているが、現在のコード中では実際の描画分岐に使用されていない。

```text
--show-term-titles
--save-total-term-figure
```

指定しても現在の出力は変化しない。

4. `ci95` は正規近似であり、厳密な有限sample t-confidence intervalではない。
5. conductivity decompositionの最終値自体にはblock由来の誤差列は付かない。block誤差はMSD曲線に付与される。
6. transport numberはblock estimateとその誤差を出力する。
7. triclinic cell、fractional-coordinate NPT unwrap、成分別異方的伝導度は未実装。

---

## 22. コマンド早見表

### 22.1 multi-temperature distinct解析

```bash
python3 onsager_distinct_transport_pdb_v1_4_0.py \
  --pdb *pdb \
  --dt-ns 0.2 \
  --atom-type-field resname \
  --track-element Li:Li:+1 \
  --track-atom-type FSA_N:72:-1 \
  --fit-start-ns 1.0 \
  --fit-end-ns 250.0 \
  --timeseries-start-ns 1.0 \
  --max-lag-ns 250.0 \
  --remove-drift all \
  --dimension 3 \
  --msd-blocks 5 \
  --msd-error-stat sem \
  --block-max-lag-ns 250 \
  --outdir 1_10_onsager_multiT_v3
```

### 22.2 transport number比較

```bash
python3 onsager_transference_compare_pdb_v1_5_2.py \
  --pdb 1_0p8_nvt330K.pdb \
  --dt-ns 0.2 \
  --temperature-k 330 \
  --atom-type-field resname \
  --track-element Li:Li:+1 \
  --track-molecule-atom-type-com FSA:72:-1 \
  --track-solvent-atom-type-com SN:71 \
  --transference-cation Li \
  --transference-anion FSA \
  --fit-start-ns 1.0 \
  --fit-end-ns 300.0 \
  --timeseries-start-ns 1.0 \
  --max-lag-ns 300.0 \
  --remove-drift all \
  --dimension 3 \
  --msd-blocks 30 \
  --msd-error-stat ci95 \
  --block-max-lag-ns 300 \
  --tn-blocks 30 \
  --tn-error-stat ci95 \
  --tn-block-max-lag-ns 10 \
  --tn-ylim 0 1.2 \
  --require-solvent-fixed \
  --outdir transference_T330K_v2
```

### 22.3 PDF図も保存

上記に追加：

```bash
--save-pdf
```

### 22.4 表示フォント・サイズ調整

```bash
--font-size 14 \
--axis-label-size 18 \
--tick-label-size 15 \
--legend-font-size 12 \
--line-width 2.0 \
--fig-width 7.2 \
--fig-height 5.2 \
--dpi 300
```

---

## 23. 最小限の結果報告例

解析結果を研究ノート・論文SIで報告する際は、最低限次を記録する。

```text
Input trajectory:
Saved-frame interval:
Temperature:
Species definition:
Reference frame:
Drift removal:
Trajectory length after discard:
Maximum lag:
Linear-fit window:
Number and duration of blocks:
Error definition:
Total conductivity:
Self/distinct decomposition:
PFG apparent t+:
eNMR t+:
Bruce–Vincent t+:
Solvent-fixed or barycentric:
```

記載例：

```text
The PDB trajectory was analyzed at 330 K with a saved-frame interval of
0.2 ns. Li+ ions were selected by element, whereas FSA− and SN were represented
by molecular centers of mass identified using Tinker marker atom types 72 and
71, respectively. The global mass-center drift was removed. Onsager displacement
correlations were fitted over 1–300 ns. Transference-number uncertainties were
estimated from 30 contiguous blocks, with an intrablock fitting window of
1–10 ns, and are reported as the normal-approximation 95% confidence interval.
```

---

## 24. まとめ

- distinct解析スクリプトは、self、同種distinct、異種distinct、direct totalを分解する。
- `FSA_N:72` はFSA分子COMではなくN marker atomのproxyである。
- block-average図の最大lagは1 block長で決まり、full trajectoryの`max-lag-ns`とは異なる。
- transport numberスクリプトは、PFG apparent、eNMR、Bruce–Vincent型を同じ軌跡から比較する。
- solvent-fixed輸率には、正しい溶媒分子groupと平均COM referenceが必要である。
- SN self-MSDとmean-COM MSDは異なる物理量である。
- `decomposed` と `direct` の一致、running estimate、block依存性を必ず確認する。
