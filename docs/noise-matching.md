# ノイズ割り当て（2026-09-08）

既定は Burn と同じ masked corruption cost による厳密な一対一対応に戻した。
今回の改善は目的関数を変えない計算の高速化であり、画質の改善はまだ測っていない。
実装は `src/lineartgen/noise_matching.py`、学習入力の組み立ては `diffusion.py`。

OT を学習するという目的から、ノイズ生成・補間経路・条件付き分布まで見直す案は
[OT の学習設計](ot-training-design.md)に整理した。以下は元の式の再現と高速化の記録。

## 維持した動作

- 各再帰解像度で独立した標準正規ノイズの候補 pool を生成する。
- 同じ解像度の画像 batch に対し、候補を重複させず合計コストが最小になる割り当てを選ぶ。
- noise level が 0 の画素は割り当てコストに寄与しない。
- 候補を平均せず、選んだノイズ画像そのものを使う。
- 割り当ては microbatch 分割前に行う。モデルの勾配には割り当て solver を含めない。

JAX 版では一つの optimizer 更新の全画像で元データの解像度を揃える。
Burn の「outer batch 内を解像度別に group 化」とは batch の構成が異なるが、
microbatch サイズを変えても同じ実効 batch・seed なら割り当ての入力列は変わらない。
これは loss 全体や浮動小数点の勾配蓄積まで microbatch 分割に不変という意味ではない。

## 等価なコストの計算

画像を `x_i`、画素ごとの noise level を `t_i`、ノイズ候補を `z_j` とすると、Burn のコストは

```text
C[i,j] = sum(adaptive_avg_pool((x_i - sqrt(1-t_i)*x_i - sqrt(t_i)*z_j)^2))
```

pool 後の辺は最大 32。pool の和を画素ごとの重み `w[p]` で表し、
`a_i = (1-sqrt(1-t_i))*x_i` と置くと、同じ式を次のように計算できる。

```text
C[i,j] = sum_p w[p]*a_i[p]^2
       + sum_p w[p]*t_i[p]*z_j[p]^2
       - 2*sum_p w[p]*a_i[p]*sqrt(t_i[p])*z_j[p]
```

後ろの二項を行列積で計算する。候補 pool 自体は必要だが、
`[batch, pool, height, width]` の二乗誤差テンソルを作る必要がなくなる。
割り切れない pool の重なりも重みに反映し、NumPy の元式とテストで比較した。
浮動小数点の丸めでコストが僅かに変わり、同点付近の候補選択は異なる可能性がある。

元コードの `low_frequency_cost_tensor` は二乗誤差を **計算した後** に pool している。
標準の 2 の冪の画像では、その和は画素 L2 コストの定数倍になる。
画像を低解像度にしてから差を取る「低周波で比較する方法」とは異なる。

## solver と実測

既定は JAX でコスト計算し、SciPy `linear_sum_assignment` で厳密に解く。
SciPy の実装は矩形問題に対応する Jonker–Volgenant 法。
GPU には画像・候補を置いたまま、小さいコスト行列と結果の index だけを転送する。
Optax の Hungarian も `matching.solver = "optax"` で選択できる。
どちらも同じ assignment 問題を解き、小さい問題の全列挙との一致を検証した。
[SciPy 公式資料](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)、
[Optax 公式資料](https://optax.readthedocs.io/en/latest/api/assignment.html)。

Radeon 8060S / ROCm 7.2.1 / JAX 0.11.1、512×512、batch 16、pool 256。
合成画像、noise level 0.5（一部 0）、JIT の初回実行を除外した 20 回の中央値。
各計測は `block_until_ready` で演算完了を待つ。

| 処理 | 時間 |
| --- | ---: |
| Burn の chunked broadcast 式を JAX で計算 | 42.99 ms |
| 同じコストを行列積で計算 | 12.75 ms |
| Optax / GPU の厳密割り当てのみ | 2.94 ms |
| SciPy callback の厳密割り当てのみ（転送込み） | 0.21 ms |
| 行列積と SciPy callback を結合 | 12.82 ms |

コスト計算はこの条件で約 3.37 倍速い。元の Rust / Burn backend 自体の速度比較ではない。
コスト行列の最大絶対差は `1.83e-4`、平均の割り当てコストは約 `402.69`。
候補生成、画像読み込み、逆伝播を含む学習全体の速度向上率を表す数字でもない。

```bash
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen --extra rocm python scripts/benchmark_noise_matching.py \
  --size 512 --batch 16 --pool 256 --repeats 20 \
  --output runs/noise-matching-benchmark.json
```

SciPy callback には CPU backend も必要なので `JAX_PLATFORMS=rocm,cpu` を使う。
GPU 上の Optax solver を使う場合は `JAX_PLATFORMS=rocm` だけでもよい。

## 次に比較したい方法

最初の画質実験として、**候補数を実効 batch と同数にする厳密割り当て**を勧める。
例えば実効 batch 16 を固定して、次の三条件を比較できる。

| 設定 | 比較する性質 |
| --- | --- |
| `method = "exact"`, `pool_size = 256` | Burn と同じ、多い候補から部分集合を選ぶ方式 |
| `method = "exact"`, `pool_size = 16` | 生成した全ノイズを並べ替える方式 |
| `method = "independent"` | 割り当てをしない対照条件 |

同数なら生成済みノイズの集合をそのまま使える。多い候補からの選別はコストを下げる一方、
選別後のノイズ分布を標準正規から偏らせ得る。これは実装からの考察であり、
この線画モデルで同数方式の画質が優れるという実験結果ではない。
minibatch OT を生成モデルの coupling に使う研究はあるが、この masked diffusion の
学習・推論則にその結果を直接当てはめることはできない。
[Tong ほか: minibatch optimal transport](https://arxiv.org/abs/2302.00482)。

本当の低周波コスト、例えば「マスクを反映した残差を縮小してから二乗する方式」も候補になる。
計算量と画像構造への感度が変わるため、元の目的関数との比較実験として扱う。
単純にノイズ候補を縮小して差し替えるとノイズの分散まで変わる点にも注意する。

OTT-JAX の Sinkhorn は大きな問題で比較する候補だが、正則化した soft な輸送計画を解く。
矩形の一様周辺分布では全候補に質量を配るため、現状の「重複なく一部を選ぶ」問題と異なる。
soft な重みで候補ノイズを平均するとノイズの統計も変わり、各行の argmax は重複を防がない。
採用するなら周辺分布、正則化、hard な対応への変換を合わせて設計する必要がある。
現在の小さい assignment の 0.21 ms を短縮する目的では、導入を優先しない。
[OTT-JAX の linear OT / Sinkhorn](https://ott-jax.readthedocs.io/solvers/linear.html)。

比較時は同じ画像分割・実効 batch・更新数・LR・複数 seed を使い、固定した入力から生成して
線の連続性、細線、保護した線の変化を調べる。現在の validation loss は学習と同じ matching
設定で入力を作るため、異なる matching 間の値をそのまま画質の順位に使わない。
共通の独立ノイズを使う評価入力も固定して測る必要がある。
