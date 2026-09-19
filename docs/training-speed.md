# Burn / JAX 学習速度比較（2026-09-09）

同じ Radeon 8060S 上で、学習更新の計算部分は JAX が **3.29〜3.43 倍**速かった。
一方、`images_32x` の PNG 復号には両実装とも約 2.3 秒 / 16 枚かかる。
学習全体の所要時間には、この読み込みと GPU 処理の重なり方も影響する。

## 学習更新の結果

1 更新 = 実効 batch 16（microbatch 4 × 4）。各条件で初めの 5 更新を除き、
続く 20 更新を測定した。表は算術平均。

| 再帰段数 | Burn / 更新 | JAX / 更新 | JAX の速度倍率 | Burn / 秒 | JAX / 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 段 | 2.913 秒 | 0.849 秒 | 3.43 倍 | 5.49 枚 | 18.85 枚 |
| 6 段 | 3.760 秒 | 1.144 秒 | 3.29 倍 | 4.26 枚 | 13.98 枚 |

20 回の P10〜P90 は、Burn が 1 段 2.884〜2.944 秒、6 段 3.725〜3.794 秒、
JAX が 1 段 0.827〜0.863 秒、6 段 1.118〜1.152 秒。
JAX の 6 段では 1 回だけ 1.282 秒かかった更新も、平均に含めた。

計測範囲は、画像・強度マップの pyramid、ノイズ生成、OT 割り当て、混合、
再帰 forward、loss、backward、勾配蓄積、AdamW 更新まで。
JAX は現行の勾配ノルム確認・clipping・EMA 更新も含む。
入力はあらかじめ GPU に置き、各更新の GPU 処理完了を待つ。
画像読み込み・転送、ログファイル書き込み、検証、サンプル生成、checkpoint 保存は除外した。

## 初回更新

| 再帰段数 | Burn | JAX |
| --- | ---: | ---: |
| 1 段 | 5.08 秒 | 7.72 秒 |
| 6 段 | 8.25 秒 | 22.24 秒 |

初回更新にはその場で必要なコンパイル等を含む。プロセス起動・import・モデル作成は含まない。
初回更新の前の初期化は、Burn が約 0.05〜0.06 秒、JAX が約 3.5〜3.6 秒だった。
既存のコンパイラ・ドライバー等のキャッシュは消していないため、完全な初回環境での
コンパイル時間を比較したものではない。Rust の release ビルドは別に実施した。

## ノイズ準備の独立測定

同じ入力・候補数で、縮小、マップ生成、Gaussian 候補生成、OT 割り当て、混合だけを
独立に測定した。ここも warmup 5 回 + 測定 20 回。

| 再帰段数 | Burn 平均 | JAX 平均 | JAX 中央値 |
| --- | ---: | ---: | ---: |
| 1 段 | 506.1 ms | 16.8 ms | 16.9 ms |
| 6 段 | 599.0 ms | 46.6 ms | 26.7 ms |

JAX の 6 段には 155.9 / 292.9 ms の 2 回があり、平均に含めた。
外れ値の原因は特定していない。

Burn は候補を chunk に分けて全画素の組み合わせ誤差を計算し、Rust の割り当て solver を使う。
JAX は同じコストを行列演算へ変形した実装と SciPy の CPU solver を使う。
この実装差も全体の速度差に含まれる。詳しいコスト式は [noise-matching.md](noise-matching.md)。
独立測定では fusion や同期の境界が変わるため、全更新からこの数値を引いて
CNN 単体の正確な時間とはしていない。

## PNG 読み込み・復号

両実装の既存の crop reader を使い、同じファイル 16 枚から 512×512 を 4 並列で切り出した。
各 source の中央付近を使い、繰り返しごとに座標を 1 pixel ずらして同じ crop の結果の
キャッシュ利用を避けた。最初の 1 回を除き、続く 3 回の平均。
OS のファイルキャッシュは消していない。

| source | 元画像サイズ | Burn | JAX / pyvips |
| --- | --- | ---: | ---: |
| images | 1024×1024 | 0.010 秒 | 0.025 秒 |
| images_32x | 32768×32768 | 2.290 秒 | 2.278 秒 |

大きな PNG では、切り出す領域までの復号が JAX の学習更新 1.144 秒より長い。
高解像度 source の読み込みを GPU 処理と重ねても、入力供給が制約になりうる。
既存の tiled TIFF 前処理を次に比較する価値があるが、今回は TIFF の速度は測っていない。

2026-09-19にJAX側でTIFF比較を追加した。train由来4元画像・4並列・16cropの
最大scaleではPNG約2.998秒に対しTIFF約0.01631秒だった。画像とGrain batchは完全一致。
小さなwarm-cache標本であり、Burnとの新たな全学習比較やGPU速度の測定ではない。
条件・移行方法は[残りの実験計画](completion-campaign-20260919.md#画像読み込みの比較と保存形式の移行)を参照。

これは reader の独立測定であり、Grain / Burn DataLoader を含めた通常の学習 run 全体の
時間ではない。読み込みと GPU 演算の数値は単純には加算できず、全体が一律 3.3 倍になるとは
判断していない。

## 条件と比較の範囲

- Burn: `87cb7b2`、Burn 0.21.0、Vulkan + Fusion、Mesa RADV 25.2.8、Rust 1.98.1、release build。
- JAX: `88f0eec`、JAX / jaxlib 0.11.1、Flax 0.12.9、Optax 0.2.8、ROCm 7.2.1。
  `JAX_PLATFORMS=rocm,cpu`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- GPU: AMD Radeon 8060S / gfx1151。各 GPU 測定は別プロセスで順に実行した。
- モデル: 両方 126,765 parameters、channels `[4, 10, 20, 40]`、3 conv / block、float32。
- 入力: `images/0000.png` から先頭 16 枚の中央 512×512 crop を一度作成し、同じ float32
  データを両実装に渡した。再帰 6 段の演算測定にも同じ crop を使用した。
  6 段の解像度は 512→256→128→64→32→16。
- batch: 両方実効 16 / micro 4、各更新を一つの scale group とした。
  Burn 標準の batch 128 / micro 16 での測定ではない。
- OT: 各段の pool 256、厳密割り当て。再帰の重み共有・勾配伝播を維持。
  `scale_loss_multiplier=0`、tone balance 有効、identity 0.1、edge 0.1、contrast 0.01。
- 学習率: base `1e-4` × sqrt(16)。Burn benchmark は一定 `4e-4`、JAX は warmup を 0 にし、
  現行 100,000 更新 cosine schedule の冒頭を使用した。
- 初期重み・乱数系列、noise map、loss の細部、optimizer の細部は各実装のもの。
  JAX には EMA / clipping があり、Burn には同じ処理はない。
  現行実装の処理速度を比較しており、学習の収束速度・画質は比較していない。
- 通常のデータ列は Burn が更新内に複数 scale を混在させ、JAX が更新ごとに一つの scale を選ぶ。
  今回の演算比較ではその差を避けた。混在 scale、異なる microbatch、別のモデル構造では再計測が必要。

## 記録と再実行

設定、入力の SHA-256、ファイルと crop 座標、全計測値は
[benchmarks/training-speed-20260909.json](benchmarks/training-speed-20260909.json) に保存した。

ローカルの `runs/training-speed-20260909/` には共有入力、計測スクリプト、stdout / stderr、
比較対象ソースの tar を置いた。`burn-harness` は元の `src/train.rs` を include して
既存の `TrainStep::step` と optimizer を呼ぶ。元の Rust ソースは編集していない。
JAX は既存の `training_inputs` / `make_steps` を、Trainer と同じ microbatch 蓄積方法で呼ぶ。

以下は JAX worktree から実行する例。GPU 計測を同時には走らせない。

```bash
# Burn: 最後の引数を prepare にするとノイズ準備のみ。1 を 6 にすると再帰 6 段。
../../target/release/lineartgen-speed-burn \
  runs/training-speed-20260909/batch16-512.f32 512 16 4 1 256 5 20 full

# JAX: --mode prepare でノイズ準備のみ。
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python runs/training-speed-20260909/benchmark_jax.py \
  runs/training-speed-20260909/batch16-512.f32 --levels 1 --warmup 5 --measured 20

../../target/release/lineartgen-speed-io runs/training-speed-20260909/io-manifest.json
.venv/bin/python runs/training-speed-20260909/benchmark_io.py
```
