# lineartgen — JAX rewrite

このブランチは累計100,200更新のEMAを同梱したKrita向けgraph試用版（runtime 0.3.0）。
Strength 0.5・Lineart passes 3から試せる。ベタ塗りの復元と曖昧な二重線には失敗が残る。
最新の配布方式は[graphモデルの説明](docs/graph-preview-20261019.md)を参照。
下記の学習CLI・実験履歴は従来のUNet系も含む。

2026-10-07、[芯の保持・局所誤差を含む6条件の追加実験](docs/ink-structure-experiment-20261007.md)を開始。

2026-10-06、[線画lossの5条件×追加10k比較を完了](docs/lineart-loss-results-20261006.md)。
領域別lossでにじみは減ったが細線の保持に副作用があり、配布モデルの更新は見送った。

2026-09-30、[予定した実験を完了](docs/completion-results-20260930.md)。
A2/A3、共同S/M/L・独立S・大型基準の3 seed×100k学習、未使用test、匿名画像確認、
ONNX／入出力監査まで実施した。全品質条件を満たす候補はなく、最小十分サイズは未確定。

JAX / Flax NNX で線画モデルを学習し、ONNX に出力する。
Krita の Diffusion Drawing は小さな別パッケージ `lineartgen-runtime` で推論する。
このブランチに Rust ソース・Cargo・ネイティブ拡張のビルドはない。

複数サイズで特徴計算を共有する入れ子モデルは `configs/nested.toml`。
[構造・学習比較・切り出しの手順](docs/nested-models.md)を参照。

合成下描きからの復元比較は`configs/sketch.toml`。
[4条件の10,000更新結果と比較画像](docs/sketch-experiment-20260911.md)に、改善点・副作用・保存モデルをまとめた。
続く[再帰条件と拡張率75%・50%・25%の比較](docs/sketch-rates-experiment-20260912.md)は
`configs/sketch-rates.toml`。背景の改善と下描き復元のトレードオフを記録した。
[生成線画20%の追加](docs/geometry-data-experiment-20260914.md)と
[境界近傍loss](docs/boundary-loss-experiment-20260914.md)の比較も完了した。
[半係数の追試](docs/boundary-half-experiment-20260915.md)では、EMAの線幅改善を一部残し、
切断増加を抑えた。解消には至らず、rawも含む全面採用は保留している。
[灰色線画向けの中心線loss比較](docs/cldice-loss-experiment-20260915.md)も2条件10,000更新まで完了。
背景は改善したが細線消失が全サイズで増え、EMA S/Mでは切断も増えたため、今回の設定は採用しない。
[N1 ノイズ強度sampling比較](docs/noise-strength-experiment-20260917.md)も2条件10,000更新まで完了。
低〜中強度80%の案は細線消失・切断が増え、高強度の復元も悪化したため採用しない。
[N2の矩形mask境界sampling](docs/mask-sampling-experiment-20260918.md)も10,000更新まで完了。
同じ面積・強度の8候補から境界ink最大を選ぶ案は、境界F1が安定して改善せず、
Sの切断率もraw/EMAとも増えたため採用しない。
続く[pool前skipとdilation2の比較](docs/completion-campaign-20260919.md)も不採用となり、
pool後・dilation1で長期比較を完了した。

## 環境を作る

Python 3.12 と uv を使う。依存関係は `uv.lock` で固定している。

```bash
uv sync --frozen --extra export
uv run --frozen --extra export lineartgen doctor
```

このマシンの Linux / Radeon 8060S (`gfx1151`) / ROCm 7.2.1 では、次の構成で
実データの学習と保存を確認した。ROCm はホストにインストール済みであることが前提。

```bash
uv sync --frozen --extra rocm --extra export
JAX_PLATFORMS=rocm,cpu uv run --frozen --extra rocm --extra export lineartgen doctor
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen --extra rocm --extra export lineartgen train \
  --config configs/default.toml --run-dir runs/experiment-001
```

`JAX_PLATFORMS=rocm,cpu` は GPU を既定にし、GPU 初期化の失敗をエラーにする。
CPU backend はノイズの割り当てを解く SciPy callback に必要。
共有メモリの GPU なので、この例では JAX の事前メモリ確保を無効にする。
CPU を明示する場合は `JAX_PLATFORMS=cpu`。他の GPU については
[JAX のインストール手順](https://docs.jax.dev/en/latest/installation.html)を参照。
uv の extra は sync / run の両方で指定する。忘れると不要と判定された GPU package が外れる。

## 学習・再開・出力

まず小さい crop で、実データを使う一連の処理を確認できる。

```bash
uv run --frozen --extra export lineartgen train \
  --config configs/smoke.toml --run-dir runs/smoke --stop-after 2
uv run --frozen --extra export lineartgen train --run-dir runs/smoke --resume
uv run --frozen --extra export lineartgen export \
  --run-dir runs/smoke --output runs/smoke/model.onnx
```

GPU 環境では上の run に `--extra rocm` も付ける。
既存の run ディレクトリは上書きしない。設定を変える実験では新しい run 名を使う。
`--resume` は保存済み設定を読み、モデル・optimizer・EMA・入力列と乱数の位置を復元する。
`--stop-after` は追加の optimizer 更新回数だけを制限し、学習率スケジュールを変えない。
Ctrl-C は完了した更新境界で保存する。突然のプロセス終了では直近の保存から再開する。

run ごとに次が残る。

| ファイル | 用途 |
| --- | --- |
| `config.json` | 実際に使った全設定 |
| `dataset.json` | 元画像単位の分割、名前・サイズ・mtime による指紋 |
| `metrics.jsonl`, `events/` | 学習・検証の loss、学習率、勾配ノルム、データの scale と再帰段数、TensorBoard イベント |
| `samples/` | 検証画像と生成結果を並べた PNG |
| `checkpoints/` | Orbax によるモデル・optimizer・EMA・step |
| export 先の `.onnx`, `.json` | EMA モデルと JAX / ONNX の数値比較結果 |

## データと前処理

既存の `dataset/images` から `images_32x` まで、同名画像を同じ元画像として扱う。
`images_colored` は保持するが標準の線画学習には使わない。
画像を解像度ごとに分割せず、元画像名で train / validation を分ける。
訓練では元画像と解像度を一様に復元抽出し、固定サイズの crop・回転・反転を行う。
各 optimizer 更新の実効 batch 全体で解像度選択を揃える。
その batch 全体にノイズを割り当ててから microbatch に分割する。
Burn は一つの outer batch 内に解像度別の group を作るため、batch の構成は異なる。
同じ画像・解像度の出現確率を保ち、再帰段数ごとに JIT する。

`scale_dirs` は標準で先頭から scale level 0, 1, … と解釈し、画像寸法の倍率も検査する。
一部だけを使う場合は、例えば `scale_dirs = ["images_8x", "images_32x"]` と
`scale_levels = [3, 5]` を組み合わせ、元の再帰段数を維持する。

PNG の読み込みには pyvips を使う。32K PNG は crop でもそこまでの行をデコードするため、
長い学習ではロスレスの tiled TIFF キャッシュを作るとランダム crop を読みやすくなる。
キャッシュ用の追加容量が必要。まず `--limit 1` でサイズと時間を確認できる。

```bash
uv run lineartgen prepare --config configs/default.toml --output .cache/tiles --limit 1
# 全画像を準備するときは --limit を外す
```

準備が済んだら TOML の `[data]` に `prepared_root = ".cache/tiles"` を指定する。
キャッシュは元画像の画素と alpha を保持し、元ファイルのサイズ・mtime 変更を検出する。
データを更新したら prepare を再実行する。学習中のデータ変更は避ける。
標準設定は従来の線画画像と同様に RGB の輝度を使い alpha を無視する。
透明画像を白に合成するデータなら `alpha_mode = "white"` に変更する。
前処理の変更箇所は `data/images.py`、サンプリングは `data/pipeline.py`。

## モデルと学習処理を変える

```text
configs/                  実験設定 TOML
src/lineartgen/
  config.py               型と設定間の整合性検査
  models/blocks.py         Conv + ReLU の部品
  models/unet.py           1 解像度の U-Net
  models/nested.py         小さいprefixへ大きい側の出力を混ぜない入れ子モデル
  diffusion.py            ノイズ・ピラミッド・小→大の再帰・sampling
  sketch.py               入力だけの不透明度・blur・太線・重ね描き拡張
  noise_matching.py       Burn と同じコスト・一対一ノイズ割り当て
  data/images.py          画像 decode / crop / optional cache
  data/pipeline.py        元画像分割と Grain の入力列
  training/losses.py      loss の数式
  training/steps.py       勾配・更新・評価の JIT 関数
  training/trainer.py     学習ループ・蓄積・ログ・評価
  training/comparison.py  共通条件を保った実験variantの定義
  training/checkpoints.py 保存・復元
  export.py              ONNX 変換と数値検証
packages/lineartgen-runtime/
  src/lineartgen_runtime/  Krita に配る推論処理・ONNX asset
tests/                    データ、再開一致、変更した構造のexport、BGRA契約
```

NNX の通常の Module / Optimizer を使う。モデルは NHWC の image、noise_level、
半分の解像度の insert を受け取り、同じ解像度の線画を返す。
標準 U-Net は従来のチャンネル構成 `[4, 10, 20, 40]`、3 conv/block、
MaxPool、nearest upsample、skip 接続を維持する。小さい解像度の出力を次へ渡し、
全解像度で重みを共有する。再帰を通る勾配も残す。

`channels`、`convolutions_per_block`、`kernel_size` は TOML で変更できる。
crop の辺は 2 の冪かつ `2 ** len(channels)` 以上にする。
モデル内部に新しい演算を入れたら exporter の対応も確認する。

次の学習設計は Burn 版を維持する。

| 設計 | 設定と動作 |
| --- | --- |
| 再帰段数 | `diffusion.original_steps + dataset scale level`。標準は images で 1 段、images_32x で 6 段。モデルの最小サイズと `stop_size` で打ち切る |
| 各段の loss | 最大解像度を段 0 として重み `scale_loss_multiplier ** 段番号`、重みの総和で正規化。標準値 0 は最大解像度だけに直接 loss を掛ける。小さい段にも再帰経由で勾配が届く |
| 学習率 | `learning_rate * sqrt(batch_size * accumulate_steps)` をピークにする。標準は `1e-4 * sqrt(16) = 4e-4`。`minimum_learning_rate = 1e-5` はスケーリングしない |
| ノイズ割り当て | 各再帰解像度で独立した候補 pool を作り、Burn と同じコストの厳密な一対一対応を選ぶ。標準は `matching.pool_size = 256` |

学習は optimizer step を単位とし、AdamW、warmup + cosine、勾配 clipping、EMA を使う。
実効 batch は `batch_size * accumulate_steps`。分散学習や mixed precision はまだ含まない。
loss はノイズ量で正規化した誤差、濃淡のバランス、identity、Sobel、contrast から構成する。
上表以外の loss の係数・式、ノイズ領域の生成、warmup などは JAX 版の設定を使う。
独立ノイズとの比較は `matching.method = "independent"`、GPU 上の厳密 solver との比較は
`matching.solver = "optax"` で切り替える。どちらも新しい run で行う。
コストの等価な高速化と候補手法の検討は [ノイズ割り当て](docs/noise-matching.md)を参照。
標準の長時間学習や生成品質はこれから評価する必要がある。

以前の JAX 設定の `recursion_levels` / `scale_loss_decay` は廃止した。
学習の意味が変わるため、それらを使った run は旧 commit `9d41a36` で再開し、
この設計での学習は新しい run にする。

## Krita への配布

runtime 0.3.0のgraph試用版は、共有重みから解像度段別に展開した6個のONNXを使う。
モデルは画素の濃淡ではなく、線の位置・接続・幅・塗りを予測する。
前処理・再帰・不透明線分の描画はNumPyで実行し、Krita内ではJAX / Flax / Orbaxをimportしない。
Lineart passesは1〜6、既定3。入力を局所的な濃さで正規化してから縮小し、
小さい段の占有予測を次の段へ渡す。Gaussianノイズは加えず、Seedは出力に影響しない。
既存のBGRA `LineartModel.infer` API、下描きのalpha利用、描き途中の線画の保護を維持する。
詳細、学習の出自、既知の品質制約は[graph試用版](docs/graph-preview-20261019.md)を参照。

```bash
uv build --project packages/lineartgen-runtime --wheel --out-dir dist
```

`assets/graph-model.json`はcheckpointと6個のONNXのhash、学習更新数、描画条件を記録する。
`assets/graph-reference.npz`はJAXから独立に生成した6段の検査用出力。
CIは実際に同梱するONNX Runtimeで各段と最終描画を照合する。
ONNXはopset20 / IR10、空間寸法は可変。大きな入力では重なり付きtileを使う。

従来のUNet export CLIは引き続き利用できる。明示的に`.onnx`パスを渡すと旧runtime経路を使う。
graphの配布重みの更新にUNet用`lineartgen export`を使わない。

ランタイムは Python 3.10 以上をサポートし、専用 `uv.lock` を持つ。
配布するWindows ZIPはPython3.13 x64用。DirectMLとCPU fallbackを同梱する。
実際の ZIP ビルドは Diffusion Drawing 側の README / `scripts/package.py` を使う。
この作業の consumer worktree は `../diffusion-drawing-jax`。

[選定理由と一次資料](docs/decisions.md)、[検証結果と残る制約](docs/validation.md)も参照。
