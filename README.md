# lineartgen — JAX rewrite

JAX / Flax NNX で線画モデルを学習し、ONNX に出力する。
Krita の Diffusion Drawing は小さな別パッケージ `lineartgen-runtime` で推論する。
このブランチに Rust ソース・Cargo・ネイティブ拡張のビルドはない。

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
  diffusion.py            ノイズ・ピラミッド・小→大の再帰・sampling
  noise_matching.py       Burn と同じコスト・一対一ノイズ割り当て
  data/images.py          画像 decode / crop / optional cache
  data/pipeline.py        元画像分割と Grain の入力列
  training/losses.py      loss の数式
  training/steps.py       勾配・更新・評価の JIT 関数
  training/trainer.py     学習ループ・蓄積・ログ・評価
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

export するのは 1 解像度の U-Net のみ。BGRA 入出力、padding、seed、段階的な denoise と
再帰は NumPy で実行する。Krita 内では JAX / Flax / Orbax を import しない。

```bash
uv run --frozen --extra export lineartgen export --run-dir runs/experiment-001 \
  --output packages/lineartgen-runtime/src/lineartgen_runtime/assets/model.onnx
uv build --project packages/lineartgen-runtime --wheel --out-dir dist
```

同梱済み asset は旧 epoch 1500 の学習済み重みを Python で移したもの。
新しい学習を完了する前でも Krita で使える。元の MPK はこのブランチでは不要。
移行ツールは `scripts/import_burn_checkpoint.py`。乱数生成器と backend の差により
旧版と同じ Seed の出力を完全再現するものではない。

ONNX は opset 20 / IR 10。jax2onnx の可変形状 Pool / Resize 変換には制約があるため、
固定形状で trace した後、許可した演算だけを対象に Resize を定数倍率へ変換し、
空間次元を可変にする。未知の演算は明示的にエラーにする。
export のたびに小サイズ・非正方形・大サイズで JAX と ONNX Runtime を数値比較し、
成功してから出力先を置き換える。

ランタイムは Python 3.10 以上をサポートし、専用 `uv.lock` を持つ。
Windows は DirectML、Linux などは CPU 用の wheelhouse を作って ZIP に同梱する。
実際の ZIP ビルドは Diffusion Drawing 側の README / `scripts/package.py` を使う。
この作業の consumer worktree は `../diffusion-drawing-jax`。

[選定理由と一次資料](docs/decisions.md)、[検証結果と残る制約](docs/validation.md)も参照。
