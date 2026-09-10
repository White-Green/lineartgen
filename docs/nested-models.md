# 入れ子モデルの実装と実験手順

`configs/nested.toml` は、特徴計算を共有する S / M / L のモデル。
現行の単独 U-Net は `architecture = "unet"` のまま利用できる。
[2,000 更新の比較結果](nested-experiment-20260910.md) に速度・品質・実機での切り出し検証を記録した。

## 構造

| 出力 | encoder の累積 channels | parameters |
| --- | --- | ---: |
| S / subnet 0 | [4,6,8] | 5,933 |
| M / subnet 1 | [6,8,12] | 10,206 |
| L / subnet 2 | [8,12,16] | 17,893 |

内部 pool は 3 段、各 block は 3×3 Conv / ReLU を 2 層。
parameter 数は各 prefix を実際に切り出して数えた値で、内部で必要な小型の出力 head も含む。
通常の dense U-Net の同じ channels に対する parameter 数とは異なる。

- チャンネル群 i の各 Conv は群 0..i のみを読む。禁止した接続の重み・optimizer state は持たない。
- skip と decoder でも群の所属を保持する。最後の decoder は各群に 2ch を残す。
- 各サイズの head は入力画像とそのサイズまでの最終特徴から 1ch を出力する。
- 低解像度の予測は `[S, M, L]` として保持し、それぞれを対応する特徴群に挿入する。
  S に M / L の予測が入る経路はない。M / L は小型の内部状態も保持する。
- 学習中の 1 回の再帰 forward では全サイズの特徴計算を共有する。
  複数回 denoise する評価では入力系列がサイズごとに異なるため、切り出した各モデルの
  sampler を独立に実行する。

`nested_execution = "fused"` は、許された小さな kernel をゼロで埋めて結合し、
大きな Conv として計算する。`"blocks"` は同じ重みを群ごとの Conv として計算する。
この二つは同じモデルであり、出力・勾配の一致をテストしている。
ゼロ部分が GPU で自動的に省略されるとは仮定しない。実測結果は実験レポートを参照。

## 学習の条件

`L = (L_S + L_M + L_L) / 3` とし、各 `L_*` は従来の各段 loss の重みと正規化を使う。
`scale_loss_multiplier = 0` では最高解像度だけに直接 loss を置くが、低解像度予測を通る
勾配は維持する。再帰段数は `original_steps + dataset scale level`。

3 サイズが同じ画像を使うため、実効 batch は 16 のままで peak LR は
`1e-4 * sqrt(16) = 4e-4`。勾配蓄積後に一度 clipping・AdamW・EMA 更新を行う。
画像と強度マップを縮小してから各解像度で独立にノイズを生成し、従来の OT で割り当てる。

大型の loss は小型の共有重みも更新する。これは推論時に大型の出力を必要とすることとは
異なる。容量不足と共同学習の干渉を区別するため、同じ初期重みの独立 S を対照にする。
初期化はサイズ群ごとの乱数系列を使い、追加群の有無で S の初期重みが変わらない。

`train.model_seed` は初期化だけ、`train.seed` はデータ・ノイズ系列を制御する。
`data.split_seed` を固定すれば、分割は学習 seed に依存しない。
`data.final_test_count` で最終評価用の元画像を学習・開発評価から除外できる。
分割とデータ fingerprint は各 run の `dataset.json` に保存する。

## 再現する

README の手順で venv と必要な extras を準備した後、既存環境を `--no-sync` で使う。
GPU 環境では、以下のコマンドに `JAX_PLATFORMS=rocm,cpu` と
`XLA_PYTHON_CLIENT_PREALLOCATE=false` を指定する。CUDA では前者を `cuda,cpu` にする。

通常の学習:

```sh
uv run --no-sync lineartgen train --config configs/nested.toml --run-dir runs/nested --stop-after 2000
uv run --no-sync lineartgen train --run-dir runs/nested --resume --stop-after 8000
```

同じ crop と全 batch の OT 割り当てで、共同学習と独立 S を比較する:

```sh
.venv/bin/python scripts/compare_nested_training.py \
  --config configs/nested.toml --run-dir runs/nested-comparison --stop-after 2000 \
  --stop-file runs/nested-comparison-stop
.venv/bin/python scripts/compare_nested_training.py \
  --config configs/nested.toml --run-dir runs/nested-comparison --resume --stop-after 8000
```

`--stop-file` で指定したファイルが存在すると、全モデルの更新・評価・保存後に停止する。
再開前にそのファイルを除去する。`--baseline` を付けると現行 126,765 parameters の U-Net も
同じデータで学習する。再開時も同じフラグを付ける。
各モデルは通常の Trainer 形式で保存される。比較グループの再開位置は、全モデルが
保存された後に更新する `progress.json` で管理する。

使用条件に合わせた固定評価:

```sh
.venv/bin/python scripts/evaluate_nested.py --config configs/nested.toml \
  --cases runs/nested-evaluation --prepare-only
.venv/bin/python scripts/evaluate_nested.py --config configs/nested.toml \
  --cases runs/nested-evaluation --run-group runs/nested-comparison --step 2000
```

実画像の 6 scales × 9 条件と合成 16 条件、計 70 条件で denoise 1 / 2 / 5 回を比較する。
ここでは推論時と同じ独立 Gaussian を使い、OT によるノイズ選択をしない。
最終評価用の元画像は使わない。`--raw` で EMA ではなく更新中の重みも評価できる。
しきい値に基づく ink / boundary F1、領域面積、連結成分、保護領域の誤差と比較画像を保存する。
これらは診断用であり、線幅や位相を完全に評価する指標ではない。

## 切り出しと配布形式

```sh
uv run --no-sync lineartgen extract --run-dir runs/nested --subnet 0 --output runs/extracted-s
uv run --no-sync lineartgen extract --run-dir runs/nested --subnet 1 --output runs/extracted-m
uv run --no-sync lineartgen export --run-dir runs/nested --subnet 0 --output runs/s.onnx
```

`extract` は config・NumPy weights・由来情報だけを保存する。optimizer と大型専用の重みは含まない。
`lineartgen.models.serialization.load_inference(path)` で単独読み込みできる。

S は通常の UNet に変換でき、既存の 1ch `insert` の ONNX 契約に一致する。
M / L はそれぞれ 2 / 3ch の内部再帰状態が必要なため、現在は JAX での切り出しに対応する。
現行 ONNX runtime へ誤った 1ch として出力しないよう export で検証している。
Krita plugin と runtime は今回変更していない。

## ファイル構成

| 場所 | 役割 |
| --- | --- |
| `models/nested.py` | 一方向の Conv、U-Net、prefix 切り出し、S の通常 UNet への変換 |
| `models/serialization.py` | 推論用の config / weights の保存・読み込み |
| `diffusion.py` | サイズ別状態を保つ再帰、縮小後のノイズ、sampler |
| `training/steps.py` | 各サイズの目的関数、勾配蓄積、単一 optimizer 更新 |
| `training/trainer.py` | 通常の学習・評価・チェックポイント |
| `scripts/compare_nested_training.py` | データを共用した独立実験の比較 |
| `scripts/benchmark_nested.py` | GPU 常駐 batch による更新・推論時間の測定 |
| `scripts/evaluate_nested.py` | 固定条件の準備、運用条件での診断と比較画像 |
