# 入力保持とノイズ割当ての比較 — 2026-09-11

入れ子構造の成立を確認した後、幅・パラメータ数を固定して学習方法を比較する実験。
4 条件を各 2,000 更新で評価し、動作上の異常がなければ各 10,000 更新まで継続する。
現在は実行中。品質の採否は、以下の固定評価の結果が出てから判断する。

## 比較する条件

| run | 出力 | ノイズ割当て |
| --- | --- | --- |
| `current` | 完成画像を予測 | Gaussian 256 候補から 16 個を選ぶ現行 exact OT |
| `residual` | `clip(x_t + sqrt(t) * F(x_t, t, insert), -1, 1)` | current と同一 |
| `ot_full` | current と同一 | 16 候補をすべて使う一対一の exact OT |
| `independent_noise` | current と同一 | Gaussian を画像と独立に対応させる |

全条件で S/M/L = 5,933 / 10,206 / 17,893 parameters。条件間で重み・optimizer・EMA を共有しない。
各条件の内部では S/M/L の特徴を共有し、3 サイズの loss を平均する。
同じ 16 枚を 3 サイズで使っても、実効 batch は 16 のまま。

候補は共通の乱数で 256 個生成する。`ot_full` と `independent_noise` は、画像を見る前に
先頭 16 個を取り出してから割当てる。配列形状による乱数列の差を比較に混入させない。
この設定は `matching.generation_pool_size` として checkpoint の config に残す。
全使用 OT は候補集合の経験分布を保つが、画像や mask を条件にした独立性を意味しない。
現行の輸送コスト、solver、Gaussian 生成器そのものは変えない。

残差出力はネットワークの生出力を clip する前に入力へ加える。
`t=0` ではモデルの修正量が 0 になり、正規化された入力を保持する。
JAX sampler でも、その画素の代数的に相殺する計算による丸め誤差を避ける。
direct 出力の動作は維持する。Krita plugin と runtime は変更しない。

## 固定した条件

- `configs/nested.toml` 相当、crop 512、FP32、microbatch 4 × 蓄積 4。
- 元画像を train 891 / development 64 / final test 64 に分割。final test は使わない。
- 全条件が同じ crop・augmentation・強度マップ・Gaussian 候補を使用。初期パラメータは完全一致を assert。
- `original_steps + dataset scale_level` の 1〜6 段再帰と、再帰を通る勾配を維持。
- 各段 loss の係数は従来どおり。`scale_loss_multiplier=0` なので最高解像度だけに直接 loss。
- base LR 1e-4 × sqrt(16) = peak 4e-4。warmup 500、schedule の終点は最初から 100,000 更新。
- 画像と強度マップを縮小してから、解像度ごとに独立 Gaussian を加える。
- 学習 loss の条件は異なるので、validation loss は全条件で共通の独立 Gaussian 入力を使用。
  運用評価も独立 Gaussian で行い、学習時の OT のコストと復元品質を区別する。
- 4 条件の更新順序を毎更新ローテーションする。input / preparation / model update の時間を記録。

## 固定評価

計 434 条件を、3 サイズ × 4 手法 × denoise 1 / 2 / 5 回 × raw / EMA で評価する。
1 checkpoint あたり 31,248 件。S の比較画像も固定した条件で保存する。

| 集合 | 条件数 | 内容 |
| --- | ---: | --- |
| 前回の評価 | 70 | 実画像 54 + 合成 16。白地に近い crop や既知の失敗例も維持 |
| 同じ crop の条件比較 | 108 | 6 scales × 2 元画像 × 9 条件。各 crop に強度 .1/.3/.5/.7 × 全体/部分 + 強度 0 |
| 正解形状を持つ合成 | 256 | 8 線種 × 8 幅・角度の組合せ × 4 ノイズ条件。16 条件は強度 0 |

追加実画像は 12 元画像・12 crop。同じ crop を使う条件間の比較を可能にするためのセットであり、
108 枚の独立した元画像ではない。正解の線領域が 0.1% 以上ある crop を、モデル出力を見ずに選んだ。
合成の線種は直線・細線・曲線・角・分岐・近接平行線・意図した隙間・入り抜き。
学習データへ合成を追加する実験は、この比較には含めない。

評価は F1・画素 MAE に加え、以下を個別に保存する。

- 正解中心線の法線上で、アンチエイリアスを含む濃度を積分した幅の誤差。
- 同じ濃度分布の位置ずれと、その変動。線の濃度が足りない点は位置誤差から除き、欠落率を別記。
- 正解ではつながる点対の切断率、つながらない点対の誤接続率。
  閾値は正規化値 -0.25 / 0 / 0.25。正解の raster 自体で成立しない問い合わせは除外し、その数も記録。
- 保護領域の平均・最大変化、mask 境界の両側約 3px の変化。
- ノイズを加えた領域だけの F1・MAE。入力保持の改善と、ノイズ領域の復元を分けて見る。
  追加実画像ではノイズ領域に正解の線が 0.1% 以上ある条件の集計も別記する。
- 白地・identity・線を含む実画像・合成を分けた集計。

これらは診断指標であり、線幅・曲率・位相を完全に評価するものではない。
切断、橋、幅増加、位置ずれ、線の消失を正解へ加え、対応する指標が悪化することをテストした。
接続の改善と引き換えに意図した隙間を埋める変更を、平均 F1 だけで採用しない。

## 実行と継続の判定

run root は `runs/methods-20260911/`。このディレクトリは Git 管理対象外。
全 93 テスト、4 条件の連続実行と停止・再開での model / optimizer / EMA の完全一致を確認した。
主比較の前にノイズ領域の評価を追加し、その分離を確認する 2 テストも成功した。
元画像・合成形状を単位とした bootstrap のテストを含め、追加した評価関連の 15 テストが成功した。
残差出力の S は dynamic ONNX 出力も検証済み。

2,000 更新の継続判定は、全件評価完了、残差出力の保護画素の誤差 0、
全サイズの raw・5 回推論の F1 が追加実画像・合成でともに 0.30 以上とした。
この閾値は崩壊を検出するための暫定値であり、十分な線画品質の基準ではない。
順位が低いだけの条件は落とさない。異常があれば自動継続を止めて原因を調べる。

2,000 と 10,000 更新の raw / EMA を、optimizer のない独立した archive として保存する。
後続学習で古い checkpoint がローテーションされても、両時点を再評価できる。
最終候補の組合せ実験・30,000 更新以降・複数 seed は、この比較結果に基づいて選ぶ。

```sh
# README の ROCm 環境を用意し、JAX worktree から実行する。
export JAX_PLATFORMS=rocm,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PWD/runs/methods-20260911/jax-cache"

.venv/bin/python scripts/evaluate_training_methods.py \
  --config runs/methods-20260911/config.toml \
  --cases runs/methods-20260911/evaluation --prepare-only

# config.toml と評価入力の準備後、2,000 → 評価 → 10,000 → 評価を実行・再開する。
.venv/bin/python scripts/run_training_methods.py --root runs/methods-20260911
```

最初の学習を別に開始している場合は `--attach-pilot` で完了を待って接続できる。
`runs/methods-20260911/stop` を作ると、全条件の更新・保存が済んだ境界で停止する。
再開前に stop ファイルを除去する。

終了後は、学習済み重みで切り出し・非混入・保護画素を再検証し、
snapshot と checkpoint の一致、全条件の optimizer step が 10,000 であることを確認する。
ONNX は各手法の S を raw / EMA とも出力し、通常の dynamic サイズに加えて
実際の 512px 入力と下位段の予測を使って照合する。

```sh
.venv/bin/python scripts/verify_training_methods.py \
  --root runs/methods-20260911 --step 10000 --check-checkpoints
JAX_PLATFORMS=cpu .venv/bin/python scripts/verify_training_methods.py \
  --root runs/methods-20260911 --step 10000 --exports-only
.venv/bin/python scripts/summarize_training_methods.py \
  --root runs/methods-20260911 \
  --output benchmarks/training-methods-20260911.json
```

集計は全評価ケースの組合せ、入力・archive・ONNX の SHA、checkpoint の更新数を検査する。
2,000 更新だけの中間集計には `--through 2000` と別の出力先を指定する。
現行との差の区間推定は、実画像では元画像、合成では形状を単位として 5,000 回 bootstrap する。
同じ元画像の mask・強度違いを独立した標本として扱わない。
これは学習済みモデルを固定した開発セット上の探索的な比較であり、
学習 seed による変動や多重比較の補正は含まない。

| ファイル | 内容 |
| --- | --- |
| `training-source.json` | 学習開始時のコード SHA と比較条件 |
| `training/*/config.json` | 条件ごとの正確な設定 |
| `training/progress.json` | 全条件で保存済みの更新位置 |
| `training/timings.jsonl` | 入力・ノイズ準備・各条件の更新時間 |
| `noise-assignment.json` | 固定実画像での、解像度・強度・mask・seed 別のノイズ分布診断 |
| `evaluation/cases.json`、`case-*.npz` | 元画像・crop・乱数 seed・固定入力・形状の正解 |
| `evaluation/step-*-{raw,ema}/metrics.json` | 全評価値と条件別集計 |
| `snapshots/step-*/手法/{raw,ema}/` | checkpoint の保持数に依存しない推論用重み |
| `verification/step-10000-raw-ema.json` | 学習済み prefix の非混入・保存復元・checkpoint の照合 |
| `extracted/step-10000/手法/{raw,ema}/{0,1,2}/` | 各サイズの独立した推論用重み |
| `exports/step-10000/validation.json` | 全条件の S の ONNX 照合と SHA |
| `pilot-gate.json`、`workflow-status.json` | 継続判定と実行状態 |

測定結果は学習・評価の完了後に追記する。
