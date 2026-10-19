# ローカル検証結果（2026-10-19 更新）

## Graph試用版（runtime 0.3.0）

累計100,200更新のEMAを既定モデルとして検証した。
[学習・実画像評価・既知の失敗](graph-preview-20261019.md)を参照。
未使用例のベタ塗りの失敗を含め、品質目標をすべて満たしたとはしていない。

| 検証 | 結果 |
| --- | --- |
| 学習監査 | 追加84,000更新、深さ1〜6各14,000回、全記録・raw/EMAが有限、固定したソース・データのhash一致 |
| 最終JAX/ONNX照合 | 20条件、raster最大差2.921e-6、1byte超差0、分類反転0、tile差0、保護画素完全一致 |
| 目視 | raw/EMA比較30ページ、既知527例/1,180条件の指定57ページ、未使用16例の全32ページ |
| runtime・学習・export全体 | 最終graphを既定にした230テスト成功、360.61秒、既存のdeprecation warning 3件 |
| Ruff | checkとformat --check成功（155ファイル） |
| Krita UI・描画イベント・ワーカー | 30テスト成功、5.781秒 |
| 最終wheel・ローカルZIP | Linux Python3.12 / ORT1.29.0、同梱依存のみでBGRAと全6段の独立JAX参照を照合して成功 |

Windows ZIPはプラグインの`rewrite/graph-preview-20261019`のActionsでビルドする。
Windows CIはDirectML wheel同梱を確認し、利用可能なproviderで推論と公式importerを検査する。
実際のWindows GPU・Krita GUIでの操作確認とは区別する。
以下は旧UNetを配布していた時点の履歴で、当時の出自は現在の`legacy-model-source.json`にある。

Burn / JAX の学習更新と PNG 読み込みの実測比較は [training-speed.md](training-speed.md) を参照。

## 学習済み最大モデルのKrita同梱（2026-09-30、runtime 0.2.2）

同梱assetを旧epoch 1500から、JAXの`seed42_baseline`・100,000更新・EMAに変更した。
現在の学習済み候補で最大の126,765パラメータ、channels `[4,10,20,40]`・3 conv/block。
最終評価でseedを選び直さず、主実験のseed 42（data seed 47）を使用した。
出自・checkpoint・ONNXのhashはruntimeの `assets/model-source.json` にある。
ONNXは検証済みの実験exportとbyte単位で同一で、SHA256は
`2c3ad50d25e7fc977087386eef1cce3ba6d2a92daa4f707d59073cc9b271d415`。

| 検証 | 結果 |
| --- | --- |
| runtimeの入出力テスト | 11件成功。新しい同梱モデルでBGRA、padding、再帰、固定seed、alpha、入力エラーを確認 |
| Krita UI／描画イベント／ワーカー | 29件成功 |
| lockfile | runtimeと学習workspaceの両方で `uv lock --check --offline` 成功。runtime自身の版だけ0.2.2に変更し、依存の版は維持 |
| wheel | `uv build --project packages/lineartgen-runtime --wheel --out-dir dist` 成功。ONNX・metadata・出自を同梱 |
| 配布ZIP | Linux x64 CPython 3.10＋ORT 1.23.2、3.12＋ORT 1.29.0の両方で作成・同梱依存だけの推論に成功 |
| 実推論 | 33×17と145×129、複数回denoise、保護線画、alpha-only scribble、固定seed再現、入力エラーを確認。JAX／Flaxはimportしない |
| Krita公式インポーター | SHA256を固定したKrita 5.2.9 importerでcp310 ZIPの3,172ファイル、cp312 ZIPの1,391ファイルをインストール後に全件照合 |
| Windows設定 | 固定依存のDirectML exportに成功。Python 3.10・DirectMLの既存CI手順を維持 |

consumer worktreeの `dist/diffusion_drawing-linux-x64-cp310-0.2.2.zip` と
`dist/diffusion_drawing-linux-x64-cp312-0.2.2.zip` が今回のローカル成果物。
両ZIPの同梱ONNX hashは上記と一致する。依存wheelのhashもruntime lockfileと全件一致。
ビルドの記録は `dist/baseline-100k-0.2.2-build.json` に保存した。
Windows DirectMLおよびKrita GUI上の動作はこのLinuxホストでは実行していない。
Windows ZIPはconsumerの既存GitHub Actions、またはREADMEのWindowsビルド手順で作成する。

モデルの品質評価は[最終報告](completion-results-20260930.md)を参照。
実験終了後に同梱assetとruntime版を更新したため、以前の実験のソースhash監査を再実行する際は
実験rootに保存したsource archiveまたは対応commitを使う。

## 作業場所

- lineartgen: `rewrite/jax`、`.worktrees/jax-rewrite`
- Diffusion Drawing: `rewrite/jax`、同じ `.worktrees` 内の `diffusion-drawing-jax`
- 元の main / feat/native-lineart の作業ディレクトリは変更していない。
- dataset は 223.04 GiB、7,149 ファイルを独立コピーした。名前・サイズ・mtime を全件照合し、
  元ファイルと共有する inode は 0。全画素の hash 比較ではない。

## 推論の縮小→ノイズ付加への修正（2026-09-09、runtime 0.2.1）

画像と強度マップを先に各解像度へ縮小し、その後で各段に独立したノイズを一度だけ加える。
複数ステップでは段ごとの状態を更新して引き継ぐ。Krita の NumPy runtime と
JAX の検証用 sampling の両方を変更した。学習側の OT 割り当ては変更していない。

| 対象 | 結果 |
| --- | --- |
| テスト全体 | `JAX_PLATFORMS=cpu pytest -q`: 58 passed / 27.20 秒。学習再開・構造変更時の ONNX export も含む |
| 推論の回帰テスト | 縮小後のノイズ付加、マスク境界と強度 0、段ごとに一度のノイズ生成、複数ステップの各段の状態、小さい段の clean 予測の受け渡しを検証 |
| ノイズ分布 | NumPy / JAX とも、低解像度でも生成ノイズの分散が約 1 で、縮小した上位段のノイズと独立であることを標本で確認 |
| JAX / runtime | 同じノイズを与えた 1 段・3 段のマスク付き複数ステップ推論で、JIT した JAX と NumPy が絶対許容誤差 1e-6 以内で一致 |
| GPU 推論 | `runs/burn-design-six-levels` の step 1 EMA、実画像の 512² crop、6 段・3 ステップ、一部強度 0 のマスクで実行。出力が有限で固定 Seed の再実行が完全一致。`runs/resize-before-noise-gpu.json` |
| Krita 連携 | UI / 描画イベント / ワーカーテスト 29 ケース成功 |
| 配布 ZIP | runtime 0.2.1 を同梱した Linux x64 CPython 3.12 / 3.10 の両方で、同梱依存だけを使う推論に成功。145×129、部分マスク・3 ステップの再帰推論と固定 Seed の再現性も確認 |

ZIP は consumer worktree の `dist/diffusion_drawing-linux-x64-cp312-0.2.1.zip` と
`dist/diffusion_drawing-linux-x64-cp310-0.2.1.zip`。ONNX の入出力形式・同梱モデルは変更していない。
従来版との多段推論の画素一致や、変更による画質の改善を確認したものではない。

## Burn の学習設計を戻した後の検証（dd7aa5e）

| 対象 | 結果 |
| --- | --- |
| テスト全体 | `JAX_PLATFORMS=cpu pytest -q`: 50 passed / 25.85 秒 |
| 再帰と各段 loss | scale 0〜5 で 1〜6 段、最小サイズ・stop_size による上限、各段の重みと正規化、最大解像度の loss から小さい段への勾配を検証 |
| LR | 実効 batch による平方根スケーリング、同じ実効 batch の異なる microbatch 分割、絶対値の最低 LR を検証 |
| ノイズ割り当て | Burn の元式とコスト一致、mask の 0 領域、矩形・正方形・同点の厳密最適性、候補の一意性、SciPy / Optax の両方を検証 |
| データ | 標準データ 1,019 元画像の全 6 scale の寸法比が一致。subset の明示的な scale level、誤った寸法の検出、入力列の途中再開を検証 |
| 学習再開 | 厳密ノイズ割り当てと可変段数を含め、連続実行と中断→再開のモデル・optimizer・EMA 全 leaf が完全一致（CPU の合成データ） |
| 実データ GPU 学習・再開 | crop 32、scale 0 / 1、batch 2 × 蓄積 2、pool 8、2 更新→保存→再開→4 更新。`runs/burn-design-verified` |
| 実データ GPU 6 段再帰 | images_32x の scale level 5、512→256→128→64→32→16、batch 2 × 蓄積 2、pool 8、1 更新・検証画像・保存。`runs/burn-design-six-levels` |
| GPU 保存 → CPU export | 新設計で学習した step 4 の EMA を ONNX 化。16² / 32×48 / 128² で JAX / ORT 最大絶対誤差 1.79e-7 |
| ノイズ割り当ての GPU 測定 | 512² / batch 16 / pool 256。コスト計算 42.99→12.75 ms、SciPy callback 込み 12.82 ms。[条件と詳細](noise-matching.md) |
| 静的検査・lock | Ruff lint / format、git diff --check、uv lock --check --offline 成功 |

GPU 学習は `JAX_PLATFORMS=rocm,cpu` と `XLA_PYTHON_CLIENT_PREALLOCATE=false` を使用。
コスト計算・モデル演算は ROCm、厳密 assignment の solver は SciPy の CPU callback。
学習 smoke の pool は 8。標準 pool 256 での測定はノイズ割り当て単体であり、
長時間学習の速度・画質を検証したものではない。

## 初回 rewrite で成功した検証（9d41a36）

以下の学習 run は設計を戻す前のもの。学習設定を戻した後の結果は dd7aa5e の表を参照。
学習設計の修正（dd7aa5e）では Krita runtime と配布処理に変更を加えていない。

| 対象 | 結果 |
| --- | --- |
| JAX 側のテスト | 21 ケース。TIFF の小画像タイル制約を修正後、該当 8 ケースを再実行して成功。NNX 保存 API 調整後の再開一致も再実行して成功 |
| 学習再開 | synthetic dataset で連続実行と中断→再開のモデル・optimizer・EMA 全 leaf が完全一致 |
| 実データ CPU 学習 | crop 32、2 段再帰、batch 2 × 蓄積 2、2 更新→保存→再開→4 更新、検証画像生成、EMA export |
| 実データ AMD GPU 学習 | `RocmDevice(id=0)`、crop 32、2 更新・検証・保存。`runs/smoke-rocm` |
| 標準サイズの GPU 演算 | crop 512、3 段再帰、batch 4 × 蓄積 4、1 更新・検証・保存。元画像は images 解像度を使用。`runs/smoke-rocm-512` |
| GPU → CPU export | GPU で保存した checkpoint を CPU で復元し ONNX へ変換。JAX / ORT 最大絶対誤差 1.79e-7 |
| 既存学習済み重み | epoch 1500 MPK を Python で取り込み。16×16、32×48、128×128 で JAX / ORT 最大絶対誤差 1.32e-6 未満 |
| 構造変更 | channels `[3, 6, 12]`、2 conv/block で export・数値比較・145×129 BGRA 推論成功 |
| runtime 契約 | BGRA、alpha-only scribble、padding、縦横比の大きな入力、再帰、固定 Seed、入力エラー、JAX を import しないこと |
| 前処理 | 元画像単位の分割、入力列の途中再開一致、PNG と tiled TIFF の crop / alpha の完全一致 |
| Krita 連携 | 既存 UI / 描画イベント / ワーカーテスト 29 ケース成功 |
| 配布 ZIP | Linux x64 CPython 3.12 + ORT 1.29.0、CPython 3.10 + ORT 1.23.2 の両方で同梱依存だけを使う推論に成功 |
| ZIP インポート | Krita 5.2.9 の SHA-256 固定公式 importer で cp312 ZIP の全 1,390 ファイルが一致 |
| 静的検査 | Ruff lint / format、git diff --check 成功 |

GPU は AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`)、ホスト ROCm 7.2.1。
JAX / jaxlib / jax-rocm7-plugin 0.11.1、Flax 0.12.9、Optax 0.2.8、Orbax 0.11.36。
このホストでは ROCm の PCIe 帯域検出 warning が出るが、演算・保存は完了した。
学習確認にシステムの ROCm 更新は行っていない。

作成したローカル ZIP は consumer worktree の `dist/` にある。
Windows 用 ZIP と同名にせず、`diffusion_drawing-linux-x64-cp310.zip` と
`diffusion_drawing-linux-x64-cp312.zip` に分けた。

## 未検証・制約

- Windows 実機の DirectML 推論速度と Krita GUI 上の描画は、この Linux ホストでは未検証。
  Windows CI に wheel 取得・DirectML provider の実推論・公式 importer の検証を用意した。
  CI をリモートで実行したという意味ではない。
- 100,000更新・3 seedの長期学習と品質・速度の比較は[最終報告](completion-results-20260930.md)で完了。
  最小十分サイズは未確定であり、batch / LR / loss が最適と確認できたわけではない。
- 旧版と同じ Seed の画素一致は保証しない。数値比較は JAX と出力 ONNX の間で実施した。
- export は現在の fully convolutional U-Net 向け。新しい演算を追加したら変換と数値検証を更新する。
- model / data の変更は新しい run で開始する。厳密な再開は同じ設定・同じデータで行う。
- このエージェントの sandbox では asyncio のスレッド完了待ちが停止するため、
  Orbax と Krita ワーカーテストは通常プロセス権限で実行した。一般の学習に特権は不要。
