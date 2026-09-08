# ローカル検証結果（2026-09-08）

## 作業場所

- lineartgen: `rewrite/jax`、`.worktrees/jax-rewrite`
- Diffusion Drawing: `rewrite/jax`、同じ `.worktrees` 内の `diffusion-drawing-jax`
- 元の main / feat/native-lineart の作業ディレクトリは変更していない。
- dataset は 223.04 GiB、7,149 ファイルを独立コピーした。名前・サイズ・mtime を全件照合し、
  元ファイルと共有する inode は 0。全画素の hash 比較ではない。

## 成功した検証

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
- 100,000 更新の長時間学習、全解像度でのスループット、画質の比較は未実施。
  初期設定の batch / LR / loss が最適と確認できたわけではない。
- 旧版と同じ Seed の画素一致は保証しない。数値比較は JAX と出力 ONNX の間で実施した。
- export は現在の fully convolutional U-Net 向け。新しい演算を追加したら変換と数値検証を更新する。
- model / data の変更は新しい run で開始する。厳密な再開は同じ設定・同じデータで行う。
- このエージェントの sandbox では asyncio のスレッド完了待ちが停止するため、
  Orbax と Krita ワーカーテストは通常プロセス権限で実行した。一般の学習に特権は不要。
