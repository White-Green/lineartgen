# Krita graph preview — 作業記録

利用者の依頼は長期学習を完了してからKritaで試せるようにすること。
学習開始だけでは完了にしない。最後にWindows Python 3.13のCIとダウンロード可能なZIPを確認する。

## 学習

- 実験branch: `rewrite/graph-krita-20261019`、凍結学習commit `f05b683`。
- Run: `runs/graph-long-20261019`。IJ controlの累計16,200更新から84,000更新追加。
- detached driver PID 3072103、training PID 3073576。進行中の学習を止めない。
- `progress.json` / `operation.json` / `training-summary.json`で実際の進行・終了を確認。
- `scripts/graph_long_training.py`とplan bindingsの学習コード・設定・データは変更禁止。
- 同一batch・更新・全状態checkpoint復元のpreflightは完了。
- 追加2,000で標準25例のd3比較9ページを原寸確認。大きな後退なし、端の閉ループは残る。
- 追加12,000（累計28,200）でも同じ9ページを確認。36の左端は開き気味だが、
  38やscene1/5の閉ループ・二重線は残る。大きな改善とは判断せず継続。
- 追加12k時点で全40 parameter arrayが初期から変化、最大絶対変化0.208。
  最初の6kと次の6kの各深さの平均lossは下がっている。これは品質合格の根拠にはしない。
- 追加12k/36k/60k/84kで同様に比較。末尾でも改善中なら延長を検討する。

## 配布準備

別worktree `../jax-graph-runtime-20261019` / `rewrite/graph-runtime-20261019` は
既存runtime branchから作成。実験用のローカルデータに依存しないCIを維持する。
`../diffusion-drawing-jax` / `rewrite/graph-preview-20261019` はKrita用。
まだ両方とも最終重みを入れていない。初期16,200重みを配布しない。

- 6個の動的形状ONNX、JAX不要のNumPy再帰・graph描画、矩形入力、部分保護を実装。
- ONNX/JAXの20条件でraster最大誤差2.09e-6、1byte超差なし。
- ネットワーク重なり付きtileと描画tileを確認。最終重みでも照合を再実行。
- 初期重みの独立JAX reference fixtureを作成。最終重みに合わせて再生成が必須。
- Runtime全体229テスト、追加reference込みgraph16テスト成功。
- plugin30テスト成功。Linux Python3.12仮ZIPの同梱依存のみでのsmokeも成功。
- Windows/DirectML/Krita GUI実機の検証はまだしていない。
- 線分描画をbatch化し、初期1024px出力がpixel単位で一致することを確認。
  描画3.14秒→0.42秒、全体約7.46秒。renderer参照5テストとruntime16テスト成功。
- 33×17画像の幅1–2pxの短い線は初期モデルで消える。長期学習後に再確認する。
  長い細線や幅3px以上は出る。paddingを拡大しても変わらず、adapterの不具合ではない。

## 完了までの残作業

1. 最終raw/EMAと末尾checkpoint比較、幅・劣化・シーン確認。MSE最小で選ばない。
2. 選択後に未使用16合成例 `runs/graph-preview-holdout-20261019` を評価し実画像を確認。
3. 最終export、独立JAX fixture、20条件照合、小画像問題の再確認。
4. runtime assetsとREADME、出自・学習更新数・既知制約を更新し、最終デフォルトでCI相当の検査。
5. runtime commit/push → plugin submodule更新 → plugin commit/push。
6. Windows CI、ZIP再ダウンロード・hash・インポーター検査の成功を確認。利用者にartifactを案内。

pushは既に利用者が許可している。新たな確認は不要。
初期モデルのローカルsmoke ZIPを最終成果と呼ばない。

学習終了後の監査・raw/EMA書き出し・比較画像は`prepare_graph_delivery.py --wait`が自動実行。
watcher PIDは`runs/graph-long-20261019/delivery-watcher.pid`、状態は`delivery-preparation.json`。
出力は`runs/graph-export-final-raw`、`runs/graph-export-final-ema`、`runs/graph-final-comparison`。
選定・未使用16例への推論・公開は自動で行わない。各処理の終了コードとlogを確認する。
