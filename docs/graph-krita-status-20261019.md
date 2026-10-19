# Krita graph preview — 2026-10-19

累計100,200更新（親16,200＋追加84,000）の学習を正常終了した。
追加学習の時間は27,533秒。深さ1〜6は各14,000更新、全記録とraw/EMAは有限。
凍結学習ソースは`f05b68387a40202c0adc5b900c11eb8eb740754d`。
runは`runs/graph-long-20261019`。全状態checkpoint、最終raw/EMA、ソースarchiveを保持する。
学習ソース・設定・データのbindings監査も通過した。

## 配布重みと評価

選定は最終EMA、SHA256 `09e4f6d1a775cacc3e0b6a75805cc2912b3dc9a33646ff48eb5018599ca09aa8`。
終盤のd3画像変化は小さく、raw/EMA比較30ページでも大きな優劣はなかった。
既知527例・1,180条件の指定57ページを目視したうえで選定を固定した。
未使用16例はその後にBGRAから64条件を実行し、全32ページを目視した。
画素MSEによる選定、未使用例からの重みの選び直し・調整はしていない。

薄い曲線、交差、太い線からの中心線などは描けるが、近接線の閉じ込み、余分な点、局所的な太りが残る。
新しい寸法のベタ塗りは穴だらけになる、または中心線だけになる大きな失敗があった。
これは品質目標達成ではなく、十分な更新を行ったモデルをKritaで試すための区切りである。

- [更新数と重みの監査](benchmarks/graph-training-20261019.json)
- [選定記録](benchmarks/graph-selection-20261019.json)
- [既知例の目視所見](benchmarks/graph-broad-review-20261019.json)
- [未使用例の目視所見](benchmarks/graph-heldout-review-20261019.json)

## 配布と検証

配布runtimeは`rewrite/graph-runtime-20261019`、consumerは
Diffusion Drawingの`rewrite/graph-preview-20261019`。
研究用のtestsにはローカルの実験runが必要なものがあるため、この研究記録のpushはCIを起動しない。
配布runtimeは実験データから独立したCIを持ち、最終graphを既定にして全230テスト成功、
Ruff/checkとformat成功。プラグインの30テストとLinux Python3.12のZIP smokeも成功した。
最終JAX/ONNX20条件ではraster最大差2.921e-6、1byte超差・分類反転0、tile差0、保護領域一致。
全6段の独立JAX参照fixtureも同じ最終checkpointから再生成した。

推論はJAXを同梱せず、6個のONNXとNumPy graph rendererを使用する。
Windows Python3.13のCIを実行し、artifactを取得できるところまで確認する。
Windows GPU実機とKrita GUIの評価は利用者による試用が必要。
最終モデル・入力別出力・失敗画像・配布仕様はruntime branchの`docs/graph-preview-20261019.md`に集約する。
