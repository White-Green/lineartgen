# CZ: 区間lossは欠損を減らすが、灰色の縁を増やした

16ch/15,809paramsを固定し、CY profile raw6000からclipped/interval各6,000更新。
数値ソース `59558ec`。同一512合成例、バッチ・ノイズ・段数・LRを対応させた。
**intervalは未採用。** 欠損率改善だけでは線画らしさの改善と判断しない。

|split|方法|全経路断面誤差|断面欠損率|接続切れ率|許容帯外ink比|
|---|---|---:|---:|---:|---:|
|new_probe|clipped|0.148608|0.005270|0.016034|0.000286|
|new_probe|interval|0.173872|0.000368|0.013440|0.001380|
|new_train|clipped|0.134800|0.003758|0.020101|0.000218|
|new_train|interval|0.160489|0.000245|0.016512|0.001405|
|old_probe|clipped|0.139556|0.004330|0.025352|0.000585|
|old_probe|interval|0.166976|0.000000|0.021679|0.002156|
|old_train|clipped|0.147939|0.011275|0.024070|0.000139|
|old_train|interval|0.177294|0.001716|0.018195|0.001444|

全16等倍シートを確認。細い枝(old_train24)はまだ不安定。線の縁・端点に薄い灰色が増え、
old_probe36/new_probe36の近接線も明瞭に分離しない。広い黒内部の白粒は減るが外周に灰色縁。
白紙と保護領域は維持。近接線の数が変わること自体は入力の曖昧さも考慮するが、
今回の灰色・粒状の縁は自然な清書として改善とは言えない。

- [枝・近接線](../runs/interval-family-20261013/evaluation/native-review/old_train-whole05-1.png)
- [未学習形状](../runs/interval-family-20261013/evaluation/native-review/new_probe-whole05-1.png)
- [全評価](../runs/interval-family-20261013/evaluation/summary.json)

CPU3テスト・ruff合格。6更新GPU再現は許容誤差内で、bit exactではない
(state最大差clipped4.10e-8、interval3.35e-8、事前atol1e-7/rtol1e-5)。
全状態保存復元・12raw/EMA exportは完全一致。4,608評価行・1,536参照行、全保護画素一致。

区間Charbonnierは範囲外の距離が大きいと、表示上の誤差が小さいAA画素でも勾配が大きくなる。
例えばraw opacity=-.5、target=.01では勾配の大きさが約.995である。
次は有効範囲の境界でのCharbonnierの傾き(.196)をそのまま範囲外へ延長する。
範囲内のloss、全候補の最小選択、推論clipは維持し、微小AAと重大な欠損の修正強度を区別する。
これは灰色の原因が完全に証明されたという意味ではなく、追加の制御比較で検証する仮説。

固定opacity・有限候補の診断。Kritaへの採用や実際の下描きでの品質確認ではない。
