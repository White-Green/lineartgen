# 線画品質の現在地（DF完了）

複数の許容清書から一つを選ぶlossで、灰色の平均を出す傾向は減った。
候補固定・誤差関数・断面・飽和勾配の比較でも、細線の粒状感と枝の不安定さは残る。
透明度・ぼかし・局所的な揺れを変えるDBは難しい入力への追従を改善したが、
薄い細線の灰色と黒内部の模様が残り、製品未採用。36枚＋512px6枚を等倍確認済み。

DCでsource-onlyを同条件学習し、粒状感は減ったが、滑らかな灰色帯・輪郭リングは残った。
16枚＋512px6枚を確認。製品未採用。

DDで入力濃度と出力線幅を切り離した2pxスタイルを比較し、細線の濃さと一部の灰色帯が改善した。
薄い入力・交点・黒の輪郭リングは残る。16枚＋512px6枚を確認し、製品未採用。
DEで共有CNNへの縮小段情報を比較したが、明確な画質改善はなかった。
16枚＋512px6枚を確認し、不採用。
DFで固定スタイルに学習された出力priorを加えたが、画質はほぼ同じだった。
22枚を確認し、不採用。判別器は灰色平均と清書を区別でき、Gへの勾配は既存lossの平均7.45%。
次は学習済みG/Dからprior係数.2と2を比較する。

- [DF: 固定清書スタイル＋出力prior](style-prior-results-20261013.md)

- [DE: 縮小段情報](pyramid-family-results-20261013.md)

- [DD: 清書スタイルの比較](style-family-results-20261013.md)

- [DC: source-only比較](source-family-results-20261013.md)

- [DB: 下描き劣化の拡張](augmented-family-results-20261013.md)
- [DA: 境界勾配の延長](linear-family-results-20261013.md)
- [CZ: 区間loss](interval-family-results-20261013.md)
- [CY: 経路断面](path-profile-results-20261013.md)
- [CW/CX: 候補固定×誤差関数](robust-family-results-20261013.md)
- [CV: データ数](family-expansion-results-20261013.md)
- [CU: 許容清書から一つ選ぶ](plausible-family-results-20261013.md)

合成データのみ。Krita・配布・封印データは変更していない。
