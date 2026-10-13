# 線画品質の現在地（DC完了）

複数の許容清書から一つを選ぶlossで、灰色の平均を出す傾向は減った。
候補固定・誤差関数・断面・飽和勾配の比較でも、細線の粒状感と枝の不安定さは残る。
透明度・ぼかし・局所的な揺れを変えるDBは難しい入力への追従を改善したが、
薄い細線の灰色と黒内部の模様が残り、製品未採用。36枚＋512px6枚を等倍確認済み。

DCでsource-onlyを同条件学習し、粒状感は減ったが、滑らかな灰色帯・輪郭リングは残った。
16枚＋512px6枚を確認。製品未採用。次は入力濃度と出力線幅の結び付きも再考する。

- [DC: source-only比較](source-family-results-20261013.md)

- [DB: 下描き劣化の拡張](augmented-family-results-20261013.md)
- [DA: 境界勾配の延長](linear-family-results-20261013.md)
- [CZ: 区間loss](interval-family-results-20261013.md)
- [CY: 経路断面](path-profile-results-20261013.md)
- [CW/CX: 候補固定×誤差関数](robust-family-results-20261013.md)
- [CV: データ数](family-expansion-results-20261013.md)
- [CU: 許容清書から一つ選ぶ](plausible-family-results-20261013.md)

合成データのみ。Krita・配布・封印データは変更していない。
