# 線画品質の現在地（DA完了）

複数の許容清書から一つを選ぶlossで、灰色の平均を出す傾向は減った。
その後のデータ数・候補固定・誤差関数・断面・飽和勾配の比較でも、細線の粒状感と枝の不安定さは残る。
CZ/DAは欠損率を改善しても灰色縁が増え、製品未採用。単一劣化のloss探索はいったん区切る。

次はopacity・blur・局所的な揺れを変えた合成下描きへの汎化を比較する。
16chの自己再帰、全段loss、sqrt(batch)LR、縮小後ノイズ、Krita I/Oを維持。

- [DA: 境界勾配の延長](linear-family-results-20261013.md)
- [CZ: 区間loss](interval-family-results-20261013.md)
- [CY: 経路断面](path-profile-results-20261013.md)
- [CW/CX: 候補固定×誤差関数](robust-family-results-20261013.md)
- [CV: データ数](family-expansion-results-20261013.md)
- [CU: 許容清書から一つ選ぶ](plausible-family-results-20261013.md)

合成データのみ。Krita・配布・封印データは変更していない。
