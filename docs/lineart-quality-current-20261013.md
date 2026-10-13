# 線画品質の現在地（DJ完了）

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
DGでprior係数.2と2を比較。Gへの勾配は58.4%に増えたが線画の欠陥は残り、薄い/粗い黒ベタのリングが悪化。
22枚を確認し不採用。
DHで同容量の局所コントラスト特徴を比較。薄い線は少し濃くなったが、灰色側線・交点・黒ベタのリングは残る。
22枚を確認し不採用。次は新しい初期値から同じパラメータ数で参照範囲を比較する。

DIは新しい初期値からplain/dilated各18,000更新を完了し、22枚を等倍確認。
dilatedは薄い/粗い黒ベタと一部の滑らかさを改善したが、灰色側線・交点・輪郭リングは残る。
両者とも不採用。次は端点/交点の局所領域をlossで明示する比較を行う。

DJはDI plain raw18000から、端点/接合点17x17領域の追加lossの有無を各6,000更新で比較した。
22枚を確認したが、交点の切れ・灰色側線・輪郭リングは残り、不採用。
次は入力の局所濃度を全解像度で正規化してから縮小し、特徴と出力anchorの両方を変える。
同じ16chを新しい初期値から学習し、同条件のDI plainを対照として再利用する予定。

- [DJ結果](structural-family-results-20261013.md)
- [DJ計画](structural-family-plan-20261013.md)
- [DI結果](fresh-context-family-results-20261013.md)
- [DI計画](fresh-context-family-plan-20261013.md)
- [開始前の修正](fresh-context-preflight-repair-20261013.md)

- [DH: 局所コントラスト特徴](local-contrast-family-results-20261013.md)

- [DG: priorの重み](style-prior-weight-results-20261013.md)

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
