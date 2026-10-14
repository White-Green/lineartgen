# Latest update: DT completed, 2026-10-14

[DT results](field-sign-balance-results-20261014.md): matched6,000-update continuations show that balanced inside/outside supervision substantially reduces line erasure and fill holes.9,408 outputs and28 native pages reviewed. Wide rough strokes still produce black bars/fragments, and some junctions/parallel curves break. Partial progress, not product adoption. Next: measure the remaining field errors and regional gradient weights before choosing the next loss change. Reserved fresh confirmation untouched.

# Latest update: DS completed, 2026-10-14

[DS results](contour-coverage-results-20261014.md):24 known-geometry cases pass the renderer prerequisite.384 learned-field outputs and16 native learned pages show persistent line erasure; contour rendering alone cannot recover missing geometry. Next: explicit balanced inside/outside geometry supervision versus matched continued-training control. No adoption or reserved fresh evaluation.

# Latest update: DR completed, 2026-10-14

[DR results](subpixel-field-results-20261014.md): two geometry-field models trained6,000 updates each;9,408 outputs and28 native pages reviewed. Major line erasure and fill holes in both. No adoption; fresh confirmation untouched. DS compares rendering of the same fields before changing supervision.

# Latest update: DQ rendering oracle completed, 2026-10-14

[DQ results](opaque-geometry-results-20261014.md): known geometry renders cleanly across24 cases and six scales;8 native pages plus one enlarged coarse display inspected. This is a rendering prerequisite, not a learned predictor. DP learned models remain rejected. Next bounded direction: learn a subpixel geometry field with distance/gradient supervision while preserving the recursive training conventions.

# Latest update: DP completed, 2026-10-14

[DP coherent-profile results](coherent-path-results-20261014.md): two matched 6,000-update arms, 9,408 outputs and 28 native comparison pages reviewed. Gray secondary bands, crossing cuts, wide partial-mask erasure and damaged fill corners remain. No adoption; fresh confirmation untouched. Direct pixel optimization can yield clean opaque solutions, but the trained CNN does not reproduce this improvement. Next: oracle and gradient validation of opaque geometry rendering before training a geometric predictor.

# 線画品質の現在地（DO完了）

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
同じ16chを新しい初期値から学習し、同条件のDI plainを対照として再利用する。

DKは全解像度sourceを正規化してから縮小する構成を18,000更新した。
標準/劣化16枚・512px6枚・runtime比較6枚とoracle2枚を等倍確認。
薄い線とfaint/rough黒ベタのリングを大きく改善したが、近接線の灰色帯は残り一部で悪化。製品未採用。
元の100k EMAとのruntime段数比較では、両研究モデルが合成下描きの線をはるかに明瞭に保持した。
構成・学習・前処理全体の比較であり、容量だけの因果比較ではない。
最終の候補選択は108例中100例以上が既に中央で、中央固定curriculumはまだ実施しない。
次は曖昧な近接線を一本へ整理する候補もlossに含める比較を検討する。

- [DK結果](full-source-normalization-results-20261013.md)
- [DK計画](full-source-normalization-plan-20261013.md)

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

512pxのscene1/7には意図的なgapがある。旧レビューの「縦線の途切れ」は広すぎたため、
[幾何を確認した補足](scene512-interpretation-20261013.md)で区別した。

DKのactive anchorは正規化sourceなので、t=0ではrawを保持するがt→0+の連続性は保証しない。
実用採用時にはこの強度の挙動も確認が必要。

追加で、実際のruntimeと同じ128px停止（512pxでは3段）の比較を完了した。
元の最大100k EMA、DI raw、DK normalizedを同じuint8入力で比較した。出力thresholdなし。
元モデルの入力倍率1.1と、合成学習で使った研究モデルの1.0は区別して記録する。
研究configには倍率1.1が継承されているが、研究scriptは1.0のデータで学習するため、将来の配布時には整合が必要。

DLは入力が支持する近接線に一本清書の候補を加え、各6,000更新を完了した。
25枚を等倍確認。scene5の副線は少し減ったが、太い灰色帯と交点の欠陥は残り、製品未採用。
許可されたprobe7例すべてで一本線costが最小となるため、候補不使用だけが原因ではない。
wide種別だけ16–32pxの太線教師を残している点を次の比較対象とする。
[DL結果](merge-family-results-20261013.md) / [DL計画](merge-family-plan-20261013.md)

DMは同じ15,809パラメータで参照範囲を87pxに広げ、wide教師を保持/2pxへ統一する各18,000更新を完了。
33枚を等倍確認した。thinwideで一部の副線は減ったが、太い下描きが薄い残骸や白紙になり、blackの角も悪化した。
単純な参照範囲拡大と教師統一だけでは不十分で、両群不採用。
次はlossの正規化で細線消失を過小評価していないか、縮小段ごとの被覆率をモデルが区別できるかを切り分ける。
[DM結果](wide-style-context-results-20261014.md) / [DM計画](wide-style-context-plan-20261014.md)

DNは縮小段情報と清書inkによるloss正規化の2×2比較を完了（新規3群各18,000更新、DM対照再利用）。
33枚を等倍確認。清書ink正規化で太い入力から中央の線を残す効果はあったが、欠落・端の残骸・灰色側線が残る。
段情報だけでは太線の消失が悪化。一部の近接線では灰色副線が再発し、blackの角や交点も解決せず、全群不採用。
別seedの512px16例は事前に予約したまま、採用条件未達のため推論していない。
次は少数の既知形状で固定入力への学習可能性を診断し、容量不足と決めつける前に失敗の所在を切り分ける。
[DN結果](scale-style-balance-results-20261014.md) / [DN計画](scale-style-balance-plan-20261014.md)

DOは既知16形状に対するfixed/augmented各3,000更新を完了し、384出力・等倍16枚を確認。
両群とも太い入力の中央に濃い連続線を描けた。全ての失敗を容量不足とみなす根拠は弱い。
一方、近接線の灰色副線・交点・黒ベタの角は残り、部分処理の接続も不十分。
少数例への適合を汎化や製品候補と扱わず、拡張の停止だけを解決策にしない。
[DO結果](focused-fit-results-20261014.md) / [DO計画](focused-fit-plan-20261014.md)
