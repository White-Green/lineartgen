# DM: 一部の灰色副線は減ったが、太い下描きを消すため不採用

16ch・3block・15,809パラメータのまま、CNN参照範囲を15pxから87pxへ広げた。
元のwide教師を保つcontextと、wideも2px清書へ揃えるthinwideを各18,000更新した。
同じ初期重み・入力・学習条件のDK normalized 18,000更新を対照として再利用した。
数値commitは`313381c`。[計画](wide-style-context-plan-20261014.md)。

thinwideではnew_probe36/38や512px scene5の一部で灰色の副線が減り、一本の清書に近づいた。
しかし、太い入力の多くが薄い残骸やほぼ白紙になった。これは線画らしい細線化とは認められない。
blackの内部は不透明だが、角の白い筋と外側の灰色の跡も増えた。両群とも製品には採用しない。

| 確認箇所 | 結果 |
|---|---|
| new_probe36/38、scene5 | thinwideで副線が減る。一部の灰色縁・端点の残りはある |
| new_probe37/39、old_probe40 | thinwideで弧の多くが薄くなり、途切れや灰色の断片が残る |
| old/new_probeのwide全12例 | thinwideは安定して濃い中心線を復元できず、線の消失・二重の灰色帯・端の跡が出る |
| 劣化wide全6形状×sharp/rough/rough部分編集 | 同じ失敗が継続。ぼかしや部分編集だけが原因ではない |
| black | contextにも細い内側の筋。thinwideは角の白い筋と外側の跡が目立つ |
| crossing30、scene7 | 交点の切れ・薄さが残る。scene1/7の意図的gapとは区別した |

標準8枚・劣化8枚・512px6枚・parallel/wide全probe6枚・wide劣化5枚の計33枚を等倍で確認した。
出力のthresholdやcontrast補正は行っていない。

- [一部改善と線消失が同居するparallel](../runs/wide-style-context-20261014/diagnostics/strokes/native-1.png)
- [wideの失敗](../runs/wide-style-context-20261014/diagnostics/strokes/native-3.png)
- [512pxの比較](../runs/wide-style-context-20261014/diagnostics/scene512/native-whole-4-5.png)
- [blackの角と外側の欠陥](../runs/wide-style-context-20261014/stress/native-review/rough-whole05-0.png)
- [全33枚の観察記録](../runs/wide-style-context-20261014/diagnostics/review-observations.json)

## 検証と範囲

CPU33 tests・Ruffを通過。初期重みと全6深さの初期関数は一致。
自身の最初6更新を再実行し、raw/EMA最大差は両群1.49e-8、既定atol1e-7/rtol1e-5内。
0/6/6000/18000の完全状態復元および16 raw/EMA exportの一致を確認した。
contextの全18,000batch入力hashはDKと一致。両新規群は入力・幾何が同じで、wideを含むbatchだけ候補が異なる。
各群72,000学習item中wideは6,758。再利用DK対照の標準/劣化3,072出力も元runと完全一致した。

標準4,608行・劣化4,608行、512px128出力を生成した。
保護領域は入力と完全一致。追加Gaussian2drawは標準/劣化各2,304 model pairs、512px64 model/encoding pairsで完全一致。
source-onlyなので、これらを独立な生成品質の標本とは扱わない。
uint8/floatの出力RMS差はcontext平均.001502/最大.004127、thinwide平均.001252/最大.003517。
画像ごとの幾何は以前の検討でも参照しており、独立した最終確認ではない。
太線を保つ旧数値参照への誤差でthinwideを不採用にしたのではなく、実際の線の消失を判断根拠とした。

dilation4/16を数学的に等価な剰余別batch畳み込みへ変更した。
旧実装6更新とのraw最大差はcontext1.05e-7、thinwide7.46e-8で、事前基準atol1e-6/rtol1e-4内。
最後600更新の更新部分は両群平均約.105秒。データ準備等を含む全step時間ではない。
初回340更新、第2試行473更新は実装速度のため停止し、完了した画質実験には数えない。
[速度と保存記録](wide-style-performance-20261014.md)。

## 次に切り分けること

元の太線ink量でlossを正規化しているため、細線教師の消失を小さなコストとして扱う可能性がある。
許容清書のink量を全候補で共通に平均した正規化との比較が必要。候補ごとに分母を変えて細線を有利にする方式ではない。

縮小段の意味も未解決。2pxの入力をそのまま扱う場合と、4pxの入力を半分へ縮小した場合には、
正規化guideの局所領域が一致しても、2px清書教師の濃度は異なる。
zero insertを持つ最小段の構成例で、RF15学習済みモデルと非zero headのRF87モデルとも中心32px領域の出力が完全一致した一方、
必要な教師にはsigned画像値で最大1.0の差があった。
[構成例と数値](../runs/wide-style-context-20261014/diagnostics/scale-ambiguity/results.json)。
これは起こり得る入力の曖昧さの確認であり、今回の全欠陥の原因や改善効果の証明ではない。
DEのwarm start・追加scalar FiLM 6,000更新では改善しなかった。次の比較では既存9次元の条件特徴を使い、
容量を増やさず新しい初期値から段情報を学習する候補を検討する。

[主実験完了記録](../runs/wide-style-context-20261014/completion.json)。
合成のみ。Krita・配布は変更していない。
