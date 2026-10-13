# DE: 縮小段の条件付け

比較完了。今回の段番号の条件付けには明確な画質上の利点がなく、採用しない。
灰色の側線、端点、交点の不自然さ、薄い黒対照の輪郭リングが残った。
全ての段条件付けが無効という結論ではなく、この初期値・方式・6,000更新での結果である。

DD canonical raw6000から、16ch共有CNNのstemと3残差blockへ各32個、計128個の
ゼロ初期化係数を追加した。両条件15,937パラメータ。
constantは常に1、levelは最細段からの縮小回数kに応じて1+k/5をFiLMへ与える。
残りdenoise progressとは別の値。全6段で親モデルと同じ出力から開始する。
同じ512合成下描きと2px許容清書、同じloss・全段重み・sqrt(batch)LRで各6,000更新した。

9,216モデル出力の数値評価と16枚の等倍確認を完了。
追加512pxの複合図形・全体/部分mask・float/uint8入力128出力と6枚も確認した。
二つの条件は目視でもほぼ同様。元教師との幅の違い、揺れた入力への追従は許容して判定した。

- old_train24の短い枝、old_probe30/new_probe30の交点、端点の毛羽は解消しない。
- faintの細線は薄く、soft/roughの平行線には灰色の側線が残る。
- faint/roughの黒対照は灰色の内側と輪郭リングが残り、partialの編集側にも現れる。
- 512px scene1/5の長い曲線の側線、scene2の近接曲線の乱れ、scene7の切れ目も残る。
  その他の多くの単純な線は明瞭だが、これらを除外して成功とはしていない。

共通54候補の断面距離と有効接続問い合わせの切断率、whole .5。
MSEや元教師への厳密な幅一致を成功基準としていない。

| 条件 | constant 断面 | level 断面 | constant 切断率 | level 切断率 |
|---|---:|---:|---:|---:|
| new_probe | .162712 | .164174 | .020840 | .021330 |
| old_probe | .159997 | .160409 | .006384 | .006501 |
| faint | .233096 | .232391 | .026661 | .024960 |
| soft | .199845 | .200628 | .003411 | .003411 |
| rough | .229573 | .231969 | .030621 | .030866 |

数値commit `a233dbc`。CPU24テストとRuff、GPUで親との全6段初期一致、
全6,000更新の両armの入力・教師・乱数hash一致、6/6000の全状態復元、12raw/EMA exportを確認。
最初6更新のGPU再実行はconstant最大4.47e-8、level最大2.98e-8で、
事前固定atol1e-7/rtol1e-5を通過。ビット一致ではない。
追加Gaussianの768対×標準/劣化は各モデルで同一、512pxも64 model/encoding対で同一。
保護画素は全て完全一致。float/uint8出力RMS差の平均は.001021/.001010で、画質の指標ではない。

次は、DDの固定スタイルを基礎に、出力そのものを清書の分布へ近づける学習されたpriorを
改めて比較する余地がある。CTの弱いpriorは異なる入力・目的での比較であり、
現在のsource-only・許容清書選択との組合せはまだ試していない。

合成データのみ。モデル保存はJAX形式。段入力がない既存ONNX exportは明示的に拒否する。
Krita・配布・封印データは変更していない。

- [計画](pyramid-family-plan-20261013.md)
- [完了記録](../runs/pyramid-family-20261013/completion.json)
- [薄い入力の比較](../runs/pyramid-family-20261013/stress/native-review/faint-whole05-0.png)
- [512pxの比較](../runs/pyramid-family-20261013/diagnostics/scene512/native-whole-4-5.png)
- [512px完了記録](../runs/pyramid-family-20261013/diagnostics/scene512/completion.json)
