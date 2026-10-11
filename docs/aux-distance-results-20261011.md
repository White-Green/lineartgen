# BK: 学習時だけ距離を予測する補助head

**画像出力から距離予測を分離できたが、品質の採用条件は未達。**
全方式が固定学習例の22条件を通過。補助lossありは保持32条件中2失敗で、
どちらも未学習直線807のF1（同時学習control・凍結AOの各比較）だった。
補助lossなしcontrolは保持16条件を全通過したが、既存のグレー問題は残る。

AO7200のcore重みから両方式を開始し、同じ新規Adam/EMAで追加1200更新、
一定LR1e-5（base5e-6×sqrt(batch4)）。4固定学習対・4診断対、512px、再帰1–6。
条件差は補助loss係数0/.1のみ。9→16→16→1の補助headを学習時だけ追加し、
総134382パラメータ。推論時は130605パラメータのcoreだけを取り出す。
数値コード`f724661`、テスト29件、合計57.83秒。

| 未学習例 | 同時学習control F1 / 欠損 | 補助loss F1 / 欠損 | control halo → 補助halo |
|---|---:|---:|---:|
| 直線807 | .99779 / .14876 | .98960 / .11145 | .000514 → .000749 |
| 細線903 | .99483 / .00124 | .99409 / .00141 | .000192 → .000181 |
| 分岐1191 | .71363 / .22831 | .72940 / .21566 | .020851 → .019131 |
| 太線1479 | .94590 / .24284 | .94351 / .24932 | .009160 → .008365 |

5枚すべてを目視。細線の大きな消失はない。直線は濃くなる一方で輪郭F1が低下。
分岐の切断率.2667は同時学習controlと同じ。太線は内部がグレーで幅不足のまま。
補助field誤差は約.936から.00508へ低下したが、画像品質の解決を意味しない。
BKは初期値・optimizer条件がBJと異なるため、数値差を出力分離だけの効果とは
解釈しない。同時に学習したBK controlとの比較が対応する対照実験である。

- [学習例・原寸](../runs/aux-distance-20261011/evaluation/train-native.png)
- [未学習例・原寸](../runs/aux-distance-20261011/evaluation/probe-native.png)
- [部分適用・原寸](../runs/aux-distance-20261011/evaluation/partial-native.png)
- [学習例・全体](../runs/aux-distance-20261011/evaluation/train-overview.png)
- [未学習例・全体](../runs/aux-distance-20261011/evaluation/probe-overview.png)
- [全数値](../runs/aux-distance-20261011/evaluation/metrics.json)
- [最終監査](../runs/aux-distance-20261011/final-audit.json)

補助headを変更しても画像の全再帰段が不変、係数0なら元のloss・全core勾配に一致、
距離lossから共有encoderへ勾配が到達することを検証。既存ONNX回帰テストも通過。
AO8出力の再現、最初の6更新の完全再生、4時点のwrapper・Adam・EMA復元、
16 core-only raw/EMA export、72評価行・21予測block・保護画素の完全一致を確認。
実モデルのWindows/Krita GUI品質は未検証。本番・プラグイン・既定設定は不変。

補助距離学習の着想は[Audebertら](https://arxiv.org/abs/1909.01671)、
学習後に外す補助branchは[Dangiら](https://arxiv.org/abs/1901.01238)を参照。
これらの領域分割結果は、この線画モデルの品質を保証するものではない。
