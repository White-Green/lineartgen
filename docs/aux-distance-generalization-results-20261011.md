# BL: 多様な完全合成教師での補助距離学習

**4,000更新・6,264評価行を完了。補助lossの十分な品質改善は確認できず、不採用。**
補助なしcontrolは229条件中44失敗、補助ありは47失敗。失敗数は品質スコアではなく、
既定の保持条件を外れた数として記録する。

同一E4000 coreから新規Adam/EMA、学習時134382・推論時130605パラメータ。
両方式の差は距離補助loss係数0/.1のみ。seed2026101111、batch4、LRはsqrt4後
peak1e-4、warm100/cosine4000→1e-5。再帰1–6、最高段lossと小段への勾配を維持。
数値コード`da7f678`、テスト24件（35.01秒）。途中の出力は選択していない。

教師16,000枚は全て手続き生成。geometry14,428枚、太線321、黒312、楕円299、
多角形329、白311。各再帰深さ667/667/667/667/666/666更新。
合成の濃度・ぼかし・重ね描きで学習。評価のcorpusはclean画像への合成劣化であり、
人が描いた実下描きとの対応対ではない。既知の開発条件であり、独立確認でもない。

全体適用・強度.5の平均値:

| 対象 | control F1 / 欠損 / halo | 補助あり F1 / 欠損 / halo |
|---|---:|---:|
| 幾何 blur | .88214 / .04987 / .027061 | .88675 / .05281 / .025763 |
| 幾何 strokes | .70501 / .02966 / .131960 | .70579 / .03107 / .130919 |
| corpus blur | .93448 / .08123 / .005008 | .93544 / .08444 / .004890 |
| corpus strokes | .85331 / .04733 / .077356 | .85390 / .04885 / .074219 |

補助ありはhaloが少し下がる一方、4群とも濃度欠損が増えた。幾何blurの切断率は
両方.00625、誤接続.29167。strokesの幅誤差は2.0197→2.0058px、誤接続は
両方.39583。白紙の最大濃度はcontrol.01728に対し補助あり.15806で基準.05を超えた。
黒領域2例717/727の編集内部は全方式mean/min alpha1.0だった。

6枚の原寸シートを目視。ぼかし曲線・分岐の灰色、平行線の融合、隙間の閉鎖、
重ね描きからの太すぎる復元が残る。corpusの細かい線も欠損・濃度不足がある。
学習専用の距離headを分けるだけでは、今回の問題を解決できなかった。

- [ぼかし・原寸](../runs/aux-distance-generalization-20261011/development/native-review/blur-05.png)
- [重ね描き・強度.1](../runs/aux-distance-generalization-20261011/development/native-review/strokes-01.png)
- [重ね描き・強度.5](../runs/aux-distance-generalization-20261011/development/native-review/strokes-05.png)
- [corpus・原寸](../runs/aux-distance-generalization-20261011/development/native-review/real.png)
- [部分適用](../runs/aux-distance-generalization-20261011/development/native-review/partial.png)
- [白紙の最大値付近](../runs/aux-distance-generalization-20261011/development/native-review/blank-worst.png)
- [集計・全基準](../runs/aux-distance-generalization-20261011/development/summary.json)
- [最終監査](../runs/aux-distance-generalization-20261011/final-audit.json)

全4,000更新の教師hash・種類・深さ、171点の独立再生成、先頭6更新の完全再生、
4時点の全状態復元、16 core-only raw/EMA exportを確認。凍結E/Rの3,132評価行と
保存panelの予測も完全一致。初期Eの古いsampling metadataだけをone_passに変え、
重みは不変で16出力を事前再現した。初回prepareのmetadata不一致とテストfixtureの
ノイズ方式不一致は学習開始前に修正し、失敗ログも保持した。

シートはKrita相当の表示変換を使うが、生BGRA API・Windows GUIの品質確認ではない。
独立seed・封印条件は未実施。本番・Krita・既定設定は変更していない。
