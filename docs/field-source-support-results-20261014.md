# DU: 入力の広がりに合わせた外側loss — 明確な改善なし

DT balanced raw6000から同じ16,064パラメータの2条件を各6,000更新した。対照はDTのloss、変更版は入力の線が広がる範囲も外側符号lossで重く扱う。数値実装`bf65b80`、11テストとRuff通過。再帰段数・各段loss・sqrt(batch)学習率を維持した。

事前の勾配診断では、太線失敗例59/62の誤った黒領域のうち、許容線から離れた部分への修正勾配が約29倍になった。実装が信号を変えることは確認できた。しかし学習後の出力は対照とよく似ており、太い黒帯、端の断片、ぼかした近接線の消失・途切れが残った。外側の重み不足だけでは今回の失敗を説明できない。

両条件ともDTからの継続学習で一部の枝・交点・近接曲線が改善した。暗く滑らかな一本にまとまった例は有効な清書として評価する。この改善を入力支持領域の追加による効果とは扱わない。

- [改善した枝・交点と滑らかな統合線](../runs/field-source-support-20261014/evaluation/native-review/new_probe-whole05-1.png)
- [残る太線の黒帯と断片](../runs/field-source-support-20261014/evaluation/native-review/new_probe-whole05-2.png)
- [強いぼかしでの途切れ](../runs/field-source-support-20261014/stress/native-review/soft-whole05-0.png)
- [全28ページの等倍観察](../runs/field-source-support-20261014/diagnostics/review-observations.json)

標準4,608＋劣化4,608＋512px192＝9,408出力を保存し、事前指定28ページを等倍確認した。初期関数6段数一致、6,000回の入力一致、先頭6更新の再現、状態復元と16 raw/EMA exportは通過。先頭再現は完全一致だった。DN参照3,072出力は以前と同一、Gaussian draw間の同一性も確認。保護領域保持とfloat/uint8差を検査した。量子化差RMSはsupported平均0.001986、最大0.005647。画質順位には使用しない。

学習1,809.04秒、評価241.97秒。最後600更新の平均はcontrol0.11809秒、supported0.11854秒。最終の条件間パラメータ差はRMS0.000576、最大0.006227で、別のモデルになっている。出力も同一ではないが、目標とする線画品質の明確な差にはならなかった。

次は現在の幾何出力モデルを難しい既知の16学習例に集中して当てる。入力固定とデータ拡張ありを比較し、同じ形状さえ表現・最適化できないのか、入力変動への対応が主な問題なのかを切り分ける。以前のDOは画像濃度を出すモデルなので、今回の幾何場モデルの学習可能性は未検証である。一般化や製品採用の試験とは区別する。

両条件とも製品採用なし。Kritaと予約済み新規確認データは変更していない。

[計画](field-source-support-plan-20261014.md) / [完了記録](../runs/field-source-support-20261014/completion.json) / [整合性](../runs/field-source-support-20261014/diagnostics/integrity.json) / [固定出力での勾配診断](../runs/field-source-support-20261014/diagnostics/support-effect/results.json)
