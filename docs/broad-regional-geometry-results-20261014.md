# EA: 全形状で線の保持は回復したが、太線の清書は未解決

少数例への集中学習前のDU control raw6000を共通初期値として、512形状すべてで各18,000更新した。対照は従来balanced loss、変更版は境界付近のみの距離一致・RMS符号違反・近傍/遠方の独立正規化を組み合わせたregional loss。同じ16,064パラメータ、再帰1–6段、各段loss、sqrt(batch)学習率を維持。数値実装`6bfbe83`、17テストとRuff通過。

全形状で学ぶと、DZの少数例学習で消えた長い曲線や交点は両条件でかなり保持された。Regional版は512px場面1の近接曲線と枝の途切れ、場面2の曲線の途切れ、場面7の交点を対照より改善した。多くの通常線は暗く連続し、広い灰色のにじみは目立たない。黒ベタの角の穴も小さくなった。これは少数例の穴改善が一部の広い入力でも有効な結果だが、矩形形状自体は学習時と共通である。

一方、太線の59/61ではregional版が対照より太い黒帯や枠状になった。58/60/62にも端の棒状断片や途切れが残る。強いぼかしをかけた短い近接線36は両条件で消失・分断する。Rough63の細線はregional版のほうが残るが、端の余分な線も出る。したがって全体の品質基準は未達、製品採用は行わない。画素MSEの順位を採用根拠にしていない。

- [改善した512pxの線の連続性](../runs/broad-regional-geometry-20261014/scene512/native-whole-0-1.png)
- [通常線の保持](../runs/broad-regional-geometry-20261014/evaluation/native-review/new_probe-whole05-5.png)
- [穴の改善と太線の失敗](../runs/broad-regional-geometry-20261014/evaluation/native-review/new_probe-whole05-2.png)
- [強いぼかしでの近接線の欠落](../runs/broad-regional-geometry-20261014/stress/native-review/soft-whole05-0.png)
- [全28ページの観察](../runs/broad-regional-geometry-20261014/diagnostics/review-observations.json)

9,408出力を保存、指定28ページを確認した。両条件18,000回すべてで入力一致、初期6段数の関数一致、先頭6更新の状態/指標は完全一致で再現。全状態復元と16 raw/EMA exportも完全一致。全512形状に全6段数で5–40回の提示があり、3マスクの提示数は23,934/23,925/24,141。Gaussian drawの同一性、保護領域保持、DN参照3,072出力の再利用一致も確認した。

学習5,574.25秒、標準評価145.12秒、劣化90.89秒、512px8.72秒。最後600更新の平均はcontrol0.11732秒、regions0.13089秒。初期重みからの変化RMSは0.02406/0.02661。512px float/uint8出力差RMS平均/最大はcontrol0.002024/0.009028、regions0.001503/0.004297。これらは動作・整合性の記録であり、線画品質の順位ではない。

次は固定重みに同じ画像の90度回転・反転を入力する診断を行う。元の向きへ戻した出力を比較し、太線の失敗が向きに強く依存するかを確認する。新たな大規模学習や容量変更の前に、形状の多様性・向きへの対応を切り分ける。予約済み新規確認、Krita、製品重みは変更しない。

[計画](broad-regional-geometry-plan-20261014.md) / [完了記録](../runs/broad-regional-geometry-20261014/completion.json) / [整合性](../runs/broad-regional-geometry-20261014/diagnostics/integrity.json)
