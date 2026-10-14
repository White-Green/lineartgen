# DZ: 少数例での改善は広い入力には転移しなかった

DU control raw6000、DY control raw3000、DY regions raw3000の固定重みを、既存の16開発プローブと8枚の512px場面で比較した。再帰3/6段、全面/部分マスク、float/uint8入力の576出力を保存し、事前指定32ページを確認した。数値コード `f6863d1`。学習・チェックポイント選択は行っていない。

DY regionsの黒ベタの穴を減らす効果は残る。しかし両DYモデルとも、DUが比較的きれいに出していた近接曲線・交点を途切れさせ、512px場面の長い線を消す。未知の太線58/59/62では断片や非常に薄い中心線が残り、清書として使える細線にならない。3段から6段に増やしても解消しない。滑らかな一本への統合は許容し、その有無を失敗とは数えていない。

- [近接線・交点の退行](../runs/geometry-transfer-20261014/evaluation/native-probe-d6-whole-4.png)
- [太線での消失・断片](../runs/geometry-transfer-20261014/evaluation/native-probe-d6-whole-12.png)
- [512px場面の線の欠落](../runs/geometry-transfer-20261014/evaluation/native-scene-d6-whole-4.png)
- [全32ページの観察記録](../runs/geometry-transfer-20261014/evaluation/visual-review.json)

この比較は少数16形状への集中学習の転移を調べたもので、新lossが広い学習でも劣ることや、モデル容量不足を証明するものではない。黒ベタ56も学習時と同じ矩形形状なので、穴の改善を新形状への一般化とは呼ばない。

重みは評価前後で不変、全出力は有限、保護領域は入力と完全一致。評価8.70秒。未量子化入力60条件に限ったfloat/uint8出力差RMSの平均/最大はDU 0.001210/0.002801、DY control 0.001434/0.003252、regions 0.001425/0.003477。画質の選択指標には使わない。もともと量子化済みの入力で差がゼロになることを、量子化耐性の証拠として扱わない。

次は少数例学習前のDU controlを共通初期値として、512形状すべてで従来balanced lossとDW/DX/DYを組み合わせたregional lossを比較する。モデルサイズは据え置き、各18,000更新で固定する。製品採用なし。Kritaと予約済み新規確認データは変更しない。

[計画](geometry-transfer-plan-20261014.md) / [完了記録](../runs/geometry-transfer-20261014/completion.json)
