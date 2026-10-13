# 512px画像の意図的なgapを区別する補足

DK学習中にscene1/7とscene2/5の幾何情報を再描画し、2枚を等倍で確認した。
入力と元のclean、同じ幾何の2px peak版を並べている。
scene7の構成はgap・straight・gap・bezierで、縦線の中央と左上の斜線に意図的なgapがある。
scene1にも斜めの長い線の意図的なgapがある。
以前の512pxレビューにある「縦線の途切れ」という表現は広すぎた。これらの意図的なgap自体を
モデル欠陥と扱わない。正解との完全一致も要件ではない。

DJのscene7部分適用では、意図的な中央gapとは別に、上の斜線との接点および下の曲線との交点で
小さな切れが見える。今後は意図的gapと追加の切れを分けて記述する。
scene2/5の幾何は近接した別々の曲線を含むが、曖昧な下描きから一本へまとめること自体は
失敗としない。比較する欠陥は、不規則な灰色側線や線周辺の面状の灰色である。

この補足で学習・数値評価・checkpoint選択は変更していない。
旧レビューJSONを上書きせず、解釈上の訂正をここに記録する。
既存の標準256px crossingの画像30番は別の入力で、この意図的gapの補足をそのまま適用しない。

- [scene1/7の幾何](../runs/full-source-normalization-20261013/diagnostics/scene512/oracle-1-7.png)
- [scene2/5の幾何](../runs/full-source-normalization-20261013/diagnostics/scene512/oracle-2-5.png)
