# DT: 線内部・外部を分けた幾何学習 — 消失は軽減、太線と接続は未解決

同じDR warm raw6000から、従来lossと符号マージンを加えたlossを各6,000更新した。16,064パラメータ、入力、再帰段数、各段loss、sqrt(batch)学習率は同条件。数値実装は`f60282e`。符号マージン以外の継続学習条件を揃えた比較である。

符号マージンによって、消えていた細線・枝・近接線が明瞭に戻った。近接線が暗く滑らかな一本へ整理される例もあり、これは許容する。黒ベタ内部の白い穴も減った。一方、交点・枝の付け根での切れ、近接線の断片化、太い下描きが太い黒い帯になる問題は残る。線の存在を確保する効果は認められるが、製品の品質条件は満たさない。

- [細線・枝・交点・近接線](../runs/field-sign-balance-20261014/evaluation/native-review/new_probe-whole05-1.png)
- [太い下描きと黒ベタ](../runs/field-sign-balance-20261014/evaluation/native-review/new_probe-whole05-2.png)
- [512pxでの回復と残る切れ](../runs/field-sign-balance-20261014/scene512/native-whole-4-5.png)
- [全28ページの観察記録](../runs/field-sign-balance-20261014/diagnostics/review-observations.json)

標準4,608、劣化4,608、512px192の計9,408出力。事前に指定した28ページを等倍で確認した。512pxのscene1/7の大きな隙間は意図したものとして区別した。画像の二値化や選別したcheckpointへの差し替えは行っていない。全体/部分の処理とt=0保持、float/uint8入力を比較した。512px出力の量子化差RMSはbalanced平均0.00212、最大0.00585。この値は画質の順位付けには使わない。

7テストとRuff通過。両条件の初期関数を全6段数で照合し、6,000回の学習入力一致、先頭6更新の再現、完全状態の復元、16個のraw/EMA exportを検証した。DN参照出力3,072個はDRと同一。source-onlyの異なるGaussian drawで標準2,304組・劣化2,304組・512px96組が同一。学習は1,795.14秒、評価は合計245.21秒。最後600更新の平均はcontrol0.1146秒、balanced0.1186秒。異なるloss同士の値の大小を画質改善の根拠にはしない。

次は残る太線出力の幾何場とloss勾配を調べる。現状では正解輪郭から離れた領域の重みが0.01なので、太い入力の外側を消す信号が弱い可能性がある。ただし原因と断定せず、予測された黒領域がどこにあり、どの項が修正を促しているかを計測してから次の比較を決める。

製品採用・Krita変更なし。予約済み新規確認データは未使用。研究用の幾何場format2は現行Krita runtimeに直接読み込めない。

[計画](field-sign-balance-plan-20261014.md) / [完了記録](../runs/field-sign-balance-20261014/completion.json) / [整合性](../runs/field-sign-balance-20261014/diagnostics/integrity.json)
