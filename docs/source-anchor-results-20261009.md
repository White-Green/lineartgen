# 下描きを出力の基準にした比較の結果

4,000更新の比較を完了した。**出力の基準を変えるだけでは不十分で、不採用**。
「ノイズ画像＋補正」を「元の下描き＋補正」にすると粒状のノイズは減るが、
細線は太い灰色の帯になり、角・線端・白紙に架空の枝や輪郭が残る。
最終8枚と旧モデルとの補助比較2枚を等倍で確認した。

| 最終指標 | noisy基準（出力層reset対照） | source基準 |
| --- | ---: | ---: |
| TRAIN blur 平均halo | .062157 | .083582 |
| PROBE blur 平均halo | .051420 | .067068 |
| PROBE blur / 全体 / 強度.5 F1 | .708576 | .734710 |
| 同条件 ink deficit | .134037 | .170109 |
| PROBE細線 width error px | 1.848412 | 2.600667 |
| PROBE細線 cut rate (8 draws) | .000000 | .187500 |
| 固定比較基準の失敗数 | 対照 | 14 |

source基準のhaloは対照よりTRAIN34.47%、PROBE30.43%多い。F1の一部改善を
採用根拠にしない。4ケース×8ノイズのfitting基準は両モデルともTRAIN0/32、
PROBE0/32。保護領域は全条件で厳密に維持した。

両モデルの最終出力projectionのみ同じくzero resetし、その他のpreclip-opacity
親モデルの重みは同一。130,605パラメータ、再帰1–6、段loss0、sqrt(batch)LR、
縮小後Gaussian、16,000例の新規geometryとaugmentation、lossは共通である。
異なる初期関数（対照はnoisy、候補はrough）を意図した出力基準比較。

実用的な比較として、出力層をresetしていない従来のprocedural pixelとも、
同じ64/456条件で照合した。こちらに対しても9基準失敗、haloは悪化した。
初回1,000評価前に固定した補助比較であり、初期化が異なるため純粋な基準変更
の因果比較とは区別する。補助8枚を生成したが目視したのは最終blur.5の
TRAIN/PROBE2枚のみ。主比較は最終8枚全て、中間はPROBE blur.5の1枚のみ確認。

6更新の全状態再実行、1,000/4,000の全状態復元、両段階raw/EMA計8exportの
checkpoint一致と固定ソースを確認。機能テスト36件にはONNX経由のBGRAと
1/2/5反復を含むが、実学習モデルのKrita GUI品質確認ではない。
結果: runs/source-anchor-20261009/{final-audit,snapshot-audit}.json、
broad/step-4000/{summary,metrics,visual-review}.json、warm-reference/以下。

次は出力だけでなくCNNの特徴入力も元の下描きにして、追加Gaussianを
特徴抽出から外す。この比較は本試験のsourceモデルをそのまま対照に再利用し、
パラメータ・loss・入力生成・再帰・学習率を保つ。乱数seedが出力に影響しなく
なる実験であり、確率的な挙動まで維持すると主張しない。実ノイズを含む
ユーザーの下描きはそのまま入力される。製品モデルはまだ置き換えない。
