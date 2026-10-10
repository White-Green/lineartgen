# 線画のグレー化を減らす実験 — 継続記録

採用候補はまだない。13万パラメータで固定例を十分に復元できることは確認できたが、
未学習の線画にも通用するかは別に検証している。合成下描きだけを使用し、配布モデルは変更していない。

| 実験 | 結果と意味 |
| --- | --- |
| AL: インク量でlossを正規化 | 固定曲線のhaloは約49%減ったが、濃度不足が増えた。22基準中1つに未達。 |
| AM: 同じlossで6,000更新 | 曲線F1は.95888→.97529。学習を続ける効果はあるが、基準.98には未達。 |
| AO: 6,000更新後に学習率を比較 | 各1,200更新。低学習率側が固定4例の22基準と8保全条件をすべて通過。曲線F1 .98276。未学習の太線は依然グレー。 |
| AP: 通常データの学習を延長 | 16,000更新まで完了。コーパスblurのhaloは約22%減ったが、濃度不足が増え、229基準中56件が未達。未採用。 |
| AQ: 同じ下描きに複数の正解がある解析例 | 既知の2正解を使う参照計算では、中間のグレーを出す1回の推定と、どちらかの形を選ぶ逐次輸送を区別できた。ニューラルモデルの性能改善ではない。 |
| AR: 全体のチャンネル幅を拡大 | 同一loss・データで13万→52万パラメータ、4,000更新。図形blurの融合率は.2708→.1458に改善したが、グレーの細線と重ね描きの太さの誤りは残った。229基準中41件が未達で未採用。 |
| AS: 輪郭の濃度変化を正規化したloss | 固定曲線のhaloは約32%減少したが、F1は.95909で基準.98に未達。未学習の線の濃度不足も残り、通常データへの展開は行わない。 |
| AT: 複数の正解がある例で逐次復元を学習 | 同じ下描きから1本の太線か2本の細線を出すCNNを各6,000更新で比較。32回の復元でグレーの平均から離れる場合はあるが、一方への偏りやノイズが残り、36条件中30件に未達。通常の線画での改善ではない。 |
| AU: 学習済みの逐次復元を256回まで反復 | 同じ重み・乱数で比較。回数を増やしても線画への偏りや背景のノイズは解消せず、256回でも36条件中28件に未達。反復回数だけでは改善できない。 |

ALは46テスト・2枚、AMは24テスト・2枚、AOは25テスト・2枚、AQは7テスト・1枚を確認。
APは28テストと最終14枚を確認した。これらには重複した共通テストがあり、合計を独立した検証数とは扱わない。
ARは39テストと最終14枚を確認した。4,000更新すべてのデータ割り当ては小さいモデルの対照と一致している。
ASは37テストと最終2枚を確認した。lossの正常動作と実際の画質改善は区別して記録している。
ATは16テストと最終3枚を確認した。64pxの2種類の線画だけを使った機構の検証である。
APは初期の評価集計で比較名の不具合があり、別worktreeで修正した。最初の6更新について
重み・optimizer・EMA・データ・初期出力の完全一致を確認してから続行している。

## 詳細と実際の出力

- [ALの結果](../../jax-ink-mass-fit-20261010/docs/ink-mass-fit-results-20261010.md)
- [AMの結果](../../jax-fit-duration-20261010/docs/fit-duration-results-20261010.md)
- [AOの結果](../../jax-late-fit-settling-20261010/docs/late-fit-settling-results-20261010.md)
- [AO: 学習に使った4例](../../jax-late-fit-settling-20261010/runs/late-fit-settling-20261010/evaluation/step-7200/train-native.png)
- [AO: 学習に使っていない4例](../../jax-late-fit-settling-20261010/runs/late-fit-settling-20261010/evaluation/step-7200/probe-native.png)
- [APの最終結果](../../jax-paper-band-long-run-eval-fix-20261010/docs/paper-band-long-run-results-20261010.md)
- [APの事前計画](../../jax-paper-band-long-run-eval-fix-20261010/docs/paper-band-long-run-plan-20261010.md) / [集計修正の記録](../../jax-paper-band-long-run-eval-fix-20261010/docs/paper-band-long-run-evaluation-repair-20261010.md)
- [AQの解析結果と限界](../../jax-conditional-bridge-oracle-20261010/docs/conditional-bridge-oracle-results-20261010.md)
- [AQの比較画像](../../jax-conditional-bridge-oracle-20261010/runs/conditional-bridge-oracle-20261010/oracle-native.png)
- [ARの結果と出力画像](../../jax-full-width-paper-20261010/docs/full-width-paper-results-20261010.md)
- [ASの結果と出力画像](../../jax-edge-energy-fit-20261010/docs/edge-energy-fit-results-20261010.md)
- [ATの結果と出力画像](../../jax-learned-bridge-toy-20261010/docs/learned-bridge-toy-results-20261010.md)
- [AUの結果と出力画像](learned-bridge-solver-results-20261010.md)

独立した学習seed・封印した合成条件での確認・実際のBGRA入出力での品質確認は、
採用候補が成立した後の工程として残る。Windows/Krita GUIでの確認は行っていない。
ANという先行の条件付き計画は開始条件を満たさず未実行のまま保存した。APはAOの途中・最終結果を見る前に
別の診断として固定したもので、ANの条件を書き換えて動かしたものではない。
