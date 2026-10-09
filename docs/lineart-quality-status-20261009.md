# 線画復元実験の現状 — 2026-10-09

まだ製品モデルを置き換えていない。直近で最も改善したのは、追加Gaussianを
CNNの特徴入力から外し、元の下描きから補正を求める構成。線の周囲のノイズや
幅の不安定さは大きく減ったが、偽接続と一部の濃度不足が残る。

実行中: 同じsource特徴で、既存pixel loss、region.1、region.1+connection.1
を比較している。各4,000更新の固定条件で、pixel対照は完了済みモデルを厳密に
再利用する。[計画](../../jax-signal-structure-20261009/docs/signal-structure-plan-20261009.md)。

| 完了した比較 | 主な結果 | 判断 |
| --- | --- | --- |
| 4ケース共有・接続query /2k | gap中央濃度75.30%/58.76%減、PROBE保持条件4項目失敗 | 不採用 |
| 毎更新新規形状・region/structure /4k | halo約23–25%減、細線保持など15/17項目失敗 | 不採用 |
| 局所opacity輸送 /4k | halo64.85%/62.37%減、薄線の切断・濃度が悪化、49項目失敗 | 不採用 |
| 下描きを出力基準に変更 /4k | 滑らかだがhalo約30–34%増、14項目失敗 | 不採用 |
| 下描きを特徴入力にも使用 /4k | halo61.80%/57.88%減、PROBE細線幅誤差2.60→.56px、12項目失敗 | 有望な仕組み、品質は未完 |

最新候補は非resetの従来pixelモデルに対してもhalo56.48%/52.46%減で、
補助比較の失敗はPROBE gapの偽接続1項目。ただし本来の対照とは一部の濃度が
悪化し、全判定を通過していない。失敗数は重複する比較項目である。
[最新結果](source-signal-results-20261009.md)、
[等倍出力](../../jax-source-signal-20261009/runs/source-signal-20261009/broad/step-4000/probe-blur-0.5-native.png)。

直近の各比較で最終8枚を実際に目視した。最新は補助3枚も確認した。
白紙はほぼ白だが微小な端の誤差がある。保護領域は厳密一致。ONNX/BGRAの
機能テストは通るが、実学習モデルのKrita GUI品質を確認した意味ではない。
元画像特徴の候補はGaussian seedで変化しなくなる。この挙動変更は意図して
記録し、実学習出力でも検証した。入力画像自体のノイズは残る。

モデルは130,605パラメータ。再帰1–6、段loss0、sqrt(batch)学習率、縮小後の
ノイズ生成を維持する。合成下描きのみ使用。新規形状の開発実験は幅・方向・
形状を毎回変えるが、より広い実線画由来の合成下描き、重ね描き、反復推論、
独立学習seed20261005、封印した新規合成drawによる確認はまだ未実施。
十分な最小モデルサイズは未確定。封印評価は未開封、配布中100kモデルは据え置き。

過去の詳細: [connection](connection-query-results-20261009.md)、
[procedural](procedural-structure-results-20261009.md)、
[transport](local-transport-training-results-20261009.md)、
[anchor](source-anchor-results-20261009.md)。それ以前のloss/guide/反復経路の比較は
各日付の結果文書に記録している。輸送lossの起動再実行にだけ非bitwise差があり、
[例外記録](local-transport-startup-amendment-20261009.md)に明示した。
