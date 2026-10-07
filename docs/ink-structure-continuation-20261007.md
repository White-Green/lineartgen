# Joint screen selection and 10k continuation

2026-10-07 09:45 UTC、control / neighbor_guard / ink_guard の2000→10000追加ステップ継続を開始した。
model、optimizer、EMAを2k checkpointから複製し、独立に復元して全配列の一致を確認した。
学習率scheduleは元の10k clockをそのまま使う。再帰、各段loss、sqrt(batch) scaling、入力劣化は変更しない。

親6条件と固定特徴量3条件を合算した選定である。
特徴量条件の品質結果を見る前に固定した `joint-selection-plan.json` に従い、
screen通過候補をreal / geometryのstrong strokes halo改善率の合計で順位付けした。

| 順位 | 方法 | 改善率合計 | 継続 |
| --- | --- | ---: | --- |
| 1 | neighbor_guard | 0.280646 | はい |
| 2 | ink_guard | 0.161489 | はい |
| 3 | random_features | 0.150399 | いいえ |
| 4 | learned_features | 0.145830 | いいえ |

強い特徴量lossはhaloを15.4 / 14.1%減らしたが、geometry strong blurの細線欠落が+4.02ポイント増え、screen不通過。
選定対象は数値だけでなく実際のKrita表示相当画像、raw、反復、白紙も確認した。
親2候補も、にじみと線の保持について最終採用条件にはまだ届いていない。

親2k結果: [ink-structure-results-20261007.md](ink-structure-results-20261007.md)。
特徴量2k結果: `../jax-features-20261007/docs/teacher-feature-results-20261007.md`。
親screenの `selection-review.json` にgate SHA、事前計画SHA、目視画像SHA、選定理由を保存した。

継続rootは `runs/ink-structure-20261007-continuation`。
`continuation-audit.json` は全状態一致、`source.json` はsource・設定・検証ログの固定記録。
数値sourceのcommitは `06562d5c590c34032bf9dc9827f75df451eebb3e`。
親の同一数値コードの全suiteと6段GPU preflightを引き継ぎ、全状態のforkテストを再実行して1件通過した。
`launch.json` / `analysis-launch.json` のcontrollerとwatcherが学習、10k EMA/raw、固定反復、白紙、補助評価を実行する。

10kの固定採用条件、目視確認を通過した候補だけを、別fine-tuning seedと事前固定の新しい合成条件で再検証する。
これらは未実施。現行Kritaモデルへの採用・書き出しは行っていない。
