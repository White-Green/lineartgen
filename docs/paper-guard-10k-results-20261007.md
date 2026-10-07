# STE neighborの10k結果

採用しない。2kの全状態を保存して10kへ継続したが、EMA・rawとも最終条件で96判定に失敗した。
条件やstratumが重複するので、独立した欠陥が96個という意味ではない。

| 対control・EMA | 結果 |
| --- | ---: |
| real strong strokes halo改善 | 10.53% |
| geometry strong strokes halo改善 | 12.03% |
| geometry strong blurのprofile欠落差 | −3.44ポイント |
| geometry strong toneのink不足差 | +1.32ポイント |
| geometry strong toneの切断率差 | +3.70ポイント |
| geometry strong strokesのbridge率差 | +2.08ポイント |
| geometry strong blurの遠方背景opacity excess差 | +0.000514 |

主halo条件は両方通ったが、ink不足・背景・切断等の安全条件を満たさなかった。
失敗内訳は遠方背景52、ink不足26、F1 5、bridge 5、位置誤差5、切断3。
rawも主haloは10.73% / 12.21%改善し、最終判定は不合格だった。
背景の汚れは白紙でも残り、40条件×1/2/5回における最大平均opacityはEMA 0.03313、raw 0.02305。
これらは異なる条件の最大値同士で、同じ入力のEMA/raw差ではない。
real cleanのF1低下が大きいcase20/37/46は編集領域の教師inkが0で、F1が1から0へ変化した。
このF1差を線の消失率と解釈せず、空白の誤inkに敏感な指標として扱う。固定判定からは除外していない。

主halo絶対差のcluster 95%区間はreal −0.01954〜−0.01228、geometry −0.02993〜−0.02029。
realは46ケース・12source、geometryは128ケース・32群。学習seed間の不確実性を含まない。

同じneighbor lossを使うhard clipの直接対照は、halo改善16.68% / 17.57%で背景もより清潔だった。
一方、STEはhard比でgeometry strong blurのprofile欠落を約4.68ポイント減らした。
線を拾う性質と紙を白くする性質の両立には至らなかった。

EMA geometry24例、raw strong blur8例、real合成下描き3詳細とraw1詳細、
4例の1回/5回反復、部分編集白紙1例を実際に確認した。
細線消失、線の融合、閉じるべきでない隙間、灰色の縁や背景の点状汚れが残る。
モデルを更新する根拠として十分な改善とは判断しない。

root: `runs/paper-guard-20261007-continuation`、数値source: 130360c。
`analysis-with-baselines-10000.json`、`quality-gate-with-baselines-10000.json`、
`review.json`、`completion-audit.json` に保存。定義済みmetricは全て有限、保護領域は完全一致。
モデル・optimizer・EMAの2k転送と10k checkpoint／推論書き出しの一致を確認した。
対照は同じseed・入力・10k scheduleの完了済み軌跡を再利用した。

次は10k結果を見る前に固定した [重み補間の追試](parameter-blend-plan-20261007.md) を行う。
この失敗条件をさらに学習することや採用基準を緩めることはしない。
別fine-tuning seed・事前固定した新しい合成入力の確認は未実施で、採用候補が通った後に行う。
