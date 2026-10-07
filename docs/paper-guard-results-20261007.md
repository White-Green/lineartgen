# 背景lossとcontrast除去の2k結果

3条件中、ste_neighbor_guardだけが固定screenを通過した。全状態を保って10kへ継続する。
最終採用条件は全条件が未達で、モデルの採用・Krita更新は行わない。
[計画](paper-guard-study-20261007.md) に従い同一100k EMA、seed20261007、同一入力で比較した。

| EMA条件 | real strong strokes halo改善 | geometry strong strokes halo改善 | screen |
| --- | ---: | ---: | --- |
| ste_no_contrast | −7.77% | −5.02% | halo悪化 |
| ste_neighbor_guard | 6.50% | 8.79% | 通過 |
| ste_ink_paper | 2.19% | 4.10% | halo未達、mild blurのbridge増加 |

比較対象は同じ2k時点のhard control。rawでもste_neighbor_guardだけがscreen通過した。
rawのhalo改善は9.73% / 11.04%。EMAのhalo差のcluster 95%区間はreal −0.01274〜−0.00696、
geometry −0.02211〜−0.01443。realは定義済み46ケース・12source、geometryは128ケース・32群で、
別学習seedの不確実性を表す区間ではない。

## 保持と背景の副作用

ste_neighbor_guardのgeometry strong blurではprofile欠落が2.70ポイント減った一方、
ink不足は0.68ポイント増えた。strong strokesも欠落は0.99ポイント減り、ink不足は0.96ポイント増。
遠方背景opacity excessはこの2群で約0.00030〜0.00032増加した。
最終基準の失敗はEMA 67、raw 81判定。条件・stratumが重なるので独立した欠陥数ではない。

新しいpaper項は対応するste_ink_guardに対して背景を改善した。
geometry strong strokesの対control背景差は約0.00049から0.00020へ減った。
ただしEMAでhaloの5%条件に届かず、geometry mild blurのbridge率が対controlで4.17ポイント増加。
rawはhalo 5.07% / 6.00%改善でも、strong strokesとmild blurのbridge増加で不合格。
この条件はそのまま10kへ延長しない。

白紙40条件×1/2/5回における最大平均opacityは次のとおり。異なる条件の最大値同士の比較である。

| 条件 | EMA | raw |
| --- | ---: | ---: |
| ste_no_contrast | 0.01911 | 0.02279 |
| ste_neighbor_guard | 0.01744 | 0.01984 |
| ste_ink_paper | 0.01228 | 0.01212 |

## contrast干渉の切り分け

同じseed・入力のste_controlからcontrastだけを除いたste_no_contrastでは、
geometry strong blurのprofile欠落は対hard controlで3.18ポイント減った。
contrastありSTEで観測された12.19ポイントの増加と対照的である。
選択したstrong blur15件のclip直前最大値も、contrastあり約529からcontrastなし約6.04へ減った。
前回の正しい飽和白にも外向き勾配が残る局所診断と整合する、同一seedでの切り分け結果である。
別seedでの再現や、contrast除去だけで線画品質を解決したという意味ではない。

63入力×全3再帰段の追加診断では、通常推論との完全一致を確認した。
強いblurの選択15件で白側clip外にあるink不足量の割合は初期51.65%、
ste_no_contrast48.58%、ste_neighbor_guard48.97%、ste_ink_paper49.72%。
代理勾配を通しても、飽和した細線欠落の多くは残る。

## 実際の画像と保存

EMA geometry24例、raw strong blur8例、real合成下描き3詳細とraw1詳細、
4例の1回/5回反復、部分編集白紙1例を確認した。
細い直線・曲線の消失、近接平行線の融合、隙間をつなぐ誤り、線の周囲の灰色が残る。
反復で誤った線が濃くなる例もある。継続候補はcontrastありSTEのような新たな全面的消失を示さないが、
ここで満足な品質に達したとは判断しない。

root: `runs/paper-guard-20261007-screen`、数値source: 9fc0b40。
`analysis-with-baselines-2000.json`、`quality-gate-with-baselines-2000.json`、
`selection-review.json`、`completion-audit.json`、`posthoc-saturation/results.json` に結果を保存した。
全定義済みmetricは有限、保護領域は完全一致、checkpointとraw/EMA書き出しも一致した。
180数値source・11設定を固定し、227テストと3条件×6段のGPU更新を検証済み。
paper項が無効の経路では、既存STE対照の6段更新後のmodel・optimizer・EMAも全配列完全一致した。

10k継続には `fork_paired_study.py` を使い、選択済み1条件のmodel・optimizer・EMA・stepを
保存して独立に復元・照合する。完了済みhard対照の10k軌跡を再利用するため、対照の再学習はしない。
別fine-tuning seedと事前固定した新しい合成入力の検証は、10k採用条件を満たした後に行う。
