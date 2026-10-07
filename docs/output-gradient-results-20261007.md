# 出力clip代理勾配の2k結果

2条件とも採用・10k延長なし。forwardを変えずに飽和域へ勾配を渡すだけでは、線の保持と背景の清潔さを両立できなかった。
比較条件と固定採用基準は [実験計画](output-gradient-study-20261007.md)。
同じ100k EMA起点・seed20261007・入力・2k更新で、完了済みhard clip対照と比較した。

| EMA条件 | real strong strokes halo改善 | geometry strong strokes halo改善 | 2k判定 |
| --- | ---: | ---: | --- |
| ste_control | 8.52% | 14.03% | 線の消失が悪化、23判定に失敗 |
| ste_ink_guard | 0.22% | 2.98% | 保持のscreen条件は通過、haloの5%条件に未達 |

ste_controlではgeometry strong blurのink不足が5.83ポイント、profile欠落が12.19ポイント増加した。
strong toneもink不足8.14ポイント増。rawはさらに悪化し、halo数値の改善を有効な復元と解釈できない。

ste_ink_guardではstrong blurのprofile欠落が3.13ポイント減り、ink不足も0.38ポイント減った。
ただし遠方背景のopacity excessが約0.00047増加。strong strokesではbridge率が2.08ポイント増加した。
rawもhalo改善2.44% / 4.62%でscreen未達。最終基準ではEMA 63、raw 67の重複を含む判定に失敗した。
これらは独立した欠陥数ではない。既存hard対照は再利用軌跡で、別seedではない。

## 飽和の追加診断

固定55panelと失敗例8件、計63入力の全3再帰段で、clip直前値を記録した。
通常推論との出力完全一致を確認しており、診断はモデルを書き換えていない。
geometry strong blurの選択15件では、最高解像度のink不足量のうち白側clip外にある割合は
初期51.65%、ste_control 71.35%、ste_ink_guard 48.83%。代表性のある全評価集合の推定ではない。
ste_controlのclip直前最大値はこの群で529、ste_ink_guardは5.97だった。

局所計算では、元のcontrast項と代理微分の組み合わせに不都合がある。
正しい白へ飽和したraw=2、教師=1でもloss=0に対して総勾配−0.01が残り、さらに白方向へ更新する。
正しい黒raw=−2、教師=−1でも逆向きの同様の勾配がある。contrastなしのink_guardは両方0。
これは正確な局所診断だが、観測されたgeneratorの悪化をこの項だけに帰属する因果比較ではない。

## 出力確認と保存

EMAのgeometry24例、rawのstrong blur8例、realの合成劣化3詳細、4例の1回/5回推論、
部分編集の白紙1例を実際に確認した。薄い直線・曲線・接続の消失、近接線の融合、
線の周囲の灰色や背景の点状汚れが残る。5回反復も誤った線を強める例がある。

白紙40条件×1/2/5回の最大平均opacityはste_control EMA 0.000000918、raw 0、
ste_ink_guard EMA 0.02096、raw 0.02179。ほぼ白紙になる前者も、線の消失を伴うため採用できない。
全定義済みmetricは有限、保護領域は完全一致。checkpointとraw/EMA書き出しも一致した。

保存先は `runs/output-gradient-20261007-screen`。
`analysis-with-baselines-2000.json`、`quality-gate-with-baselines-2000.json`、`review.json`、
`completion-audit.json`、`posthoc-saturation/results.json`、`contrast-saturation-stationarity.json`
に数値・画像hash・診断を保存した。数値sourceは859377f。

次はcontrastを外したSTE単独条件、既存neighbor保持lossとの組み合わせ、
教師の背景に発生した局所的な誤inkを強く評価する項を切り分ける。
今回の不合格条件をそのまま延長せず、同じ採用基準で新しい比較を行う。
