# 境界近傍lossの半係数比較（L1b）

2026-09-15開始。前回の[局所距離場loss比較](boundary-loss-experiment-20260914.md)では、
線幅が改善する一方、EMAで強い重ね描きの切断が増えた。
**係数を0.520737から0.2603685に半減した1条件を追加**し、保存済みの係数0・0.520737と比較する。
前回のdevelopment結果を見て決めた追試であり、未見testではない。

## 学習前に固定する設計

- 新条件は`half`。既存`control`・`boundary`は再学習せず、前回の2,000/10,000更新を再利用する。
- 初期値は3条件ともD1の`geometry20/raw`追加10,000更新時点。今回追加10,000更新で通算50,000相当。
  前回boundaryの最終重みからの継続ではない。optimizerはリセット、EMAは初期rawをコピー。
- `configs/boundary-half.toml`は、前回設定から`loss.boundary_weight`だけを変更。
  モデルS/M/L 5,933 / 10,206 / 17,893 parameters、再帰1〜6段、各段loss係数0、
  micro4×蓄積4、LR 1e-4×sqrt(16)、warmup500、schedule終点100,000を維持する。
- seed46、split42、train891/dev64/final64、生成教師20%、strokes拡張50%、Gaussianとmaskは同じ。
  final64は使わず、合成下描きのみ。Krita・配布モデルは変更しない。
- 補助lossの半径4と演算は変えない。係数は前回校正値の正確な半分で、再校正はしない。

## 既存結果の再利用条件

`scripts/boundary_reuse.py`で以下を検証してから本学習を開始する。

1. 前回の成果物hash・source archive・初期重みを検証する。数値処理sourceと依存lockのhashは前回と一致。
2. training/evaluation scriptは、新CLI選択肢を除いたASTが前回archiveと一致する。
3. 前回の係数0.520737で単独30更新と従来の2条件30更新を再実行。
   データ分割、初期化、全更新のscale/再帰/生成slot/keyが一致し、10/20/30のloss内訳と勾配を比較する。
4. CPUの小型実行では、単独・2条件実行のモデル/optimizer/EMAと中断再開が完全一致することをテストする。
5. 固定評価1,562ケース＋manifestをbyte単位で複製し、初期rawの4,686出力を再評価する。
6. 本学習の全10,000更新でもscale/再帰/生成slot/key、設定、sourceの一致を監査する。

GPUの短い再実行で微小な差が出たため、bit単位の一致とは区別する。
loss内訳はrtol1e-5、gradient normはrtol1e-4、共通atol1e-8で照合し、実測差を保存する。
この許容値は短い再実行差を確認した後、本学習前に固定する。
同一seedでも長期学習の浮動小数点差が増幅されない保証はなく、独立seed比較の代替にはならない。

## 評価・判断

2,000更新で有限性・保護画素の一致・clean F1の崩壊を確認し、通過すれば同じ設定で10,000へ進む。
半係数の初期rawと2,000/10,000のraw/EMA、全S/M/Lを評価する。
512pxのruntime再帰3段、NumPy noise、denoise1回で、前回と同じ評価器を使う。
新規評価23,430出力、再利用する学習後評価37,488出力、合計60,918出力（初期値の重複は数えない）。

主要な問いは、強い重ね描き幾何形状における**線幅改善と切断・消失の折り合い**。
半係数−補助なし、半係数−元係数を、raw/EMA別、S/M/L別に対応比較する。
clean、強いblur、背景残留、誤接続も併記し、F1や線幅だけで採用しない。
前回と同じ元画像/幾何variant単位のbootstrapを使う。学習seedの不確実性は測れない。
幅のprobeは濃さと近傍残留の影響を受け、接続probeも固定queryに限られる。

例示画像は事前にcase256（clean曲線）、479/480（強いblur）、1376（薄い線）を選ぶ。
同じ入力に対する3条件のSと、半係数のS/M/Lを表示する。
最終checkpoint、抽出S/M/L、S ONNX、実runtimeと既存BGRA APIの一致を確認して完了とする。

## 再現

```bash
# 同じ初期値・前回の係数で、短い再実行をそれぞれ行う。
.venv/bin/python scripts/compare_nested_training.py --study boundary_half \
  --config configs/boundary-loss.toml --run-dir runs/boundary-half-20260915/replay-full \
  --stop-after 30 --initialize-from runs/geometry-data-20260914/snapshots/step-10000/geometry20/raw
# 上記を --study boundary_loss / --run-dir .../replay-paired でも実行する。
.venv/bin/python scripts/boundary_reuse.py --root runs/boundary-half-20260915 \
  --reference-root runs/boundary-loss-20260914 \
  --reference-artifacts benchmarks/boundary-loss-20260914-artifacts.json
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_geometry_data.py --study boundary_half \
  --root runs/boundary-half-20260915 --reference-root runs/boundary-loss-20260914
```

本実行前にconfig・preflight・tests・source archive/hashをrun rootへ保存する。
controller/observerの再起動は既存processの終了を確認してから行う。
`summarize_geometry_data.py` / `audit_geometry_data.py`にも`--study boundary_half`を指定する。
結果・実測差・成果物の詳細は完了時に追記する。

事前検証は全138テスト成功（CPU204.64秒）。単独と2条件実行の小型3更新で、
モデル・optimizer・EMAが完全一致し、単独実行の中断再開も完全一致した。
実GPUでは半係数・実効batch16・再帰1〜6段の全6更新が有限で、保護領域の教師も一致した。

30更新のGPU再実行で、loss内訳の最大絶対差は約2e-7、勾配normを含む最大差は
単独−前回3.4332e-5、2条件再実行−前回2.4140e-5、単独−2条件再実行3.7670e-5。
最大相対差はそれぞれ4.7524e-5、3.2529e-5、7.0379e-5で、上記許容値内だった。
元の2条件実行でも微小差が発生するため、単独実行特有の差と断定しない。
データ分割・初期化・再帰段・生成keyは完全一致、実行/evaluation loopのAST一致も通過した。

## 2,000更新の中間結果

2026-09-15 11:48 UTC、崩壊検知を通過し、同じ設定で10,000更新へ継続した。
初期4,686出力の全指標は前回と完全一致。今回の中間9,372出力も有限で保護画素最大誤差0。
生成教師は32,000枠中6,421（白背景319）、全2,000更新のscale/再帰/生成keyが前回と一致した。

S・強い重ね描き幾何形状の中間値:

| 重み・指標 | 係数0 | 半係数0.2603685 | 元係数0.520737 |
|---|---:|---:|---:|
| raw F1 | .5110 | .5165 | .5190 |
| raw 線幅MAE px | 2.9550 | 2.7954 | 2.7049 |
| raw 切断率 | .3646 | .3667 | .3719 |
| raw 消失率 | .0797 | .0805 | .0824 |
| EMA F1 | .5116 | .5139 | .5154 |
| EMA 線幅MAE px | 2.7597 | 2.6686 | 2.6018 |
| EMA 切断率 | .3703 | .3781 | .3781 |
| EMA 消失率 | .0944 | .0948 | .0948 |

半係数−係数0の線幅差はraw −.1596px、対応付き95%区間[−.1901,−.1269]、
EMA −.0910px [−.1101,−.0709]。幅改善は残るが、EMAの切断は元係数と同じだった。
rawの切断差は +.00208 [0,.00625]、EMAは +.00781 [0,.02083]。
半係数−元係数のraw切断差は −.00521 [−.01563,0]で、確実な改善とは断定しない。
clean F1・強い重ね描き実画像F1は両係数の中間、強いblur実画像F1も中間で補助なしを下回る。
2,000時点の順位から採用を決めず、最終評価まで追う。

中間集計: `benchmarks/boundary-half-20260915-pilot.json`。
記録された学習時間3,564秒、定常更新平均1.151秒。
1条件実行では画像読み込み待ちが目立つため、2条件に比べた実時間短縮を単純な半分と見積もらない。
