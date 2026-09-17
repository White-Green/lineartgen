# ノイズ強度の学習分布比較（N1）

2026-09-17開始。利用時に入力の線画情報が残ることを踏まえ、低〜中強度の学習頻度を増やす。
[前回の中心線loss](cldice-loss-experiment-20260915.md)は採用せず、補助lossなしで比較する。
以下は学習前に固定した計画である。

## 比較する分布

ここでのtはnoise_levelで、入力は `source*sqrt(1-t) + Gaussian*sqrt(t)`。
標準偏差そのものではない。両条件とも各解像度へ縮小してから独立Gaussianを加える。

- `control`: tを[.001,1)から一様に選ぶ。従来どおり。
- `low80`: 80%を[.001,.5)、20%を[.5,1)からそれぞれ一様に選ぶ。

同じ一様乱数uを逆CDFで写像する。追加の乱数は消費しない。
u<.8なら `.001+(u/.8)*(.5-.001)`、それ以外は `.5+((u-.8)/.2)*.5`。
全面identityの確率10%は別の既存乱数で決め、両条件で完全に共通とする。
mask形状・元画像・幾何生成・下描き拡張・Gaussianの値・再帰段数も共通にする。

identityを除くtの平均は.5005→.3504、t≤.5の頻度は約49.95%→80%、
t>.9は約10.01%→4%となる。高強度の学習を残しながら利用域へ配分する一つの候補であり、
80%や境界.5が最適だとは仮定しない。今回はこの2値を評価結果で調整しない。
既存lossのt依存の重みを維持し、逆確率による補正は行わない。
したがって学習分布に応じた期待目的関数も変わる。単に同じ目的関数の分散を変える実験ではない。

## 固定する学習条件

共通初期値は `runs/geometry-data-20260914/snapshots/step-10000/geometry20/raw`。
2条件を新たに学習し、optimizerをリセット、EMAも同じraw重みから開始する。
追加10,000更新（通算50,000更新相当）。seed46、split42。

- nested S/M/L: 5,933 / 10,206 / 17,893 parameters、チャンネル[4,6,8] / [6,8,12] / [8,12,16]。
  内部3段、各block2畳み込み、fused実行、residual出力。
- crop512、FP32、micro4×蓄積4=実効batch16。LR1e-4×sqrt(16)=4e-4、warmup500、
  schedule終点100,000。空間再帰1〜6と段間loss係数0を維持。
  直接lossは最高解像度にかかり、下位段には再帰を通じて勾配が流れる。
- 生成線画20%（生成枠の5%は白背景）、strokes拡張50%。opacity .25〜1、blur sigma0〜2、
  dilation最大2px、displacement最大2px。補助lossはどちらも0。
- 学習891 / dev64 / final64。final64は使用しない。合成下描きだけで進める。
- 定期validationは両条件に同じcontrol分布の入力を渡す。
  主な品質判断は別途固定した推論評価で行い、異なる強度分布のtrain loss同士は順位付けしない。

比較runnerは同じgeometry設定の教師生成を1回で共有し、強度分布ごとに入力を作る。
この共有でデータを変えず、同じ幾何形状を2回描画する処理を省く。
各更新で画像ごとの最大tを保存し、全更新の乱数を独立したNumPyの逆CDF式で監査する。

## 事前検証と評価

テストでは混合分布・identity頻度、maskと教師の一致、下描き拡張とGaussianの完全一致、
6解像度で強度だけが変わること、checkpointのモデル・optimizer・EMAの再開一致を確認する。
実GPUで両条件×再帰1〜6、実効batch16の12更新を行ってから本学習を始める。

主評価は前回とbyte単位で同じ1,562ケース、runtime再帰3段、NumPy noise、denoise1回。
初期raw、2,000/10,000更新のraw/EMAをS/M/L全てで測る（42,174出力）。
2,000更新で全出力有限・保護画素誤差0・raw clean F1が `min(.30, initialの半分)` 以上なら
10,000更新へ継続する。これは学習崩壊の検知であり、採用基準ではない。

重点はt=.1/.5の強い下描きのF1・切断・細線消失・線幅・背景残留とclean保持。
全体集計に加え、強度・全面/部分編集別に結果を示す。
既存clean suiteのt=.3/.7も分離して確認する。
改善が切断・誤接続やclean保持の悪化を伴わないかを検討し、単一指標だけで採用しない。
対応付き95%区間は元画像／幾何形状variantを単位にbootstrapする。
単一seedの学習済みモデルに条件付けた区間であり、seed間の不確実性を表さない。

高強度の代償は、主suiteでt=.5だった非legacyの616ケースから別途固定したt=.9で診断する。
画像・下描き・mask・Gaussian seedは同じにし、編集画素のtだけを.9へ変更する。
内訳はclean実画像24、clean幾何形状64、下描き実画像由来144、下描き幾何形状384。
初期rawと最終raw/EMAを全S/M/Lで測る（9,240出力、主評価との合計51,414出力）。
clean幾何形状のこの診断は元のt=.5ケースと同じ部分編集に限る。
これは同じdevelopment素材を使った強度診断で、独立testや純ノイズ生成の評価ではない。

例示画像は過去と同じcase256、479、480、1376を学習前に指定する。
checkpoint、S/M/L切り出し、S ONNX、既存runtime3段、BGRA APIの互換性を確認する。
Kritaと配布モデルは変更しない。

## 再現

設定 `configs/noise-strength.toml`、実行root `runs/noise-strength-20260917`。
学習前にsource commit/archive、設定、初期重み、評価入力のhashを固定する。

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/strength_diagnostics.py \
  --root runs/noise-strength-20260917 --prepare-from runs/geometry-data-20260914/evaluation
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_geometry_data.py --study noise_strength \
  --root runs/noise-strength-20260917 --reference-root runs/geometry-data-20260914
```

稼働中controllerを重複起動しない。停止はrootの`stop`ファイルで更新・保存後に行う。
`summarize_geometry_data.py` / `audit_geometry_data.py` に `--study noise_strength` を指定して
主評価を集計・監査する。`strength_diagnostics.py --root ... --output ...` で
実際の強度分布・高強度診断・主評価の強度別区間を保存する。

## 開始前の確認

CPUの既存・追加テスト150件が通過し、追加したsampling監査のテストを含む
`tests/test_noise_strength.py` 4件も通過した。GPUでは両条件の再帰1〜6段が
finiteなloss・gradientを返し、保護領域の一致を確認した。
一様分布の入力生成は直前commit `d53b6c8` と、3 seed×6段×3配列の54配列で
bit exactだった。実測ログはrun内の `tests.log`、`strength-tests.log`、
`preflight.json`、`uniform-equivalence.json` に保存する。

## 2,000更新の途中結果（2026-09-17）

固定1,562ケース×2条件×S/M/L×raw/EMAの18,744出力がfinite、
保護画素は全て完全一致し、崩壊判定を通過した。条件・停止位置を変更せず10,000更新へ進む。

強いstrokesを加えた幾何線画、Sの対照→low80：

| 指標 | raw | EMA |
| --- | --- | --- |
| ink F1 ↑ | .510974 → .502147 | .511571 → .514094 |
| 線幅MAE px ↓ | 2.954997 → 3.615783 | 2.759652 → 2.758884 |
| 消失プロファイル率 ↓ | .079720 → .049870 | .094434 → .094857 |
| 切断率 ↓ | .364583 → .280208 | .370312 → .380729 |
| 誤接続率 ↓ | .645833 → .687500 | .625000 → .645833 |
| 背景MAE ↓ | .013355 → .017461 | .012087 → .012103 |

rawではM/Lも切断・消失が減る一方、線幅・背景が悪化した。Sの切断率差は
−.084375（source cluster 95%区間 −.128125〜−.043750）、線幅差は
+.660785 px（+.537040〜+.781342）。EMA Sの切断差+.010417は区間が0を跨ぐ。
実線画由来の強いblurに対するraw S F1は .770171→.793434、
強いstrokesでは .776670→.762080。この時点で全面的な改善とは判定しない。

実測32,000例のうちidentityは両条件とも3,231例。編集対象に限った強度≤.5の割合は
対照50.2485%、low80 80.4129%、強度>.9は9.8161%→3.8792%。
学習の0始まり更新indexから全乱数を再生成し、記録値との差は最大2.98e−8だった。
開始前評価中に監査側のindex解釈とテストfixtureを修正し、学習開始前に
`7d522f613368e2939aec3631d5b9c242c5bff9dc` でsourceを再固定した。
学習・推論コードにはこの修正による変更がない。元の固定記録も `startup-source.json` に保持する。

機械可読結果は `benchmarks/noise-strength-20260917-pilot.json`、
乱数監査は `runs/noise-strength-20260917/sampling-through-2000.json`。
