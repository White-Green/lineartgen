# 灰色線画の中心線loss比較（L2）

2026-09-15開始。接続性を促す補助lossを単独で比較する。
[距離場lossの半係数比較](boundary-half-experiment-20260915.md)では線幅と切断の両立が未解決だった。
今回は距離場lossを併用せず、同じ初期重みから `control` と `centerline` を新たに学習する。
L2は実験IDであり、二乗誤差のL2 normを意味しない。

## 学習前に固定した設計

共通初期値は `runs/geometry-data-20260914/snapshots/step-10000/geometry20/raw`。
optimizerをリセットし、EMAも同じraw重みから始める。両条件10,000更新、通算50,000更新相当。

- nested S/M/L: 5,933 / 10,206 / 17,893 parameters、チャンネル[4,6,8] / [6,8,12] / [8,12,16]。
  内部3段、各block2畳み込み、fused実行、residual出力。
- crop512、FP32、micro batch4×蓄積4=実効16。LR1e-4×sqrt(16)=4e-4、warmup500、
  schedule終点100,000。空間再帰1〜6、段間loss係数0を維持する。
  直接lossは最高解像度にかかり、下位段にも再帰を通じて勾配が流れる。
- 生成線画20%（その5%は白背景）、strokes拡張50%、opacity .25〜1、blur sigma0〜2、
  dilation最大2px、displacement最大2px。各段で縮小してから独立Gaussianを加える。
- split42、train891 / dev64 / final64。データ・ノイズseed46、final64は使用しない。
- 両条件のcrop・生成例・下描き拡張・mask・Gaussian・再帰段数は同じ。
  以前のcontrolを主比較に再利用せず、新しい対を学習する。

## 原論文から変更した点

[soft-clDice原論文](https://arxiv.org/html/2003.07311v7)のsoft skeletonと、
予測・教師の中心線の被覆を測る考え方を使う。
原論文のsoft-clDiceを灰色の教師へそのまま適用すると、積によるintersectionは完全一致でも
被覆率1にならない。今回はminimum intersectionに変更し、ReLUによる不足量として計算する。
**原論文のsoft-clDiceそのものの再現ではなく、灰色線画向けの実験的な変更である。**

正規化画像のink量を `a=clip((1-image)/2,0,1)` とする。
十字3×3のerosionと正方形3×3のdilationでsoft openingを作り、
`relu(a-opening(a))`を初期中心線にする。erosionと残差のsoft unionをさらに16回繰り返す。
paddingはpoolの対象外とし、丸め誤差による超過だけを `min(skeleton,a)` で抑える。
この有限回数で十分に侵食できない太い領域では、中心線を取り切れない可能性がある。

教師inkをY、予測inkをP、中心線をSy/Sp、編集可能画素をAとすると、画像ごとに

```
precision = 1 - sum(A * relu(Sp-Y)) / (sum(A*Sp) + 1)
recall    = 1 - sum(A * relu(Sy-P)) / (sum(A*Sy) + 1)
loss      = 1 - 2*precision*recall / max(precision+recall, 1e-6)
```

を計算しbatch平均する。平滑化定数は1画素相当。
両画像が二値なら、平滑化付きclDiceの積による比率と一致する。
灰色画像ではminimum intersectionを使うため異なる目的関数になる。
ReLUの不足量として書くことで、灰色の教師と一致したときlossだけでなく勾配も0になる。

形態学的処理の前に保護領域の予測を教師値へ置き換え、集計は編集領域だけで行う。
maskの端を人工的な線の端として作らず、保護領域の予測には勾配を流さない。
有効な教師中心線がない画像（空白、全面黒、全画素保護など）は補助loss0とし、既存lossに任せる。
画素・contrast lossは残す。foregroundだけを扱う有限反復・灰色・部分maskの変更であり、
原論文の二値条件での位相に関する理論保証はこの実験には適用しない。
実装は `src/lineartgen/training/cldice.py`。

## 係数校正と事前検証

学習前にtraining splitのみ、別seed1046の12バッチ（192枠）で勾配を測る。
再帰1〜6に2バッチずつ割り当て、重みは一度も更新しない。
校正ではdataset scaleと再帰深度を独立に割り当て、全深度を確実に含める。
係数1での補助勾配norm / 既存勾配normをrとし、
`λ=min(.10/median(r), .50/max(r))` を有効数字6桁で固定する。

測定結果は **λ=0.0526158**、重み付き勾配比の中央値 .09437、最大 .50000。
今回は最大値の制限が係数を決めた。学習全体で常にこの比率になる保証ではない。
`runs/cldice-loss-20260915/calibration.json` に各バッチのnorm・cosine・生成keyを保存した。
既存の境界lossの係数とは、目的関数の尺度が異なるため大小を直接比較しない。

全146テストが成功（CPU204.52秒）。独立した二値形態学実装との一致、灰色完全一致時の
loss/勾配0、保護領域勾配0、切断・薄い線・ずれ・欠落・誤接続への反応を確認した。
異なるlossでの更新差、モデル・optimizer・EMAを含む中断再開の一致も検証した。
補助loss無効時のloss/勾配は旧commit607726aと3設定で完全一致した。

## 評価と継続条件

固定1,562ケース（manifest込み1,563ファイル）をD1からコピーしてhashを確認する。
これは繰り返し使用しているdevelopment suiteであり、未見testではない。
既存clean線画からの合成下描きと、独立した幾何形状からの合成下描きを分けて集計する。

初期raw、2,000/10,000更新のraw/EMAを全S/M/Lで評価する。
主比較は512px時のruntime再帰3段・NumPy noise・denoise1回。
合計42,174出力。2,000更新時に全出力有限・保護画素誤差0・raw clean F1が
`min(.30, initialの半分)`以上なら設定を変えず10,000更新へ継続する。
この基準は崩壊検知であり、品質の採用基準ではない。

主な確認対象は強い下描きの切断率と細線消失率。誤接続、線幅MAE、背景残留、clean保持、
blur入力、部分編集のmask境界を併記する。線を太らせることで切断だけを改善する変更を
成功と扱わず、raw/EMA・S/M/Lを分けて代償を確認する。
対応付きbootstrapは元画像・幾何形状variant単位。学習seedは1個なので、区間は
その学習済みモデルに条件付けたものでありseed間の不確実性を表さない。

例示画像は学習前にcase256（clean曲線）、479/480（実画像由来のblur）、
1376（薄い隙間付き線、部分編集）を指定する。
過去の実験でも観察したケースなので、新規test例とは扱わない。
checkpoint・S/M/L抽出・S ONNX・既存runtime3段・BGRA APIの検証を行う。
Krita・配布モデルは更新せず、合成下描きだけで進める。

## 再現

設定は `configs/cldice-loss.toml`、実行rootは `runs/cldice-loss-20260915`。
学習前にsource archive・commit・設定・source hash・共通初期値hashを保存する。

```bash
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_geometry_data.py --study cldice_loss \
  --root runs/cldice-loss-20260915 --reference-root runs/geometry-data-20260914
```

稼働中のcontrollerを重複起動しない。停止はrootの`stop`ファイルで更新・保存後に行う。
同じstudyを指定して `summarize_geometry_data.py` / `audit_geometry_data.py` で集計・監査する。

GPU事前検証も完了した。両条件×全6再帰段、実効batch16の12更新でloss・勾配が有限、
更新に成功し、maskと保護領域の教師値が一致した。結果は `preflight.json` に保存した。

## 2,000更新の中間結果

2026-09-16 01:24 UTC、両条件が崩壊検知を通過し、設定を変えず10,000更新へ継続した。
初期4,686出力はD1 geometry20 rawの指標と完全一致。
2,000更新の18,744出力（2条件×3サイズ×raw/EMA×1,562）は全て有限、保護画素誤差0。
両条件の生成例は6,421枚（32,000枠、白背景319枚）で、slot/keyが全て一致した。

Sの強い重ね描き・幾何形状の結果:

| 重み | 指標 | control | centerline |
|---|---|---:|---:|
| raw | F1 | .510974 | .505508 |
| raw | 線幅MAE px | 2.954997 | 2.935707 |
| raw | 消失率 | .079720 | .107693 |
| raw | 切断率 | .364583 | .363021 |
| raw | 誤接続率 | .645833 | .666667 |
| raw | 背景MAE | .013355 | .012097 |
| EMA | F1 | .511571 | .508660 |
| EMA | 線幅MAE px | 2.759652 | 2.724065 |
| EMA | 消失率 | .094434 | .111773 |
| EMA | 切断率 | .370312 | .381771 |
| EMA | 誤接続率 | .625000 | .625000 |
| EMA | 背景MAE | .012087 | .011308 |

rawの消失率差は +.02797、対応付き95%区間 [.01222,.04766]。
EMAでも +.01734 [.00739,.02941]。背景の改善に対して、線を残す目的では悪化が見られる。
rawの切断率差 −.00156 [−.02083,.01667] は明確な改善とは言えない。
EMAの切断率差は +.01146 [約0,.02708]、F1差は −.00291 [−.00542,−.00072]。

clean実画像/幾何形状のF1はraw .91848/.80396→.91959/.80598、
EMA .92345/.82287→.92386/.82344と小幅改善する。
強いblur・実画像由来のF1はraw .77017→.76918、EMA .76856→.76633で、
両者の差の区間は0を跨ぐ。中間段階では接続性改善を確認できず、採用は判断しない。
事前の継続基準は品質向上ではなく崩壊検知なので、10,000更新まで同じ設定で観察する。

記録された学習時間は5,899秒（約1.64時間）。定常更新平均はcontrol1.124秒、
centerline1.575秒で、補助loss側は約40%増。独立した速度benchmarkではない。
中間集計は `benchmarks/cldice-loss-20260915-pilot.json`。
