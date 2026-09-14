# 幾何形状を教師に加える比較（D1）

2026-09-14。前回の [下描き拡張率比較](sketch-rates-experiment-20260912.md) から、
`p50/raw` の 10,000 更新モデルを共通初期値として使用する。
今回の比較は **既存 clean 線画 100% と、各画像を確率 20% で生成 clean 線画へ置換**。
下描きは既存の合成拡張のみ。実際の下描き収集・Krita への反映は行わない。

## 実行前に固定した条件

- `control` と `geometry20` の2条件。モデル・loss・optimizer・拡張の強さは同一。
- S/M/L は 5,933 / 10,206 / 17,893 parameters。内部3段、各block2畳み込み。
  channels は `[4,6,8]` / `[6,8,12]` / `[8,12,16]`、一方向の依存関係を維持する。
- crop 512、FP32、microbatch4 × 蓄積4。学習率は 1e-4 × sqrt(16) = 4e-4、
  warmup500、schedule終点100,000更新。再帰1〜6段、各段lossの係数0を維持。
- strokes拡張の適用率50%、opacity .25〜1、blur sigma0〜2、dilation最大2px、
  displacement最大2px。縮小後に各段で独立Gaussianを付加。強度0の画素は入力を保持。
- source split42、初期model seed42、今回のcrop/noise seed45。
  train891 / dev64 / final64。final testには触れない。
- p50 raw重みからoptimizerをリセットし、EMAも同じraw重みから開始。
  今回の10,000更新後は、それまでの学習履歴を含め通算40,000更新相当。
- 元画像のcrop列を両条件とも同じように読み込んでから置換する。
  幾何形状用のNumPy乱数はoptimizer step / itemを含む独立namespaceに分離。
  非置換slotの入力、下描き拡張の乱数、mask、Gaussian、再帰段数は揃う。
  置換slotは画像内容が異なるので、同じ乱数でも入力・教師は異なる。

## 学習用の生成器

`src/lineartgen/data/geometry.py` に専用の生成器を実装する。
評価用の `evaluation.line_case`、画像、probe、seedは学習には使わない。
直線・細線・Bezier曲線・角・分岐・交差・近接曲線・意図した隙間・入り抜きの9種。
1画像に1〜4群、長さ・角度・制御点・位置は連続乱数で変える。
線幅は .6〜12px の対数一様（細線は最大1.8px）。4倍で描画して縮小しサブピクセルを表す。
近接曲線は法線方向にずらす。群同士は重なり得る。

生成枠の5%は白背景、全体では約1%になる。
線幅はdataset scaleに応じて変えず、512px出力上の幅を全再帰深度で経験させる。
選ばれたslot、再生成key、線種、path数を各更新のtimingsに保存する。
同じ更新数で比較するが、geometry20が見る既存元画像の累積枚数は少なくなるため併記する。
生成器プレビューは `runs/geometry-data-20260914/generator-preview/`。

## 評価と継続条件

- 同じ固定1,562ケースを前回からbyte単位でコピー。既存元画像由来のclean/合成下描きと、
  学習用とは別実装の固定幾何形状/合成下描きを含む。
  繰り返し利用するdevelopment suiteであり、新規の未見testとは呼ばない。
- 推論は実際の512px時と同じ再帰3段、NumPy noise、denoise1回を主比較とする。
  initial raw、追加2,000/10,000更新のraw/EMAを全S/M/Lで評価。
- 2,000でfinite、保護画素誤差0、clean F1が `min(.30, initialの半分)` 以上なら
  両条件を10,000へ継続する。これは崩壊検知であり品質合格基準ではない。
- clean、強いblur、重ね描き、幾何形状を分け、active ink F1・背景・線幅・細線の消失・
  切断・誤接続・mask境界を比較する。元画像/生成形状variant単位の対応付きbootstrapを使う。
  区間はこの学習済みモデルに条件付けたもので、学習seed間の分散を表さない。
- 最終checkpointのoptimizer step、全サイズ同時計算と切り出しの一致、L専用変更時の
  S/M不変性を監査する。SのONNXをexportし、実際の再帰推論とBGRA入力も確認する。
- 次のloss比較は結果から選ぶ。線幅・位置・細線ならL1境界/距離、切断が支配的ならL2
  soft-clDiceを優先する。誤接続を増やす変更は一律採用しない。

## 再現

設定は `configs/geometry-data.toml`。実行rootは `runs/geometry-data-20260914`。
`training-source.json` に実行前のcommit・source archive・各fileのSHA-256を保存する。
controllerは `scripts/run_geometry_data.py --root runs/geometry-data-20260914`。
同時に別のcontrollerを立ち上げず、中断時は既存processの終了を確認して同じcommandで再開する。
停止要求はrootの `stop` fileで受け付け、全モデルの完全な更新と保存後に止まる。

集計は `scripts/summarize_geometry_data.py`。自動observerは既存controllerのPIDを監視するだけで、
学習を再起動しない。初期値・評価入力・各成果物のhashを検証して結果を保存する。

事前検証: 全129テスト成功（CPU 146.00秒）。実GPUで実効batch16、両条件×再帰1〜6段の
12更新が成功。生成例18枚を含み、勾配normは .656〜14.938、全lossが有限、共通maskと
保護領域の教師の完全一致を確認した。目視確認後の近接曲線・隙間の非接続もテストする。

## 2,000更新の中間結果

両条件で崩壊検知を通過し、設定を変えず10,000更新へ継続した。
共通初期値の再評価4,686出力は、前回p50/rawの指標と全て完全一致。
主比較の2条件×3サイズ×raw/EMA×1,562件＝18,744出力が有限、保護画素最大誤差0。
geometry20は32,000枠中6,362枚を生成教師に置換（19.88%）、うち白背景302枚。

raw Sの途中経過:

| 指標 | control | geometry20 |
|---|---:|---:|
| clean実画像 F1 | .9260 | .9182 |
| clean幾何形状 F1 | .8313 | .7966 |
| 強いblur下描き・実画像 F1 | .6859 | .7664 |
| 強い重ね描き・実画像 F1 | .7969 | .7783 |
| 強い重ね描き・幾何形状 F1 | .3974 | .5066 |
| clean幾何形状・切断率 | .1214 | .0318 |
| clean幾何形状・誤接続率 | .3556 | .4056 |
| clean幾何形状・線幅MAE(px) | .9637 | 1.2567 |

geometry20−controlのclean幾何形状F1差は −.0348、対応付き95%区間 [−.0425,−.0277]。
切断率差 −.0896 [−.1304,−.0543] と同時に線幅MAE差 +.2930 [.2087,.3847] が生じた。
強い重ね描き幾何形状ではF1 +.1093 [.0598,.1647]、切断率 −.2495 [−.3214,−.1750]、
線幅MAE +.5278 [.1276,.9358]。誤接続率の点推定も悪化するが区間は広い。
実画像の強いblur改善の区間は0を跨ぐため、この段階で一般的改善とは断定しない。
EMAでも切断減少と線幅悪化が併存した。

画像case256ではどちらも曲線を保持するが、geometry20に背景の残留が増える。
case480ではgeometry20がblurされた輪郭を残す一方、太さとノイズが増す。
データ追加による線の消失改善と、太線・背景・誤接続の代償を分けて最終結果で判断する。
中間集計は `benchmarks/geometry-data-20260914-pilot.json`。
