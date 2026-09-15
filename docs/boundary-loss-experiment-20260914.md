# 境界近傍の距離場loss比較（L1）

2026-09-14開始。前回の [D1比較](geometry-data-experiment-20260914.md) では、
生成線画20%により切断と線の消失が減った一方、強い下描きの線幅誤差・背景残留が増えた。
今回は同じ生成20%と下描き拡張50%を保ち、補助lossだけを変える。
L1は実験IDであり、絶対値誤差のL1 normを意味しない。

## 学習前に固定する設計

比較は `control`（既存loss）と `boundary`（既存loss＋局所距離場loss）。
両方とも `runs/geometry-data-20260914/snapshots/step-10000/geometry20/raw` を共通初期値に使う。
optimizerはリセット、EMAは同じraw重みから初期化。今回10,000更新後は通算50,000更新相当。

- nested S/M/Lは5,933 / 10,206 / 17,893 parameters。内部3段、各block2畳み込み。
- crop512、実効batch16（micro4×蓄積4）、FP32。学習率1e-4×sqrt(16)=4e-4、
  warmup500、schedule終点100,000。再帰1〜6段と各段loss係数0を維持。
- geometry20（生成枠の5%は白背景）、strokes拡張50%、opacity .25〜1、blur sigma0〜2、
  dilation最大2px、displacement最大2px。各段は縮小してから独立Gaussianを加える。
- split42、今回のデータ・ノイズseed46。train891 / dev64 / final64。final64は使用しない。
- 両条件のcrop・生成例・拡張・mask・Gaussian・再帰段は全て一致する。
  lossが異なる比較では学習目的関数を分けるよう、比較runnerの数値関数cacheを変更した。

## 補助lossの定義

[Boundary loss](https://proceedings.mlr.press/v102/kervadec19a.html) は輪郭の距離を
領域上の積分として扱い、[Hausdorff lossの研究](https://arxiv.org/abs/1904.10030) は距離変換や
形態学的処理を用いる。これらを参考にするが、今回は線画のアンチエイリアスを保つための
**局所的なsoft距離場の一致**を試す。原論文のlinear signed-distance lossや厳密なHausdorff距離の
再現ではなく、この線画課題向けの実験的な変更である。

正規化画像からink量 `a = clip((1-image)/2, 0, 1)` を作る。
`D_R(a) = (1/R) Σ[r=0..R−1] (1 − MaxPool_r(a))` とし、`R=4`に固定する。
MaxPoolは正方形の半径rで、二値画像ではRで打ち切り正規化したChebyshev距離に一致する。
同じ演算を灰色のinkに適用し、inkとpaper（1−a）の両方の距離場を比較する。

教師の両距離が1未満になる境界近傍で、予測と教師の距離場の二乗差を平均する。
maskが有効な画素だけを対象にし、画像ごとに対象画素数で正規化してbatch平均を取る。
保護画素にある予測は距離計算前に教師値へ置き換え、予測への勾配を遮断する。
空白・全面黒など境界のない画像と、全画素保護の場合は補助loss0とし、既存lossを残す。

この設計では予測が教師と一致すれば、灰色の境界でも補助lossとその勾配は0になる。
遠い背景や大きな形の誤りは既存lossが扱う。局所範囲に限定するため、全域の接続保証はない。
対象となる各再帰段のloss内部へ追加し、段間の重みと再帰を通る勾配は変えない。
現設定では最も高い解像度に直接lossを与え、下位段には再帰を通じて勾配が流れる。

## 係数校正

学習前にtraining splitの別乱数seed1046で12バッチ（192枠）を読み込み、
生成20%と下描き50%を同じように適用する。各再帰段1〜6に2バッチずつを割り当てる。
この校正だけは読み込んだdataset scaleと再帰深度を独立に割り当て、全深度を必ず確認する。
学習重みは一度も更新せず、development caseや画質順位は係数選びに用いない。

global clipping前の係数1での補助勾配normと既存勾配normの比 `r` を測り、
`λ = min(0.10 / median(r), 0.50 / max(r))` を有効数字6桁で固定する。
測定結果は **λ = 0.520737**、重み付き勾配比の中央値 .1000、最大 .4647。
これは校正バッチ内の値で、学習全体で常に同じ比率になる保証ではない。
`runs/boundary-loss-20260914/calibration.json` に各batch・生成key・loss・勾配norm・cosineを保存する。

## 評価・継続条件

- 前回とbyte単位で同じ固定1,562ケースを使う。既存画像由来と幾何形状由来の合成下描きを分ける。
  繰り返し使うdevelopment suiteであり、新規の未見testではない。
- 初期raw、追加2,000/10,000更新のraw/EMAを全S/M/Lで評価。
  主比較は512px時のruntime再帰3段、NumPy noise、denoise1回に固定する。
- 2,000では全出力有限、保護画素誤差0、clean F1が `min(.30, initialの半分)` 以上なら
  両条件を10,000まで継続する。これは崩壊検知で、品質の採用判定ではない。
- 重点は強い下描きの線幅MAE・背景残留。切断、細線消失、誤接続、mask境界、clean保持を併記し、
  線を消すことで背景や太さの指標だけがよくなる変更を採用しない。
- 元画像または幾何形状variant単位の対応付きbootstrapを使う。単一の学習seedに条件付けた区間で、
  学習seed間の不確実性を表さない。probe評価は全面編集に限るケースを明示する。
- 最終checkpoint、切り出したS/M/L、S ONNX、実runtimeの3段推論、既存BGRA APIと画像を検証する。

## 再現

設定は `configs/boundary-loss.toml`。実行rootは `runs/boundary-loss-20260914`。
実行前にsource archive・commit・設定・各source hashを保存する。

```bash
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_geometry_data.py --study boundary_loss \
  --root runs/boundary-loss-20260914 --reference-root runs/geometry-data-20260914
```

既存controllerが動いている間に再起動しない。停止はrootの`stop`ファイルで完全な更新・保存後に行う。
同じstudy引数を付けて `summarize_geometry_data.py` / `audit_geometry_data.py` で集計・監査する。
Krita・配布モデルは更新せず、合成下描きだけを使う。

事前検証: 全137テスト成功（CPU169.25秒）。正解の灰色境界でloss/勾配0、保護画素の
勾配0、位置ずれ・太線・切断・誤接続への反応、二値画像で打ち切りChebyshev距離との一致を確認。
異なるlossによる更新の差と、モデル・optimizer・EMAを含む中断再開の一致も確認した。
実GPUで両条件×全6再帰段、実効batch16の12更新が成功し、全loss・勾配が有限、
生成教師のslot/keyとmaskが一致、保護領域の教師も入力に完全一致した。

## 2,000更新の中間結果

2026-09-15 01:00 UTC、両条件が崩壊検知を通過し、設定を変えず10,000更新へ継続した。
初期値4,686出力の指標は、前回geometry20 rawの評価と完全一致。
2,000更新の18,744出力（2条件×3サイズ×raw/EMA×1,562）も全て有限、保護画素最大誤差0。
両条件が使った生成教師6,421枚（32,000枠の20.07%、白背景319枚）のslot/keyは全て一致した。

raw Sのcontrol→boundary:

| 指標 | control | boundary |
|---|---:|---:|
| clean実画像 F1 | .9185 | .9216 |
| clean幾何形状 F1 | .8040 | .8167 |
| clean幾何形状 線幅MAE px | 1.2079 | 1.0408 |
| clean幾何形状 切断率 | .0263 | .0271 |
| clean幾何形状 誤接続率 | .4333 | .3833 |
| 強い重ね描き・幾何形状 F1 | .5110 | .5190 |
| 強い重ね描き・幾何形状 線幅MAE px | 2.9550 | 2.7049 |
| 強い重ね描き・幾何形状 消失率 | .0797 | .0824 |
| 強い重ね描き・幾何形状 切断率 | .3646 | .3719 |
| 強い重ね描き・幾何形状 誤接続率 | .6458 | .6250 |
| 強い重ね描き・幾何形状 背景MAE | .01335 | .01243 |

対応付き95%区間はclean幾何形状のF1差 +.01275 [.01095,.01448]、線幅MAE差 −.1671
[−.1883,−.1458]。強い重ね描き幾何形状の線幅MAE差は −.2501 [−.3018,−.1970]、
F1差 +.00805 [.00524,.01071]、背景MAE差 −.000922 [−.001139,−.000710]。
ただし切断率差 +.00729 [0,.01979]、消失率差 +.00265 [−.00087,.00753] を併記する。
線を細らせてよく見える指標だけで採用せず、最終結果で代償を確認する。

EMA Sでも線幅は2.7597→2.6018px、F1 .5116→.5154、背景MAE .01209→.01141となったが、
切断率は .3703→.3781。raw/EMAとも課題を残す。例示case1376の薄い線も両条件で弱い。
中間集計は `benchmarks/boundary-loss-20260914-pilot.json`。
