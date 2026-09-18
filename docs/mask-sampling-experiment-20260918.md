# 矩形maskの境界sampling比較（N2）

2026-09-18開始。[N1の低〜中強度80%案](noise-strength-experiment-20260917.md)は採用せず、
一様強度・補助lossなしの対照から、矩形maskの置き方だけを比較する。以下は本学習前の固定計画。

## 変えるもの

`control` は従来の画像非依存mask。全面・帯・矩形の比率は各1/3、identityは10%。
`boundary8` では矩形だけについて、同じ幅・高さを持つ8位置から、境界を横切る
下描きのinkが最も多い位置を選ぶ。候補0は従来の位置、残り7つは画像内の整数位置を一様に選ぶ。
同点は最初の候補を選ぶので、白背景のmaskは変わらない。

scoreは下描きから `clip((1-source)/2,0,1)` を計算し、矩形辺の両側で隣接する
2画素のinkの小さい方を辺に沿って合計する。画像外との辺は0。
これは両側に線がある箇所のsampling用proxyで、接続性を保証する指標ではない。
prefix sumを用い、候補数×512²のmaskを全て保持しない。

- 既存矩形を画素単位で抽出して移動するため、面積・幅・高さ・強度は保つ。
- 全面・帯状mask、identity、元画像・幾何生成・下描き拡張・Gaussianの乱数は保つ。
- 候補の乱数は既存map keyから用途専用のfold-inで得る。強度や他の乱数列を進めない。
- scoreに使うのは拡張後の入力下描きのみ。正解線画を参照するmask生成は使わない。
- maskに合わせ、編集画素の教師はclean、保護画素の教師は入力下描きとする既存仕様を維持する。

元画像は共通だが、移動後の編集領域に含まれるink量も変わり得る。
境界samplingという方針全体の比較であり、境界の接続情報だけの因果効果を分離する実験ではない。
bandの境界位置や曲線maskまで対象にする比較も今回は含めない。

## 学習と対照の再利用

初期値は `runs/geometry-data-20260914/snapshots/step-10000/geometry20/raw`、
optimizerとEMAをリセットして追加10,000更新する。seed46、split42。
S/M/Lは5,933/10,206/17,893 parameters、チャンネル[4,6,8]/[6,8,12]/[8,12,16]。
内部3段・各block2畳み込み・fused・residual。512px、FP32、実効batch16、
LR1e−4×sqrt(16)=4e−4、warmup500、schedule終点100,000。
空間再帰1〜6、scale_loss_multiplier=0（直接lossは最高解像度、下位段へは再帰を通じた勾配）。
生成線画20%・strokes50%、opacity .25〜1、blur0〜2、dilation2、displacement2。
学習891/dev64/final64で、final64は使わない。実下描きは使わない。

一様maskのcontrolは `runs/noise-strength-20260917` の保存済みモデルを再利用する。
N1のcontrolはさらに過去のL2 controlとも2k/10k raw/EMAの重みがbit exactだった。
N2では以下を確認してから、新たにboundary8だけを学習する。

1. controlの設定が既存設定と完全一致し、optimizer・モデル・データ処理の数値コードをhash照合。
2. 新機能を無効にした入力生成が直前commitと3 seed×6段×3配列でbit exact。
3. 実GPUでcontrolの最初の30更新を再実行し、data split・初期値・scale/再帰・生成キー・
   強度と、10/20/30更新のloss・勾配等を照合。FP32許容差はloss等rtol1e−5、
   gradient norm rtol1e−4、atol1e−8。
4. 再利用モデル・optimizer checkpoint・評価・初期値・乱数記録を本学習前にhash固定。
5. controlを固定1,562ケースで再評価し、追加の境界指標を除く既存結果が完全一致することを確認。

controlは今回独立に学習した条件ではなく、速度も同時期の交互実行比較ではない。
記録には新規学習と再利用を明記し、元の学習・評価ファイルは書き換えない。
定期validationには両条件で従来の一様maskを使う。

## samplingの確認

最初の128学習batch（2,048画像）を学習前に再現し、強度・面積と全面/帯/identityの一致、
矩形境界のscore増加、変更された矩形数、score>0の割合を測る。
最初の変更例4つを図示する。品質に基づくcandidate数の調整はしない。
全10,000更新の強度・scale/再帰・幾何生成キーをcontrolの既存記録と照合し、
最初の128更新では記録された面積・強度が事前再現と一致することも確認する。

## 固定評価と判断

主評価は従来の1,562ケースをbyte単位でコピーする。入力素材・mask・Gaussian seedは
両条件に共通で、推論時には学習用mask配置を使わない。runtime再帰3段、NumPy noise、
denoise1回、全S/M/L。初期raw、2k/10kのraw/EMAを測り、計42,174出力を検証する。
2kで全出力finite・保護画素誤差0・raw clean F1≥min(.30,initialの半分)なら10kへ進む。
これは崩壊検知であり、採用判定ではない。

従来のink F1、幅、消失、切断、誤接続、背景、mask境界MAEに加え、
**編集側だけの3px境界帯**でMAE、教師ink画素のMAE、背景画素のMAE、ink F1を測る。
ink F1はその帯に教師inkがある場合だけ集計し、空白例の1.0で平均を水増ししない。
保護画素の完全一致による0も編集側の平均に混ぜない。
全体集計のほか、全面/部分と強度.1/.5を分けて示す。

中心となる判断は、部分編集で境界の線が残り、背景残留・誤接続が増えないこと。
clean保持と全面編集の品質も確認し、局所MAEだけで採用しない。
roughの部分編集では教師自体がclean/roughを継ぐため、既存の全経路topology probeを
適用しない。局所F1/MAEは接続性の厳密な保証ではなく、clean幾何線画の接続probe、
既存の全面編集probe、画像の目視と併せて判断する。

対応差の95%区間は元画像/幾何variant単位のbootstrap。単一学習seedに条件付けた
development比較であり、学習seed間の不確実性や未知の実下描きへの一般化は示さない。
例示画像は従来のcase256/479/480/1376を事前指定する。
checkpoint、prefix独立性、S/M/L切り出し、S ONNX、runtime3段、BGRA APIを確認する。
Kritaと配布モデルは変更しない。

## 実行・保存

設定は `configs/mask-sampling.toml`、runは `runs/mask-sampling-20260918`。
`mask_reuse.py` で再利用証拠を固定し、`run_geometry_data.py --study mask_sampling` が
2k gate→10k→モデル検証/ONNXを実行する。main trainingには
`compare_nested_training.py --only-method boundary8` を用いる。
`stop`ファイルがある場合は更新完了時に保存して停止する。既存controllerを重複起動しない。

主集計は `summarize_geometry_data.py` / `audit_geometry_data.py` の
`--study mask_sampling`、sampling監査は `diagnose_mask_sampling.py --audit-training`。

## 開始前に確認できたこと

最初の128 batchの2,048画像のうち、非identityの矩形は592例、位置が変わったのは389例。
境界を横切るsource inkの平均は49.6253→92.9739（約1.87倍）、
score>0の割合は69.2568%→77.8716%。面積・強度・全面/帯/identityは全例で一致した。
これは学習機会を増やせた確認であり、復元品質の改善ではない。

GPU preflightでは2条件×空間再帰1〜6がfiniteで、保護領域の教師は完全一致した。
無効時の入力生成54配列、controlの30更新再実行の10/20/30更新の全記録値も
bit exactだった。全テストではONNX export後のcumsum wrapperとの呼出し互換性を修正し、
位置引数をkeyword引数にした。samplingの数式は変えていない。

修正後の全156テストが通過。GPU preflightも修正後に再実行し、全12更新が通過した。
CPUの3更新によるpaired/single実行と再開比較では、モデル・optimizer・EMAが完全一致した。
