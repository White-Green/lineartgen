# 合成下描きの再帰条件と拡張率の比較

2026-09-12開始。合成下描きのみ。前回のclean線画由来の入力1,562件をbyte単位で再使用し、final testは使用しない。Krita側と同梱モデルは変更しない。

## 1. 再帰段数の影響を分離する

前回のcontrol / tone / blur / strokes、raw / EMA、S/M/Lの保存済み10,000追加更新重みを使用する。denoise反復は1回に固定する。

- 対照: 前回と同じ、dataset scaleに応じた1–6段。幾何形状は1段。
- runtime条件: 既存の`pyramid_plan`と`stop_size=max(128, minimum_size)`に従う。今回の512×512入力は全件3段。
- 両条件で既存runtimeのNumPy Gaussian生成手順を使う。同じseedから高解像度→低解像度の順に生成するため、共通解像度の入力とノイズが完全一致する。
- 前回のJAX乱数による数値と直接差を取って段数の効果としない。今回新たに作る対照とruntime条件の対応付き差を使う。

入力は正規化済みの固定配列で、BGRA前後処理による量子化は含めない。Sでは既存runtimeの計算との照合が可能。M/Lは下位prefixの状態も保持するJAX計算であり、既存1ch insert ONNXへの対応は引き続きSのみ。

## 2. 拡張率だけを変える

前回の`strokes` raw 10,000追加更新重みを共通初期値とする。これは最初の事前学習10,000＋下描き適応10,000の重みであり、ゼロからの比較ではない。optimizerをリセットし、EMAは初期パラメータをコピーする。

| 条件 | strokes拡張の確率 | clean入力の割合 |
|---|---:|---:|
| p75 | 75% | 25% |
| p50 | 50% | 50% |
| p25 | 25% | 75% |

S/M/Lは5,933 / 10,206 / 17,893パラメータで固定。拡張の種類・強度、loss、noise、mask、optimizerなどは前回のstrokes条件と同じ。データ・noise seedは新しい44、初期化seed42、元画像の分割seed42。3条件で同じcrop・mask・Gaussianを使い、適用される画像では変形も同じ。適用対象はp25⊆p50⊆p75となる。

学習時の再帰`original_steps + scale_level`は1–6段、各段loss倍率0、最高解像度のlossから再帰を通る勾配を維持する。実効batch16（micro4×蓄積4）、base LR1e-4×sqrt(16)=4e-4、warmup500、100,000更新を終点とする固定scheduleを維持する。noise weight floor .05、編集領域のみcontrast、独立Gaussian、マスクは全体・帯・矩形1:1:1。

2,000追加更新でraw/EMA全件評価後、健全なら追加合計10,000まで継続する。継続判定は有限な全件評価、保護領域誤差0、rawのclean実画像・幾何形状F1が`min(.30, 初期値の半分)`以上。これは崩壊の検出であり、画質の採用基準ではない。低順位だけでは条件を打ち切らない。

## 評価と判断

拡張率比較はruntimeと同じ3段・denoise1回・固定NumPy noiseを主評価とする。共通初期値も同じ条件で評価する。

- clean実画像 / clean幾何形状 / 合成下描きの種類・強さごとの編集領域F1とMAE。
- 白背景（正解の正規化値≥.99、編集対象のみ）のMAEと黒い画素の比率。
- 細線の消失、線幅、位置、切断、誤接続、保護領域、mask境界。
- 下描き部分編集の幾何probeは引き続き除外し、保護領域に下描きを残す正解と混同しない。
- 元画像または(kind, variant)を単位とする対応付きbootstrap。単一の学習seedであり、学習seed間の不確実性は評価しない。

cleanの回復と下描き復元の低下を両方示す。F1だけの平均で採用せず、背景・切断・誤接続・線幅の悪化を併記する。改善が明確でなければ拡張率を確定しない。幾何形状の学習への追加、loss変更、モデルサイズ探索はこの比較に混ぜない。

## 実行・成果物

設定: `configs/sketch-rates.toml`。run root: `runs/sketch-rates-20260912/`。

```bash
export JAX_PLATFORMS=rocm,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PWD/runs/sketch-rates-20260912/jax-cache"
.venv/bin/python scripts/run_sketch_rates.py --root runs/sketch-rates-20260912 --phase recursion
# 再帰比較の結果を確認してから実行する。
.venv/bin/python scripts/run_sketch_rates.py --root runs/sketch-rates-20260912 --phase training
```

準備済みのsource hashと固定入力を各phaseの開始時に検査する。再開は前のプロセス終了を確認した後に限る。`stop`ファイルは完全な更新単位で停止する。完了済み評価は上書きしない。2,000/10,000更新のraw/EMA archive、最終optimizer checkpoint、S/M/L抽出重み、S ONNXと数値照合結果を保存する。

## 再帰比較の結果（完了）

4条件×3サイズ×2重み×2再帰条件×1,562件＝74,976出力を評価した。全出力が有限で、保護画素の最大誤差は0。全116テストと、実batch16・全6再帰段×3条件のGPU事前確認18更新が成功した。実S重みの32条件でJAXと既存ONNX runtimeも一致し、最大絶対差は1.4931e-5だった。

runtimeの3段に揃えたraw Sの編集領域F1:

| 学習条件 | clean実画像 | clean幾何形状 | 強いblur下描き | 強い太線・重ね描き下描き |
|---|---:|---:|---:|---:|
| control | .9485 | .9061 | .2274 | .2718 |
| tone | .9196 | .8285 | .7757 | .7595 |
| blur | .9192 | .8162 | .7851 | .7693 |
| strokes | .9226 | .8119 | .6460 | .7983 |

同じNumPy noiseで比べたraw Sのclean幾何形状F1は、dataset段数→runtime段数でtone .6075→.8285、blur .6043→.8162、strokes .7960→.8119となった。前回のtone/blurの大きな悪化には1段評価の影響が含まれる。strokesが全ての指標で最善とはいえない。

strokes Sのclean幾何形状に対する3段化の対応付きF1差はraw +.0160（95% source-cluster interval [.0082,.0237]）、EMA +.0841（[.0730,.0949]）。rawの白背景MAEは約.01564→.00477に減った。ただしruntime条件のcontrolは.00147であり、背景の誤差は残る。強い太線・重ね描きではrawの白背景MAEがstrokes .01272、blur .03620、tone .03764で、strokesを拡張率調整の出発点にする判断は維持する。

実画像由来の強い下描きで、strokesの3段化によるF1差の区間は0を含む。平均F1の小差だけで改善を断定しない。これらは合成入力・単一学習seedのdev評価であり、最終画質の採用判定ではない。

比較画像（上: dataset段数、下: runtime段数。列は教師・入力・ノイズ入力・control・tone・blur・strokesのS）:

![dataset recursion](assets/sketch-rates-20260912/dataset-raw-case-256.png)
![runtime recursion](assets/sketch-rates-20260912/runtime-raw-case-256.png)

集計: `benchmarks/sketch-rates-20260912-recursion.json`。全metricsとその他の比較画像はrun rootの`recursion/{dataset,runtime}-{raw,ema}/`。runtimeとの照合記録は`runtime-parity.json`。

## 拡張率比較

上記の結果を確認し、事前に固定した3条件の学習へ進む。結果は完了後に追記する。

