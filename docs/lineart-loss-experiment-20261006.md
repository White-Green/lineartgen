# 線画lossの比較実験（2026-10-06）

目的は線周囲の余分な灰色を減らしながら、細線・線幅・接続と分離を維持すること。
[設計の検討](lineart-loss-design-20261006.md)に基づく最大モデル単体の比較。
実装・事前検証・起点モデル評価が完了し、2026-10-06 04:29 UTCに本比較を開始した。
250stepの本学習checkpointをCPUで読み取り専用復元し、5条件のoptimizerと判別器の
更新回数250、全状態の有限値を確認した。画質改善はまだ確認していない。
Kritaプラグイン、ONNXモデル、配布artifactは変更しない。

## 固定した比較

| 条件 | loss | 補助係数 |
| --- | --- | --- |
| control | 現行のtone-balanced MSE + identity + edge + contrast | contrast=.01 |
| no_contrast | controlからcontrastだけを除去 | — |
| regions | no_contrast + 教師の線・近傍背景・遠方背景の誤差 | .3133429447 |
| profiles | no_contrast + 既知法線上の断面値と差分の誤差 | .2984959913 |
| patch_prior | no_contrast + 条件付き局所判別器 | .01、最初の1kで漸増 |

他の補助項を一度に組み合わせず、各項の効果をno_contrastと比較する。
controlに対する差で現行lossからの総合的な変更の影響も確認する。

- 起点は配布に使った最大UNetの100k時点EMA。
  `runs/completion-20260919-cached/arms/seed42_baseline/snapshots/step-100000/model/ema`。
  weights SHA256: `1f7745fc22a3d96ce47831f0b1d7c6da268a958ad24050b0da8ab3c9524c48ab`。
- モデルはchannels `[4,10,20,40]`、3conv/block、126,765 parameters、residual出力、post_pool。
  推論用の構造と入出力は同一。判別器は学習中だけ使う。
- 同一の初期G/EMA、同一のデータ・ノイズ・mask系列。各条件のoptimizerはリセット。
  学習seed=20261006。追加2kで中間評価、追加10kで初期比較を行う。
- 実効batch16（4×4累積）、base LR=1e-5、sqrt(batch)後の最大LR=4e-5。
  warmup100、10k cosine、minimum base LR=1e-6。判別器LRは生成側の半分。
- 学習再帰1〜6段、縮小してから独立Gaussian noise、段間loss重み0を維持。
  直接lossは最上位解像度だけ、下位段には再帰を通じて勾配が流れる。
- 従来と同じ元線画のtrain/dev/final=891/64/64、split_seed42。
  下描きはすべて合成。実際の下描きペアは使わない。
  strokes劣化率.5、幾何線画混合率.2、そのうち白紙率.05を維持する。
  既存キャッシュを再利用し、入力・教師の生成系列も全条件で共有する。

## 補助lossの実装

領域別lossは教師ink>.02を線の支持領域とし、6pxのmax-pool近傍から背景を分ける。
線内の滑らかなL1、近傍の余剰ink、遠方の滑らかなL1×.25を領域ごとに平均する。
正しい教師の灰色には罰則を与えず、保護領域への勾配は0。
評価側の6px近傍はEuclidean距離を使うため、学習の正方形近傍とは完全一致しない。

断面lossは幾何生成器のパスと幅を取り出し、接線から法線を求める。
straight/thin/bezier/taper/parallel/gapを対象とし、角・交点・端点を無理に処理しない。
画像ごとに最大16断面、各49点、線の半幅+6pxをbilinear sampling。
教師との差と隣接サンプル差の誤差を比較する。法線が分からない実線画にはこの項を適用しない。
この制限も含めた初期比較であり、全画像への方向推定の有効性は検証していない。

判別器は64px patch、3ch入力（noisy/strength/candidate）、8/16/32chのstride2 convとhead。
線内2・紙上1・一様1の座標を教師から選び、real/fakeで共用する。
保護画素の候補値は教師へ固定し、編集割合でpatchを重み付けする。
1回の生成側更新ごとに1回の判別器更新。両者とも実効batch16。
小モデルへ大モデル出力が間接的に混ざる問題を避けるため、今回の対象は最大モデルだけ。

係数はtraining splitのみの12batch（192例、各再帰段2batch）で事前に決めた。
固定初期モデルの**出力画像に対する勾配norm**を使い、中央値比10%・最大比50%を上限とする。
判別器項にはさらに.01の上限を設定した。モデルparameterの勾配norm合わせではない。
係数の最適性や学習後の勾配比を保証する手続きではなく、比較開始時の過大な項を避けるもの。
詳細はrunの `calibration.json` に保存した。

## 評価

固定development 1,562条件を利用する。元線画由来の合成下描きと幾何線画、
弱い/強い劣化、全体/部分編集を含む。これは1,562枚の独立した元画像ではない。
既に使ったdevelopmentであり、新規の最終評価とは扱わない。

- primaryはEMA、runtimeと同じ再帰3段・denoise1回。rawも感度分析する。
- 線近傍の余剰不透明度、表示に残る割合、不透明度10〜50%の灰色割合を測る。
  遠方背景は別集計。F1もalpha10/25/50/75%で測る。
- 教師ink不足、幾何形状の断面幅・中心位置・消失・切断・誤接続を併記する。
  幾何probeは全体編集のケースでのみ使う。
- 固定31ケースの比較PNGを通常濃淡とKritaのalpha処理後の白背景合成で保存する。
  後者は実際のKrita GUIを撮影したものではない。
- 31ケースの幾何線画が部分編集に偏っていたため、候補モデルの出力を見る前に
  全体編集24ケースを追加固定した（`supplemental-panels.json`）。8種の幾何形状それぞれで
  strong strokesのstrength .1/.5とstrong blurの.5を選ぶ。
  元の1,562条件の数値集計は変えず、接続・分離を目視する画像の範囲を補う。
  補足画像には100k初期モデルも並べ、追加学習だけによる差も確認できるようにする。
- 追加10k後に固定31ケースでdenoise1/2/5回を比較する。
- 既存セットに白紙だけの専用ケースがないため、別途512px白紙40条件を追加した。
  seed4種、strength=0/.1/.3/.5/.7、全体/中央矩形編集をdenoise1/2/5回で評価する。
  これは元の1,562条件へ後から混ぜず、背景への余分なinkを調べる独立の診断とする。
- 有限値・保護画素の完全一致を毎評価で確認する。clean F1の大幅崩壊なら停止するが、
  この停止判定に通ることは画質改善・採用の根拠にはしない。

画質の判定ではregions/profiles/patch_priorをno_contrastとcontrolの両方に比較する。
にじみだけの改善や線の肥大化・白紙化は採用しない。層別平均と固定出力の失敗例を併記する。
独立な学習seedや新しい未観察条件での再現性は、この初期比較の次に必要な確認になる。

## 検証と再開

- 全既存テストを含む191件がCPUで成功（321.73秒）。
- その後に追加した再開ログの退避テスト1件も成功（.90秒）。
- 補助項0で既存loss/再帰勾配と一致、正しい灰色教師の追加loss/勾配0、保護画素勾配0を確認。
- 人工haloが50%閾値F1では見逃され、新しい指標で検出できることを確認。
- 生成器・判別器・両optimizer・EMAを保存再開し、連続学習と全状態が一致。
- 実GPUで全5条件×再帰1〜6段の30更新と全状態の保存再開が成功。
  この事前検証は使い捨て重みで、比較用checkpointを変更していない。
- 本学習は5条件まとめて250stepごとにatomic checkpoint保存。
  再開時にはcheckpointより先のログを退避し、未確定の更新を二重に数えない。
- `audit_lineart_loss_checkpoint.py` は本学習のcheckpointを読み取り専用で復元し、
  全5条件のG/optimizer/EMAと判別器/optimizerの有限値・更新回数を検証する。
  節目では推論用raw/EMAの各parameterがcheckpointと完全一致することも検証する。
  250stepではG各126,765、判別器6,321 parametersであることも確認できた。
- 起点の重み・ソース・設定・評価入力のSHA256を保存し、各段階の開始時に照合する。
- 評価1,562条件の入力値域も確認した。全て[-1,1]内で、image_scale=1.1を含む
  runtime前処理から各入力を再現するBGRA層を構成できた。8bit化して実際の
  `prepare_input` に戻した差は最大.004313886、強度差は0。
  `bgra-input-audit.json` に保存。これは入力形式と値域の検証であり、量子化前後の
  モデル出力一致やKrita GUIでの操作を検証したものではない。

実行先: `runs/lineart-loss-20261006`。

```sh
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  JAX_COMPILATION_CACHE_DIR=.cache/lineart-loss-compilation-20261006 \
  .venv/bin/python -u scripts/run_lineart_loss_study.py --root runs/lineart-loss-20261006
```

`workflow-status.json` が処理段階、`progress.json` が学習進捗、`checkpoint.json` が保存済みstep。
`stop` ファイルで5条件分の更新を終えて保存したのち停止できる。
数値評価の終了後も、目視確認と結果レポートが終わるまで実験完了と扱わない。

`watch_lineart_loss_analysis.py` を独立した監視プロセスとして併用する。
checkpointの保持数が3なので、節目の推論重みが保存されたらすぐ読み取り専用の整合検証を
行う。その後、主評価完了を待って集計・白紙診断・補足画像を自動生成する。
制御プロセスの実PIDとコマンドを確認し、消失・異常終了・stopファイルで待機を終了する。
この監視は学習を起動し直さない。状態は `analysis-status.json`、ログは `analysis-logs/`。
補足GPU評価が後続学習と重なる時間があるため、本runの経過時間を純粋な学習速度の比較には使わない。

## 起点で確認した症状

本学習前のEMAで1,562条件、専用白紙40×3=120予測を評価した。
以下は改善後の値ではなく、以後の比較の基準。

- `real/strokes/strong` の近傍余剰不透明度は平均.15166、alpha50% F1は.83741。
  近傍余剰不透明度は各画像の線外6px以内の背景で平均してから、条件間で平均した値。
  全画像の15%が灰色という意味ではない。
- 同じ線画に強いblurを与えたcase479（strength .1）と480（.5）を白背景合成で目視した。
  .1でも目や鼻の細線に灰色とざらつき、.5では細線の崩れと教師にない接続が目立つ。
  これらは固定31ケースに事前指定してあり、学習後も同じ箇所を確認する。
- 白紙のstrength0は全て完全保存。strength .5では表示に残る画素数の最大が
  denoise1回169px、5回1,283px。0.7ではそれぞれ994px、7,868px。
  これはseed/全体・部分編集8条件中の最大で、同じ1枚での増分ではない。
  反復の悪化を監視する必要がある。
- 固定白紙case6/7の5回出力も確認した。これらの画像はほぼ白く見え、上記の最大ケースを
  表すものではない。目視した少数の白紙だけで異常なしとは判定しない。
- 補足幾何画像ではcase910（細線・strokes .1）で線の肥大化と周囲の灰色、
  case903（細線・blur .5）でほぼ全消失、case1295（平行線・strokes .5）で二本が一本へ合流、
  case1390（隙間・strokes .1）で意図的な隙間が閉じることを確認した。
  強い劣化で入力側にも線間の情報が失われる例を含み、lossだけで必ず復元できるとは扱わない。
  補足24例の数値504項目は主評価の同じ例と一致した（rtol=1e-5、atol=1e-6）。

起点画像は `evaluation/step-0-ema/`、白紙は `blank-evaluation/step-0-ema/`。
集計は `analysis-through-0.{json,md}`。ペア比較には元線画/幾何kind・variant単位の
cluster bootstrapを使い、同じ元画像に対する複数ノイズを独立標本と数えない。
同一予測のペア比較で2,056集計の差と区間が0になることも確認した。
学習seedの不確実性はこの区間に含まれず、多数の探索比較の補正もしていない。

2k/10kの評価後には以下も実行する（`STEP` は追加step、`MODE` はemaまたはraw）。

```sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/summarize_lineart_loss_study.py --root runs/lineart-loss-20261006 --through STEP

JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_lineart_blank_study.py \
  --root runs/lineart-loss-20261006 --step STEP --mode MODE

JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/render_lineart_loss_supplement.py \
  --root runs/lineart-loss-20261006 --step STEP --mode MODE
```

学習ソース固定commitは `5b05d12ed22fdec399e8b30bd07d632453e4abc0`。
その後の集計・白紙診断は独立したスクリプトで、学習や主評価の固定ソースには変更を加えていない。
