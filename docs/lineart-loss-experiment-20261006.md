# 線画lossの比較実験（2026-10-06）

目的は線周囲の余分な灰色を減らしながら、細線・線幅・接続と分離を維持すること。
[設計の検討](lineart-loss-design-20261006.md)に基づく最大モデル単体の比較。
現時点では実装・事前検証が完了した段階で、画質改善はまだ確認していない。
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

固定development 1,562条件を利用する。元線画由来の合成下描きと幾何線画、白紙、
弱い/強い劣化、全体/部分編集を含む。これは1,562枚の独立した元画像ではない。
既に使ったdevelopmentであり、新規の最終評価とは扱わない。

- primaryはEMA、runtimeと同じ再帰3段・denoise1回。rawも感度分析する。
- 線近傍の余剰不透明度、表示に残る割合、不透明度10〜50%の灰色割合を測る。
  遠方背景は別集計。F1もalpha10/25/50/75%で測る。
- 教師ink不足、幾何形状の断面幅・中心位置・消失・切断・誤接続を併記する。
  幾何probeは全体編集のケースでのみ使う。
- 固定31ケースの比較PNGを通常濃淡とKritaのalpha処理後の白背景合成で保存する。
  後者は実際のKrita GUIを撮影したものではない。
- 追加10k後に固定31ケースでdenoise1/2/5回を比較する。
- 有限値・保護画素の完全一致を毎評価で確認する。clean F1の大幅崩壊なら停止するが、
  この停止判定に通ることは画質改善・採用の根拠にはしない。

画質の判定ではregions/profiles/patch_priorをno_contrastとcontrolの両方に比較する。
にじみだけの改善や線の肥大化・白紙化は採用しない。層別平均と固定出力の失敗例を併記する。
独立な学習seedや新しい未観察条件での再現性は、この初期比較の次に必要な確認になる。

## 検証と再開

- 全既存テストを含む191件がCPUで成功（321.73秒）。
- 補助項0で既存loss/再帰勾配と一致、正しい灰色教師の追加loss/勾配0、保護画素勾配0を確認。
- 人工haloが50%閾値F1では見逃され、新しい指標で検出できることを確認。
- 生成器・判別器・両optimizer・EMAを保存再開し、連続学習と全状態が一致。
- 実GPUで全5条件×再帰1〜6段の30更新と全状態の保存再開が成功。
  この事前検証は使い捨て重みで、比較用checkpointを変更していない。
- 本学習は5条件まとめて250stepごとにatomic checkpoint保存。
  再開時にはcheckpointより先のログを退避し、未確定の更新を二重に数えない。
- 起点の重み・ソース・設定・評価入力のSHA256を保存し、各段階の開始時に照合する。

実行先: `runs/lineart-loss-20261006`。

```sh
JAX_PLATFORMS=rocm,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  JAX_COMPILATION_CACHE_DIR=.cache/lineart-loss-compilation-20261006 \
  .venv/bin/python -u scripts/run_lineart_loss_study.py --root runs/lineart-loss-20261006
```

`workflow-status.json` が処理段階、`progress.json` が学習進捗、`checkpoint.json` が保存済みstep。
`stop` ファイルで5条件分の更新を終えて保存したのち停止できる。
数値評価の終了後も、目視確認と結果レポートが終わるまで実験完了と扱わない。
