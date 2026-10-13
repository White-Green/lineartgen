# DC: 小型CNNの特徴と残差基準を元の下描きへ移す

DB追加診断でGaussian除去により粒状感が減った。ただし薄い灰色と輪郭リングは残った。
同じ小型CNN／複数許容清書lossで再学習し、追加ノイズ除去の効果と副作用を比較する。
過去のsource UNetや単一教師zero-noiseとは目的関数が異なる。新機構だとは主張しない。

DB augmented raw6000の同一15,809パラメータ、16chを共通初期値。fresh Adam/EMA、
各6,000更新、batch4、LR5e-5*sqrt(4)=1e-4→cosine1e-5、depth1..6、全段.5**level。
同じ512合成clean群、DBのopacity/blur/warp拡張、linear Charbonnier+.25path lossとgeometry/range。
train seed2026101321、eval2026101322。両armのすべての入力tensor/hashを一致させる。

- noisy: DBと同じ、追加Gaussian画像を特徴と残差の基準に使う。
- source: 特徴・残差の基準に元の下描きを使う。追加Gaussianは使わない。
  元画像自体のノイズはそのまま入力に残る。noise strengthの空間maskは維持。
  source + sqrt(t)*head、t=0では渡されたimageを保存。隠れ層・パラメータ数・再帰feedbackは同じ。
  追加Gaussianを変えたときの出力多様性は失う。従来のOT生成特性を維持したとはしない。

初期のforwardは因子が異なるため両armで一致しない。
source初期はnoisy初期へ追加Gaussian=0を与えた結果に全6段で完全一致させる。
sourceのnoise不変性、zero mask、serialization、JAX/NumPy one_pass orchestrationをCPUで検証。
progress FiLMは既存パラメータを維持し、one_passではu=1を渡す。
標準設定／モデル保存に意味を記録し、研究スクリプトだけのラッパーで偽装しない。

GPU最初6更新replayはatol1e-7/rtol1e-5、全状態restoreと12raw/EMA export完全一致。
数値ソースをcommit/freeze後に学習。従来4split×64×3mask×2draw×3モデル4,608行。
DB stress64形状×6劣化×2mask×2draw×3モデル4,608行。継続利用する開発セットであり確認用ではない。
DBからコピーするstress位相は元の固定値。旧opacity固定geometry指標はstressから除く。
標準等倍シートは4splitそれぞれwhole05(24/30/36/42)とpartial05(6/36/48/56)で8枚。
stressは6劣化whole05(6/24/36/56)、faint/rough partial05(6/24/48/56)で8枚。全16枚を実際に確認する。
粒状感、線の自然さ、入力追従、灰色、欠落、枝、保護、黒面を評価する。
線幅や位置を唯一の教師に一致させることは採否の目的ではない。自然な別候補も許す。

合成のみ。Krita plugin・配布は変更しない。NumPy runtimeのone_pass progress対応は研究ブランチ上で検証、
ONNX export／Krita GUIで動いたとは主張しない。独立学習seedの確認や製品採用は別判断。
