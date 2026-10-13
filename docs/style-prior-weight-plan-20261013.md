# DG: 学習済み線画priorの影響を強める

DFでは清書と灰色平均を判別できたが、画質はほぼ変わらなかった。
係数.2のpriorがGへ伝える勾配ノルムは、診断12batchで既存lossの平均7.45%。
DF prior raw6000のGとDを両条件で同一初期値とし、係数.2(moderate)と2(strong)を各6,000更新する。
方向やAdam適用後の効果は単純なノルム比では決まらないが、priorの寄与を一桁強める比較である。
両G/DのAdamとEMAは新規。両条件の最初500stepは同じ0→1 ramp。
D自体の学習式・LR・patch抽出・real候補分布はDFと同じ。Gだけ係数が異なる。

16ch/15,809パラメータのG、15,497パラメータのD、固定2pxスタイルglobal候補loss。
512合成形状、同じopacity/blur/warp拡張、深さ1–6、全段.5**level、batch4、
sqrt(batch)のpeak LR1e-4→cosine1e-5、DはAdam1e-4/b1=0/b2=.99とR1 gamma1。
train seed2026101330、eval2026101331。D初期値はDF保存状態であり、新規random criticではない。
source-onlyなので追加Gaussianに依存しない。入力画像は保持し、確率的生成の維持は主張しない。

CPUではDの厳密な重み転送と不正ファイル拒否、同じG/Dで係数だけ変えた初回の
D更新一致・G勾配/更新差を確認。既存priorの実更新・保存復元テストも行う。
GPU初期6段のG出力とD重みを完全一致確認。最初6更新再実行はatol1e-7/rtol1e-5。
6/6000のG/D/両optimizer/EMA状態復元と12raw/EMA exportは完全一致を要求。
全6,000batchの入力・候補・D注釈・rampのhashは一致させ、係数はarm設定として明記する。

同じ標準4split×64×3mask×2draw×3モデルと6劣化×64×2mask×2draw×3モデルの9,216行。
事前指定16枚を等倍確認。512px/uint8比較は入力とscriptを先に束縛して別記する。
灰色帯・端点・交点・滑らかさを重視し、欠落・架空線・塗りの崩れが増えていれば不採用。
教師一枚へのMSE、平均断面誤差、GAN lossだけでは成功判定しない。
合成のみ。Krita/ONNX配布・独立確認seed・封印データは未変更。
