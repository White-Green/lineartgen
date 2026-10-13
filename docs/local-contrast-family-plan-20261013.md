# DH: 局所的に濃度を揃えた形状cue

DF/DGで出力priorを強めても灰色側線と交点の乱れは残り、強いpriorは薄い黒ベタを悪化させた。
入力透明度・縮小による線の濃度低下を、元の濃度だけから読み分ける難しさが一因という仮説を比較する。
これは原因と確定していない。元画像の情報を残し、形状を読み取る補助特徴を与える。

DD canonical raw6000からraw/local_peakを各6,000更新。GANなしのDD objectiveに揃える。
Gは同じ16ch・3残差block・15,809パラメータで、チャンネル数もパラメータ数も増やさない。
現在のsource-onlyは最初の2入力特徴が同じguideで重複している。そのstem kernelを両条件で同じように
第2特徴へ加算し、第1特徴のkernelを0にする。その後、第1特徴だけを以下のように変える。

- raw: 従来のguide。
- local_peak: ink=clip((1-guide)/2,0,1)、peak=各段の33x33 max pool、
  normalized_ink=.4*ink/max(peak,.02)、signal=1-2*normalized_ink。

第2特徴・残差anchor・保護画素は元のguideのまま。モデル自身が補助特徴を学習して利用する。
peakがfloor以上なら一様な透明度倍率に対しcueは不変。近くの濃線や入力ノイズの影響は受け得る。
正解の透明度/線幅/経路をモデルへ渡さない。33x33は各段のnative座標で、段番号は与えない。
初期両armの出力は完全一致を要求する。元親との演算順序は変わるので、全6段で
事前固定atol2e-5/rtol1e-5を要求し、最大絶対差を記録する。親とのbit exactは主張しない。

同じ512合成形状、2pxスタイル27候補、元線量の共通normalizer、opacity/blur/warp拡張。
linear Charbonnier + .25 path + .05 range、画像全体で一候補、全段.5**level。
batch4、sqrt(batch)のpeak LR1e-4→cosine1e-5、深さ1–6、fresh Adam/EMA。
train seed2026101332、eval2026101333。両armの全6,000入力・候補・乱数hashを一致させる。
source-onlyのため追加独立Gaussianに依存しない。入力画像そのものは保持する。

CPUで局所性・濃度不変性・floorとblankの有限性、初期転送・同パラメータ数・保存復元、
cue固有の実勾配と学習、保護画素、NumPy/JAX one-pass一致を確認。関連local/source/styleテストも実行。
未検証のlocal_peak ONNX exportは明示的に拒否する。既存rawモデルのexportは変更しない。
GPU最初6更新の再実行はatol1e-7/rtol1e-5、6/6000全状態復元と12raw/EMA exportは完全一致。

標準4split×64×3mask×2draw×3モデルと6劣化×64×2mask×2draw×3モデルで9,216行。
同じ16枚を等倍確認し、512px/uint8比較も事前入力/script束縛を行って別記する。
灰色帯・交点・端点・滑らかさと、欠落・架空線・黒ベタを確認する。教師MSEや平均指標だけで採用しない。
合成のみ。Krita/ONNX配布・独立確認seed・封印データは未変更。
