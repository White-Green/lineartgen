# 局所濃淡分布の診断：学習モデルの成功ではない

進行中のprocedural3条件は固定したまま、別の人工例でlossの性質を調べた。
`local_transport.py` はこの時点で既存trainerから呼ばれず、配布・推論にも影響しない。

[Heitz et al., CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Heitz_A_Sliced_Wasserstein_Loss_for_Neural_Texture_Synthesis_CVPR_2021_paper.html)
は特徴分布を合わせるsliced Wasserstein lossをテクスチャ合成に使う。
[著者実装](https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis/blob/master/texture_optimization_slicing.py)
でも特徴の射影・sort・比較を確認した。以下はVGG特徴も射影も使わない、
16px tile内の1次元不透明度に対する独自の簡略化であり、論文の再現ではない。
教師ink/paper rankの別正規化とtile分類を加えるため、全体をWasserstein距離とは呼ばない。

予測と教師の不透明度をtileごとにsortし、対応する値の誤差を測る。
教師のAAは連続値のまま残す。保護画素は教師合成値に固定し、勾配を与えない。
白は物理不透明度≤0を許し、灰色・黒は有限の教師値と比較する。
最初に黒も≥1の区間として許した案は、偽の過剰な黒線が無料の黒rankを占有し、
正しい別の線だけに白化の勾配が出る反例で失敗した。この案は採用せず、
黒にも有限値への罰則を設けた。その代わり、見た目が正しい黒でも物理値が過剰なら
lossがかかるという副作用を明記する。

人工例は幅.6/1.4/3.4pxのAA線を±1/±2pxずらした二つの教師を等確率と仮定したもの。
41種類の線形混合、両方のinkを残すunion、白紙の43候補を比較した。
これはlossが曖昧さにどう反応するかの制御実験であり、実入力の条件付き分布が
この二択になっているという証拠ではない。NNの学習も推論も行っていない。

* 現在のsqrt-tone pixel lossは、opacity-deficit.3を残す場合も除く場合も、6例すべてで
  二つの線を両方残すunionを選ぶ。MSEが単に平均像を選ぶ、という説明だけでは不十分。
* sort後の二乗誤差を加えても、検査した係数.1/.3/1/3ではunionや中途半端な混合が残る。
* smooth absolute（sqrt(d²+1e-6)−.001）では係数3で、deficitの有無にかかわらず
  全6例が元の鋭い候補を選ぶ。係数10でも同じ傾向。これは43候補内の比較で、
  全画像空間における最適性ではない。学習係数を3とする次の仮説を支える程度の結果。

教師と一致するAA画像のloss/gradient0、白の許容区間、過剰黒への修正勾配、
保護領域を検証した。pixelの並べ替えだけで同じ分布になる画像は、このlossでは区別できない。
実際にshuffle画像もloss0になる反例をテストした。位置・滑らかさ・接続は既存の
画素・形状lossと実画像確認で引き続き検査する必要がある。

二乗版は `runs/local-transport-oracle-20261009/`、smooth absolute版は
`runs/local-transport-l1-oracle-20261009/`。plan、入力配列、全候補の値、実行コードの
hashとコピーを保存する。原案の区間黒の失敗は上記テストの開発段階で判明した。
これは学習実験や採用候補の検証完了ではない。
