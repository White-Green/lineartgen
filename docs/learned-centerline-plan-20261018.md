# HR: 不透明な局所graphを実際に予測する小規模試験

HQの1px表現を入力から学習する。DQ/HQの正解pathを推論へ渡さない。HP raw1200の特徴抽出部だけを初期値とし、出力を9chの新しいheadへ置換する（頂点logit、cell内xy、log幅、4方向edge logit、塗りcell logit）。D4成分を平均する制約のない線形headを使用。存在/辺/塗りはlogit0で二値化し、存在する両端だけを不透明capsuleで接続する。孤立点は円として残す。座標はsigmoid、幅はexp(clip(log幅,−8,log8))、4x coverageで描画する。確率をopacityに使わず、恣意的な細片除去や推論threshold探索はしない。

1セル1頂点では近接したsubpixel線を区別できない。交差/同セル衝突、coarse段での合流も記録する。塗りはhardなpixel cellなので斜めの輪郭には制約がある。本試験の教師はDQのpeak2px線とpixel整列長方形。一般的な線幅、複数slot、任意形状の塗りを解決したとはしない。

学習はDQ probe00/12/18/24/36/42/48/54/56/58の10形状、各1200更新、B4、seed2026101807。凍結した粗いsourceに追加opacity倍率.65〜1とGaussian σ0〜1を加える。位置変換はしない小規模適合であり、形状や位置への汎化とは分ける。75%whole、25%右側部分適用、強度.5。入力を縮小してからGaussianを加える既存前処理を使い、現行trunk同様Gaussian値自体は特徴に使わない。

深さはstep%6+1、全段を.5^levelで重み付けして平均する。頂点/辺/塗りは正負をそれぞれ個数正規化したBCE、座標はoccupied cellのsigmoid xyにHuber、幅はoccupied cellのlog幅にHuber、5項の係数は全1。画像MSEは使わない。teacherは各段で元pathを縮小してgraphへ再変換する。部分適用でも内部graphは全体のclean形状を教師とし、最終raster合成でstrength0のsource値を完全保持する。内部帰還は予測頂点/塗りのsoft占有の1ch要約で、表示画像のopacityではない。教師や教師graphを帰還しない。

AdamWは既存設定、base LR1e−4×sqrt(B4)=2e−4から2e−5へ1200更新cosine、clip1、EMA.999。raw/EMAを0/6/400/800/1200に保存、全state復元を確認する。自己6更新（全深さ）の再演を完全一致で検証し、新optimizerから本学習する。採用する評価点は最終raw1200に固定する。HPの画像を同入力で固定対照として描くが、目的・head・小規模データを同時に変更する表現pilotであり、単一要因の広域比較ではない。

評価はDQの24既知入力にprobe37を追加した25例。未学習6probe+37と8sceneを転移として明示する。全例depth1/3/6と、24/36/42/48/56/62・scene2/7の部分適用depth3を保存する。主比較はsource/正しい1pxgraph描画/HP d3/HR d3の9原寸ページ、depth比較9ページ、部分適用4ページの全22ページ。正解graphを与えるoracleと実予測を区別する。頂点/edge precision/recall、連結成分/孤立点、正解線の近傍coverage、域外inkを記述的に測る。targetずれを許さないgraph一致だけで画質を判定しない。

CPUで4方向接続・端点・孤立点/空/塗り・確率とopacityの分離・背景と座標の勾配を確認。新しいtarget→logits→renderer変換はHQ同rendererとのatol/rtol各1e−4で照合する。これは過去DQ/JAXとの一致閾値を変更するものではない。入力/教師/初期値/コード/hash/時刻/exit/保存stateを記録し、失敗は消さず別runで修正する。fresh16、Krita、本番、exportは未変更。このpilotだけでは採用しない。
