# HT: 中心線に投影する近傍anchorの表現監査

HSは余分な頂点を減らすと転移の線を欠損した。現在のcell内1頂点から、線からR内のcell中心にanchorを持ち、そのanchorを連続中心線の最短点へ投影する表現を試す。近傍の複数anchorは同じ線に重なり、不透明描画のため数が増えても濃度を足さない。CNN変更や学習をする前に、正解パスからの表現だけを監査する。

各段R=max(.75,4/2^level)。9channelは存在、xy offset、log幅、4方向の辺、塗りを維持。位置はcell中心+R*tanh(raw_xy)でcell外を許す。近傍の最短segmentをCPUで厳密投影し、同距離は入力順の早いsegmentを優先。幅とarc長はsegment内補間。同pathでarc差が2 stage px以下の8近傍だけ辺を立てる。異なるpathや近接した折り返しを直接つながない。端の近くは端点へ投影、孤立点は円。不透明可変幅capsule unionを4倍解像度で描画し平均する。存在logit閾値0、幅exp(clip[-8,ln8])、塗りはhard cell。幅の上限、塗りのpixel境界制限はHRのまま。

HQの40既知入力（24例+16解析fixture）だけを再利用する。32px halo、全6段のラベルを作りfineだけ描画。source/original opaque/旧cell1px/projected oracleの4列、全12原寸ページを読む。40 NPZ/240段targetsと画像hashを記録する。基準画像はHQのCPU保存値をそのまま再利用し、DQとの過去許容値失敗は上書きしない。coverage差は表現の歪みの診断であり学習目標ではない。

テストは線幅がanchorの数で増えないこと、0.25px近接2pathを誤接続しないこと、短線端/意図的gap、全段empty/fill/border、3本のsubpixel平行線を収容できない残存限界を確認する。交差では別pathのgraph component数と見た目の接続を区別する。教師の存在/座標/幅/辺を与えた結果であり、学習済み性能とは扱わない。滑らかさ・隙間・細線を壊す場合は学習前に問題を記録する。

新規学習0。fresh16、本番、Krita、export変更なし。表現が使えそうな場合に限り別計画で同パラメータの学習比較を行う。
