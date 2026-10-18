# HU: 中心線投影を学習する固定pilot

HTは既知の中心線からなら近傍anchorを重ねても不透明な細線を保った。次に入力からの予測を検証する。HTのnearest-point教師、cell中心+R*tanh offset、幅/辺/塗りのdecoderをそのまま使う。同じ22725パラメータのGraphCNNで、存在/辺だけbalanced BCE（正負別平均）とcount BCE（合計をNpositive+.001*Ncells+1e-6で除算）の2条件を各1200更新する。塗りはbalanced、座標はstage pixel単位offsetのHuber、log幅もHuber、全5項係数1。粗い段R=max(.75,4/2^level)。閾値/幅clip/後処理を結果で選ばない。

共通初期値はHRの**学習前**initial.npz（HP trunk+zero graph head）。HR最終やHSで適合済みの重みではない。fresh AdamW/EMA、B4、base1e-4×sqrt(B)=2e-4→2e-5 cosine、深さ1〜6各200、全段.5^level正規化平均を維持する。10形状、opacity/blur、whole/partialと全1200入力系列をHRと揃える。2条件間の違いは存在/辺BCE正規化だけ。旧cell方式HRに対しては位置の表現・ラベル密度・位置回帰スケール・再帰latentの意味が変わるので、単一loss係数の比較とは呼ばない。

prepared入力のbyte比較のためgraph_oracleはHRの元値を保つ。これはprepared.targetsの記録用でありforwardは参照しない。別引数のtarget0..5を投影教師に置き換える。preflightの非定数重み・depth2でprepared.targetsを全-999へ変更しても全headが完全一致することを確認する。入力auditのtarget_sha256だけは新教師のhashになり、元hashも保存。その他の1200auditはHRと完全一致を要求する。decoderのoracleは投影方式を別保存する。

CPUは幾何5テストとloss3テスト（全段の正解位置loss0、中心線へ向く勾配、背景幾何の無勾配、empty finite）を実行。GPUは各条件6更新の複製自己再演、全state/metricsの事前atol/rtol各1e-5とbit一致を別記録。0/6/400/800/1200のfullstateとraw/EMA20個を保存・完全復元。別プロセス先頭6更新は同許容値で比較する。HRの過去strict失敗は変更しない。

評価は最終raw1200のみ、既知25例×depth1/3/6+部分8=83条件×2=166。主比較9ページsource/HRcell1200/HUbalanced/HUcount、各モデルdepth比較9ずつ、部分4で計31原寸ページを全て読む。HR画像は保存hashで再利用。連続性・滑らかさ・幅・余分な塊/線・細い平行線・gap・taper・太い入力・塗り・部分境界を判定。存在precision/recallやlossが良くても線切れは不採用。教師を使ったHTの結果を学習性能と混同しない。

学習10形状・既知転移15で、広域汎化とは扱わない。HTのsubpixel gap端の後退、3本の極近接線を落とす可能性、交差graph成分と見た目の接続の違いは残る。fresh16/本番/Krita/export変更なし。結果を見てから広域学習の計画を別途固定する。
