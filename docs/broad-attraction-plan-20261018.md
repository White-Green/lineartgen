# HY: 合成形状を増やした投影graph学習（実行前計画）

HXのR1は近接線・交差部を改善したが、10学習形状からの転移で曲線37と太線59/62/63を消した。学習形状の不足を検証するため、既存512合成例を全て使い、R4/R1を同じ初期値から比較する。HX学習済み重みは初期化に使わない。

両22725パラメータ、HR学習前初期値、AdamW/EMAを新規開始、6000更新、B4、再帰1〜6各1000、全段lossの重み0.5^level。base LR1e-4をsqrt(4)倍して2e-4、cosine最終2e-5。count BCEと位置・幅・塗り、係数はHXのまま。最細投影R4/R1、段ごとmax(.75,R/2^level)。D4変換をsource/noise/guide/mask/全教師に揃える。描画は両方投影先距離2px+1e-5制約、不透明4倍描画。接続loss候補・同変headは今回入れない。

データは既存graphキャッシュの512例（64 slot×8 variants）。straight/thin/bezier/corner/junction/crossing/parallel/gap/taper/wide各48、blank/black各16。元の線幅は通常0.8〜5px、wide16〜32px、教師peak幅2px。24〜88px追加データや曖昧な軸の新教師は含めず、この比較で解決したとは主張しない。幾何warpは行わず、追加blurσ0〜1、opacity.65〜1、部分適用25%、D4各itemをseed2026101811/stepから生成。両条件に同じprepared入力を渡す。Gaussianノイズを参照しない既存CNNであり、確率的生成の比較ではない。

R4/R1教師は512×6段を事前作成済み。最細〜level4ではpath欠落0。level5のcrossing251/270にpath欠落がある表現限界は保全し、粗い段の低い重みでそのまま使用する。各kind先頭例のsource/R4/R1教師12原寸ページを開始前確認。既知25評価例とのsource bytes重複はblank/black各16記録のみ（54/56）。これは汎化の証拠に数えない。他の23例は入力bytes一致なしだが、繰り返し使用した既知評価であり独立最終評価ではない。512は記録数であり512種類の固有画像ではない。

6000後に既知25例×深さ1/3/6、8部分適用、計166条件、31原寸ページを確認。HX R1小試験を比較列として使う。採用基準は線の連続性・近接線の分離・端・幅・隙間・交差・塗り・部分適用保持。MSEやloss低下だけで採用しない。R1の欠損が減っても帯が戻れば成功ではない。

自己6更新×2条件のmetrics/fullstate完全再演、跨プロセス先頭6更新は事前atol/rtol各1e-5。保存0/6/1200/3000/6000でfullstateとraw/EMA復元一致。過去strict失敗は残す。合成のみ、fresh16・本番・Krita/exportは未変更。
