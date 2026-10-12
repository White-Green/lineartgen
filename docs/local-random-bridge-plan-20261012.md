# CP: 固定2時刻のrollout学習を連続時刻の教師補間学習と比較する

CM/CNでは32回推論で線が肥大し、COの2回評価でも画素差分lossの改善は小さかった。
現在の小型CNNのrollout学習はremaining time u=1/.5だけを監督しており、32回推論の
多くの時刻は直接には学習していない。これは入力ノイズの強度tとは別の変数である。
COの32回結果による選択前に、時刻と学習状態の作り方を変える実験を固定する。

一般のFlow Matchingは、時刻付きの確率経路上でベクトル場を監督する。
ここではその考え方を、既存のteacher bridge実装と有界endpoint表現で調べる。
理論の直接再現ではなく、interval損失、uのfloor、画像lossを持つ実装上の比較である。
[Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2#S3) を参照。

CPはCM raw20000の実際のmodel/Adam/EMAから開始し、CO controlと同じ
data index14000–17999の合成16,000教師・source/mask・Gaussianで4,000更新する。
15,809パラメータ、batch4、depth3/4/5/6/1/2からの循環、LR1e-5、最細段のlossから
全再帰への勾配を保つ。LRはbase5e-6×sqrt(batch4)、minimum1e-5、warmup0。
教師seed2026101111、Gaussian/時刻seed2026101093。実下描きペアは使わない。

CO controlは再学習せず、完了した同一開始状態・同一4,000更新の結果を対照として使う。
CP実行前にCO controlの最初の6実更新を再計算して全状態一致を確認する。
全4,000batchで、教師hash・source/maskを含む元のdirect入力hash・depthをCOと照合する。
この比較は更新回数とデータを揃えたもので、wall-clock計算量を揃えたものではない。

変更する学習方式:

- 従来はu=1からの自分の予測を使ってu=.5の入力を作り、両時刻を教師へ復元する。
- CPは既存乱数で25%をtau=0、それ以外をtau=sin²(pi*U/2)とし、u=1-tauを使う。
  各段でx=u*z+(1-u)*y、z=縮小source+.25*縮小strength*Gaussian、y=縮小教師。
  保護画素はsourceに固定する。教師を補間に使うのは学習時だけ。
- 元のendpoint headでraw Dを予測し、interval残差/max(u,1/32)の二乗を主lossとする。
  同じAD image lossをraw Dに実係数.1で加える。COのedge_energy追加項は使わない。
  変更は時刻・教師経路・lossの評価位置を合わせた学習方式であり、時刻だけの効果とは呼ばない。

このteacher bridge系は以前のU-Net実験でも試しており、新規アルゴリズムという位置づけ
ではない。小型CNNの広い合成学習では初の比較。従来の2回rollout lossは入力から教師への
直接復元を優先するため、曖昧な入力の平均的出力を促す可能性もあるが、その原因をこの
1実験だけで確定しない。ノイズ共有や最適輸送マッチングを同時に変更しない。

保存20000/20006/22000/24000、8raw/EMA exports、45教師batchの独立hash、
初期全状態・実6更新・保存復元・exportを検証する。最終raw24000のみ評価し、EMAや
中間checkpointで選別しない。初期CMの76パネルとE/R16出力も完全再現する。

2回推論（主）/32回（副）各1,566条件でE/R/CO control/CPを比較する。
各6,264行、計12,528行。CO controlとCPに既存229基準ずつを適用し、緩和しない。
各回数E/R/CO controlの4,698行と114パネルを完全再現する。計12枚のraw比較画像を
ネイティブ解像度で確認し、薄さ・粒状感・線幅・切れ・誤接続を同時に見る。
alpha<.1の表示除去はしない。finiteと保護画素完全一致を必要とする。
配布モデル・Krita設定は変更せず、独立学習seed・封印確認はまだ行わない。
