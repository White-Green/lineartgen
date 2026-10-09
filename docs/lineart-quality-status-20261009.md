# 線画復元実験の現状 — 2026-10-09

元の下描きからCNN特徴を取る構成は有望。配布中100kモデルからはまだ置き換えていない。
全1,562開発入力+4白紙、4モデル、反復を含む6,568条件を評価した。
実clean線画をぼかした入力では、強度.5全域でpixel候補のhaloが76.86%減り、
F1が.854720→.944971、濃度不足が.168443→.048584に改善した。
実画像でも輪郭や細部の改善を確認した。ただし履歴の異なる配布モデルとの差は観測比較。

未解決なのは重ね描きの太さ、近接平行線と隙間の分離、反復中の状態を利用しない点。
source-only特徴のN回推論は強度/Nの1回推論と最大1.2e-6差で、反復的に精緻化する仕組みには
なっていない。KritaのAPIを変えずに採用する場合も、この挙動を明示して扱う必要がある。
[広域結果](../../jax-source-feature-broad-20261009/docs/source-feature-broad-results-20261009.md)、
[実線画の比較画像](../../jax-source-feature-broad-20261009/runs/source-feature-broad-20261009/native-review/real.png)。

source特徴にregion.1 / region.1+connection.1を加えた4k比較は完了し不採用。
haloをさらに約28–35%減らすが、濃度・線幅など保持条件21/31項目で失敗した。
最終8枚と白紙端部1枚を確認した。
[構造loss結果](../../jax-signal-structure-20261009/docs/signal-structure-results-20261009.md)。

実行中はsource特徴に局所濃度分布loss3を加える固定4k比較。対照は完了済みpixelを厳密再利用。
起動6更新で別実行とのbitwise差が出たため一度停止した。重み最大1.86e-6、固定8入力の
出力最大4.18e-7の差を測定し、最初の実行の状態を保存して診断を継続している。
再学習の完全一致は主張しない。保存復元と品質の基準は維持。
[計画](signal-transport-plan-20261009.md)、[例外記録](signal-transport-startup-amendment-20261009.md)。

すべて合成下描き。130,605パラメータ、再帰1–6、段loss0、sqrt(batch)LR、縮小後Gaussianを維持。
直近の特徴比較ではGaussian seedの違いが出力に影響しないため、反復drawを独立標本に数えない。
広域診断の事前固定9枚と有意味な部分境界・最悪白紙の補助2枚を目視した。初期部分クロップは
線がないため判断不能だったことを記録。保護画素は全条件で厳密一致。
実BGRA入力・Windows/Krita GUI、新しい独立学習seed、封印確認はまだ未実施。
十分な最小モデルサイズも未確定。封印評価は未開封。

過去の実験: [元画像特徴](source-signal-results-20261009.md)、
[接続query](connection-query-results-20261009.md)、[オンライン図形](procedural-structure-results-20261009.md)、
[旧濃度輸送](local-transport-training-results-20261009.md)、[出力基準](source-anchor-results-20261009.md)。
