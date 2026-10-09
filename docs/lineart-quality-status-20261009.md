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

source特徴の局所濃度分布loss3は4,000更新を完了し不採用。haloは約65–77%減ったが、
PROBE blur.5の濃度不足率が.179→.375へ増加し、保持条件38項目失敗。
最終8枚と白紙最悪端部を確認した。起動時の非bitwise差は記録を維持し、
保存復元とraw/EMA書き出しの完全一致は確認済み。[結果](../../jax-signal-transport-20261009/docs/signal-transport-results-20261009.md)。

元画像特徴向けの明示的な1回復元モードを別ブランチで実装し、学習済み候補で
実BGRA 84条件＋seed変更28条件を確認。JAXとの最大差は1階調で、保護画素は一致した。
新しいONNX metadata形式2が必要で、配布パッケージはまだ切り替えていない。
[実BGRA・1回復元の検証](../../jax-one-pass-restoration-20261009/docs/one-pass-restoration-results-20261009.md)。

実clean線画＋オンライン図形でのblur対mixed追加学習4,000更新も完了し、両方とも不採用。
mixedは対照よりstrokes haloを10.24%減らしたが、blurと細線を悪化させ、
広域121チェック中40項目失敗。最終14枚を目視、4,698広域条件を評価した。
[結果](source-mixture-training-results-20261009.md)。

実行中は、事前固定のmixed raw4kからの元画像条件patch識別器あり/なし比較。
識別器のみ250更新のあと、両生成器を2,000更新。推論モデルのサイズは増やさない。
[計画](../../jax-source-patch-prior-20261009/docs/source-patch-prior-plan-20261009.md)。
[blurと透明度の識別不能例](blur-opacity-ambiguity-20261009.md)を補助診断として記録した。

すべて合成下描き。130,605パラメータ、再帰1–6、段loss0、sqrt(batch)LR、縮小後Gaussianを維持。
直近の特徴比較ではGaussian seedの違いが出力に影響しないため、反復drawを独立標本に数えない。
広域診断の事前固定9枚と有意味な部分境界・最悪白紙の補助2枚を目視した。初期部分クロップは
線がないため判断不能だったことを記録。保護画素は全条件で厳密一致。
Windows/Krita GUI、新しい独立学習seed、封印確認はまだ未実施。
十分な最小モデルサイズも未確定。封印評価は未開封。

過去の実験: [元画像特徴](source-signal-results-20261009.md)、
[接続query](connection-query-results-20261009.md)、[オンライン図形](procedural-structure-results-20261009.md)、
[旧濃度輸送](local-transport-training-results-20261009.md)、[出力基準](source-anchor-results-20261009.md)。
