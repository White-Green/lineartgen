# DD: 入力濃度と清書の線幅を切り離す

DC source-onlyは粒状感を減らしたが、薄い／ぼけた入力に灰色が残った。
可変opacityにもかかわらず教師幅を元の.8–1.2倍に限ると、入力から曖昧な幅を当てさせている可能性がある。
線の経路は維持し、明瞭な一定の線幅で清書する条件を比較する。仮説であり原因と確定していない。

DC source_only raw6000の同一16ch/15,809paramsから、original/canonicalを各6,000更新。
同じ512合成下描き、opacity/blur/warp拡張、Adam/EMAリセット、batch4、seed2026101323。
LR5e-5*sqrt(4)=1e-4→cosine1e-5、depth1..6、全段.5**level、noiseはsource-onlyで消費しない。

- original: 従来の27候補（9offset×元幅の.8/1/1.2倍）。
- canonical: 各画像の最大筆幅を2pxとし、経路・相対幅・taper形状を保つ。
  同じ9offset、widthスロット3つは同じ2pxスタイルを複製し、テンソル形状を共通にする。
  blank/black/wideは塗りの対照として元の候補を維持。

両armから入力線量・ぼかしのgeometry補助項を外す。幅を入力線量で拘束し直すことを避けるため。
これは両arm共通であり、この比較単独からgeometry除去の因果効果を主張しない。
許容候補の経路とoffset自体が入力への追従を与える。全画像・全段で一候補を選ぶ。
lossはlinear Charbonnier+.25path+.05range。pixel/pathの線量正規化には両armとも元の候補を使い、
新しい線幅によってlossのスケールが勝手に変わらないようにする。
モデル入力・normalizer・mask・depth・乱数を同一にし、異なるのは許容出力だけ。

CPUで同一normalizer時の旧cost完全一致、非ゼロgradient/更新、スタイルの幅不変性、経路/taper保持、
塗り対照維持、拡張評価族で許容線が0誤差・有効接続が切れないことを確認。
GPU最初6更新replayはatol1e-7/rtol1e-5、初期全6段一致、全状態restore・12raw/EMA export完全一致。

評価は従来4split256例とDB stress64例6劣化。seed2026101324、3モデル（初期/original/canonical）で計9,216行。
元のfamily指標を残し、元＋2pxスタイルの54候補を全モデル共通の評価族として追加する。
拡張指標はstyle_接頭辞で区別。接続のoffset参照は候補ごとに正しく対応させる。
事後に評価範囲を広げて成功を定義し直すのではなく、学習前に両評価を固定する。
全16等倍シートはDCと同じ代表（標準8、stress8）を確認。表示の入力列は実source配列を使う。
教師MSEや厳密な元幅一致を採用基準とせず、経路追従、灰色帯、連続性、端点、余分な線、塗り、保護を判断。
固定スタイル自体が有用な清書になるかを調べるもので、元の筆圧を完全復元する主張ではない。

新旧familyとも有限の診断用教師。合成のみ、独立確認seed／封印データ／Krita／配布は変更しない。
追加のnative512px評価は別途宣言・保存できるが、主比較の設定や採否条件は結果を見て変更しない。
