# DK: 縮小前の入力濃度正規化

目的は薄い下描きからも明瞭な清書を出すこと。DJの端点局所lossは採用せず、DIと同じ基礎lossを使う。
DHは各段の第1特徴だけ正規化し、第2特徴・残差anchorは入力濃度に依存したままだった。
今回は全解像度sourceを一度正規化し、そのbox縮小を両特徴と編集領域の残差anchorに使う。
これはDHとの差を単一因子で分解する比較ではない。

`ink=clip((1-source)/2,0,1)`、33x33の局所最大値をpeakとし、
`guide=1-.8*ink/max(peak,.02)` とする。局所peak alphaを.4にそろえる。
各段で独立に正規化しないため、縮小された細線の被覆率は保持する。
強度0では元のraw画像をそのまま返す。入力の形状情報は使い、追加Gaussianだけを無視する既存source-only方針。
一様な濃度倍率に対してfloor以上なら不変だが、局所的に異なる濃度や入力ノイズ増幅には保証がない。
交点の途切れを解決するという前提ではない。

新しい初期値からnormalizedを18,000更新する。raw対照はDI plain raw18000を再利用する。
seed1334、初期値seed1336、評価seed1335、全512形状・18,000batch・候補・劣化・optimizer・LRをDIと一致させる。
新しいraw学習を行ったとは数えない。開始時にrawの最初6更新を再実行し、元のmetricとraw/EMA重みに対して
既定atol1e-7/rtol1e-5を確認する。normalizedの全18,000入力hashも元journalと一致を要求する。
rawの完了記録・journal・初期/最終重み・snapshot監査をhashで拘束する。

16ch・3残差block・15,809パラメータ。両者で初期パラメータは完全一致するが、初期関数は意図的に異なる。
normalizedは正規化source、rawは元sourceを返すことを全6深さで検証する。
linear Charbonnier+.25経路断面+.05range、元候補のink量による共通正規化を維持。
2px許容清書から全画像/全段で一候補を選ぶ。exact teacher MSEは最適化しない。
再帰1–6、各段.5**level、batch4、sqrt(batch)でpeak1e-4→cos1e-5も維持。

CPUで濃度不変性・縮小順序・NumPy/JAX一致・強度0保護・実更新・全状態再開・既存raw経路を検証。
normalizedも最初6更新を再実行し既定許容差で確認。0/6/6000/18000を保存、8 raw/EMA exportを復元状態と完全比較。
raw対照の最終予測をDIの保存結果とも照合する。

標準4split・64例・3mask・2draw・3モデル（raw、normalized、参考DI dilated）=4,608行。
劣化6条件・64例・2mask・2draw・3モデル=4,608行。
同じ512px8scene、全体/部分、2draw、float/uint8、2モデル=128出力も確認。
標準8枚・劣化8枚・512px6枚を等倍で確認し、灰色側線、途切れ、端点、黒ベタのリングを重視する。
追加Gaussian2drawはsource-onlyのため独立な品質標本とは数えない。
NumPy実装の整合性まで確認するが、ONNX exportは明示拒否し製品へ混入させない。
合成のみ。Krita・配布は変更しない。
