# DM: 参照範囲と太線教師の切り分け

DLでは一本の許容清書が実際に選ばれても、近接線の灰色帯が残った。
データを確認すると、通常の経路をpeak2pxにする一方、wide種別48/512例だけ16–32pxの教師を維持していた。
薄く太い下描きから清書線を出す方針に合わせて、wideにも同じ2pxスタイルを適用する比較を行う。
太い下描きを細くすれば必ず適切になるという製品判断ではなく、合成スタイルの一貫性を調べる実験。

同時に、現行の単段CNN参照範囲15pxでは、太い線の内部と黒ベタ内部が同じ局所入力になる場合がある。
幅32pxの帯と大きな矩形を使うCPU構成例で、15x15の正規化guide領域とモデル出力の一致を確認する。
帯を2pxへ細くする場合の背景位置と黒ベタ内部では、必要な出力が異なる。
この構成例は情報不足が起こり得ることの確認であり、現在の全欠陥の原因を証明するものではない。

パラメータ数15,809、16ch、3blockはそのまま、各blockのdilationを1/4/16にする。
stem3x3と6畳み込みによるCNN参照範囲は87px。source正規化自体の33px窓とは区別する。
以下を比較する。

| 条件 | CNN範囲 | wide教師 | 学習 |
|---|---:|---|---|
| DK normalized | 15px | 元の16–32px | 完了した18,000更新を再利用 |
| context | 87px | 元の16–32px | 新規18,000更新 |
| thinwide | 87px | peak2px | 新規18,000更新 |

context対DKで参照範囲、thinwide対contextでwide教師を切り分ける。
初期seed1336・学習seed1334・評価seed1335はDKと同一で、全初期パラメータを一致させる。
zero headのため全6深さで初期出力も一致することを要求する。新規2群の合成入力domainは1.0とconfigに明記する。
DKの最初6更新を再実行してmetricとraw/EMAの既定許容差を確認し、contextの全18,000batchの入力hashをDK journalと照合する。
対照の18,000更新をもう一度実行したとは数えない。

DKの512形状、augmentation、batch4、再帰1–6、全段重み.5**level、sqrt(batch)でpeak1e-4→cosine1e-5を維持。
全解像度のsource濃度正規化、source-only、linear Charbonnier+.25経路断面+.05範囲lossも同じ。
globalに一つの許容清書を選ぶ。DLの追加merge候補は今回使わない。
元inkによるnormalizerは全群同じで、thinwideでは細線を消す方が低コストになりやすい可能性もあるため、wide全probeを目視する。
blank/blackの教師は変更しない。初期Adam/EMAも共通。

CPUでwideの細線化・他種別保持・幾何非破壊・初期関数一致・RF87正規化モデルの実更新/再開/export/samplerを確認。
dilationの許容上限だけ4から16へ拡張し、既存のデフォルト動作は維持。ONNXは引き続き未検証構成を拒否。
新規両群の最初6更新を再実行(atol1e-7/rtol1e-5)、0/6/6000/18000を保存し16raw/EMA exportを完全復元比較する。
両群のsource/annotation hashは全batch一致、候補hashはwideを含む場合だけ異なる。

標準4,608行・劣化4,608行、等倍16枚を生成し、512px8sceneの128出力・等倍6枚も確認する。
さらにold/new_probeのparallelとwide全24例を等倍6枚で確認する。出力threshold/contrast変更なし。
従来の数値指標は太線を保つ参照を含むので、今回の細線化の採否を数値誤差だけで決めない。
灰色帯、線の消失、交点、線幅の不自然な変化、黒ベタの破壊を重視する。
512px scene1/7等の意図的なgapは既存の幾何確認に従い区別する。
合成のみ。Krita・配布は変更しない。追加Gaussian2drawはsource-onlyのため独立品質標本ではない。
