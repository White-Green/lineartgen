# DF: 固定清書スタイルと出力線画priorの併用

DDのcanonicalで薄線は改善したが、灰色側線・端点・交点・塗りの欠陥が残った。
DEの段情報には明瞭な改善がなかった。DD canonical raw6000から、同じ固定スタイル学習の
controlと、出力画像だけを見る判別器を加えたpriorを各6,000更新する。
推論モデルは同じ16ch、3残差block、15,809パラメータ、段条件なし。
source-onlyなので追加独立Gaussianに依存しない。入力画像そのものの情報は保持する。
確率的生成・OTの多様性を維持できたとは主張しない。

両armとも512合成形状、DDの2px最大幅スタイル候補、同じopacity/blur/warp拡張。
linear Charbonnier + .25 path + .05 range、全段重み.5**level、画像全体で一候補を選ぶ。
元画像の線量で両arm共通正規化。教師一枚へのMSEを成功基準にしない。
batch4、sqrt(batch)によりpeak LR1e-4、cosine1e-5、深さ1–6、fresh Adam/EMA。
train seed2026101327、eval2026101328、critic seed2026101329。

priorはCTの出力のみを入力するscale-aware CNN判別器(15,497パラメータ)。
local native64以下 + global64以下の2view、物理的な縮小段を条件入力する。
realはcanonicalの9位置候補から独立・一様選択し、box縮小する。指定教師幅には戻さない。
各画像/段で8patch: 4構造anchor、2ink、1paper、1active。anchorは各pathの真の端点2個・
弧長中央1個、各path対の65点サンプル間最近点の中点。交点付近を含むが完全な構造検出器ではない。
メタデータは学習時のpatch位置選びのみに使う。モデル入力には渡さない。
保護領域を含むpatchと部分編集画像のglobal viewは重み0。境界の灰色を悪い清書と扱わない。
D logistic + R1 gamma1、Adam1e-4 b1=0 b2=.99、global grad clip1。
各G更新前にDを1回更新、Gはnon-saturating priorを係数.2で追加、最初500stepをlinear ramp。
G勾配中Dは固定、D勾配中Gは固定。推論時Dは不要。

CPUでprior係数0の既存loss/勾配一致、非ゼロ係数でGへの実勾配、D固定、指定teacher不依存、
実G/D更新・完全状態復元後の同一継続、構造位置と保護patch除外を検証する。
GPU全6深さの初期両arm一致、最初6更新の再実行はatol1e-7/rtol1e-5、6/6000の全状態復元、
12raw/EMA exportは完全一致。両armは全6,000入力・候補・D注釈hash一致を要求する。

標準4split×64×3mask×2draw×3モデルと6劣化×64×2mask×2draw×3モデルの9,216行。
既存の事前指定16枚を等倍で確認し、共通54候補のprofile/接続/分離等も記録する。
512px/uint8の追加評価は入力と手順を先に束縛して別記する。
灰色帯の減少・滑らかさ・端点・接続・分離を重視し、線の欠落・架空線・塗りの破壊を許容しない。
GAN lossや平均誤差の低下だけでは採用しない。単一seedの開発実験。
合成のみ、独立確認seedと封印データは未使用。Krita・ONNX配布は変更しない。
