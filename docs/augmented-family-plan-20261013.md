# DB: 許容清書の学習を、薄さ・ぼかし・局所変形へ広げる

固定opacity=.4/sigma1.2のloss微調整では、枝の不安定さや粒状感を十分に改善できなかった。
本比較ではモデル容量・lossを共通にし、下描き劣化の幅を変える。単一教師MSEへ戻さない。

## 学習

DA linear raw6000を共通初期値。16ch/15,809params、同じ512合成clean候補集合、fresh Adam/EMA。
各6,000更新、batch4、LR5e-5*sqrt(4)=1e-4→cosine1e-5、depth1..6、段重み.5**level。
seed2026101316。clean形状・mask・段数・Gaussianキーは対応させる。

- fixed: 従来のopacity=.4、sigma1.2、±2pxの二本を重ねた入力。
- augmented: 25%はfixed、75%はopacity uniform[.2,.8]、sigma uniform[.6,2.4]、
  x/y各方向のsin変位amplitude[0,1.5]px・周期[64,160]px・独立位相[0,2pi]。
  画像内で滑らかに揺れる下描き。許容clean群は変形せず、元の形状からの自然な清書を認める。

augmentationはstep/item別の乱数系列から生成し、選択順やGaussian系列を変えない。
劣化したsourceを各段へ縮小した後、従来通りGaussianを加える。
familyは全画像・全段で一候補の最小costを選び、境界接線延長Charbonnier+.25経路断面を共通使用。
geometry lossのsource opacity=.4という旧前提だけ、既知の合成opacityで補正する。
このタグはlossのみが使い、モデル入力・推論へ渡さない。実入力からopacityを推定できるとは主張しない。
fixedの値/gradientはDAと完全一致を検証。rangeguard・推論clip・feedback・保護は同じ。

## 評価

従来4split×64・3mask×2draw×3モデル(初期/fixed/augmented)で4,608行、16等倍シート。
さらに未使用の形状key(new_probe, slot, variant20261013)64例を学習前に固定。
base/faint(.2)/dark(.8)/sharp(.6)/soft(2.4)/rough(.3,1.8,1.5)の6劣化、
whole05/partial05・2draw・3モデルで4,608行。全経路断面と接続、帯外ink、白紙/黒、保護を評価。
旧opacity固定のgeometry評価値はstress集計から除く。劣化ごとのGaussianは対応させる。
評価seed2026101317、rough位相seed2026101318。

stressは各劣化3シートで全kind代表(0/6/24/36,12/18/48/56,30/42/54/58)、
faint/rough部分適用を追加、計20等倍シート。従来と合わせて36枚を実際に確認する。
採否は線の自然さ・入力追従・粒状感・灰色・端点・枝・余分な接続から判断。
幅や位置の許容差は維持し、唯一の教師への画素一致を要求しない。
許容候補は有限のoffset/width群なので、これ以外の自然な清書を完全に評価できるとはしない。

CPUでbase完全一致、opacity操作、blank/clean保持、再現、fixed gradient一致、実更新を検証。
GPU最初6更新は事前atol1e-7/rtol1e-5、全状態復元・12raw/EMA exportは完全一致。
新しい形状も開発用評価で、独立学習seedによる確認ではない。Krita・配布・封印データは変更しない。
