# CX: 候補選択と誤差の形を分ける2×2比較

CWの出力を見る前に追加条件を固定する。CWのdynamic/stableは候補とのopacity二乗誤差。
CXはそれぞれをCharbonnier sqrt(d²+.05²)-.05に置き換える。それ以外は全てCWと一致させる。
小さなgray残差に対する勾配を増やす効果と、候補選択を固定する効果を分けて確認する。
Charbonnierは誤差1で約.951、二乗誤差は1。係数は共通1で、正規化は全候補共通のink量。
これは同等gradient scaleの比較ではなく、誤差関数を置き換えた実際の効果の比較。

## 条件

CW数値commit ee1bd06の初期値・512訓練例・sampling・固定候補をそのままコピーしhashを記録。
16ch/15,809params、CV expanded raw8000初期値、fresh Adam/EMA、各6,000更新、batch4。
seed2026101310、LR base5e-5*sqrt(4)=1e-4→cosine1e-5、深さ1..6、各段0.5**level。
画像を縮小してからノイズ。入力/ノイズ/段数/順序/評価noise2026101311もCWと同一。
CT geometry/rangeguardは共通。変更はfamily costのpixel penaltyだけ。

- dynamic Charb: 27候補×全画像×全段のCharbonnier costを集計し、最小候補を選ぶ。
- stable Charb: CWの共通初期モデルが追加Gaussianなし深さ3で選んだ候補IDを固定して使用。

固定候補をCharbonnier基準で選び直さない。候補選択方針を変えると交絡するため。
推論は既存I/Oのまま。入力ごとの27候補は学習のみで使用する。

## 評価

CWと同じ4split×64例、3masks、2noise、最終raw6000＋初期値で4,608行と16等倍シート。
CWと同じノイズ・評価例・初期値をhash/出力で確認し、4条件を比較する。
許容断面、細い枝の生存、近接線の分離、線端/濃度むら、紙面・黒、保護域で判断する。
平均値だけの改善、線幅増加による接続率改善、teacher MSE低下だけで成功としない。
固定opacity.4、有限許容候補の探索に限る。corner数値probeの偏りはCW同様画像で補う。
追加のcheckpoint選択・評価後の係数調整はこの比較には入れない。

CWと同じ6更新GPU再現(atol1e-7/rtol1e-5)、保存復元・12exports完全一致、候補集合/保護域と
実更新のCPUテストを実行。Krita・配布・独立確認seed・封印データは変更しない。
CW学習完了後にGPUで実行し、4条件全てが失敗ならその結果を保存して次の原因切り分けへ進む。
