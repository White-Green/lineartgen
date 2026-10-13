# DA: 有効範囲の境界勾配を保つfamily loss

CZ intervalは欠損率を下げたが、灰色の縁を増やした。rawが範囲外へ深く入ると、
ほぼ白/黒のAA候補にも最大に近い勾配がかかる。次の一条件をCZと対応比較する。

physical opacity a、候補y、b=clip(a,0,1)、r=b-y、q=sqrt(r²+.05²)に対し
penalty=q-.05+(a-b)*r/q。有効範囲内はCharbonnierと同じ、範囲外は境界の接線で延長する。
正しい白/黒は0、重大な誤飽和には内向き勾配、小さなAA誤差には小さな勾配。
値と一階微分は境界で連続。真のスカラーlossであり、clipの代理勾配ではない。
画像/段/断面で集計後、一つの許容候補を選択。断面.25・geometry・rangeguardは同じ。

CZと同じCY profile raw6000初期値・16ch/15,809params・512例・batch4・6,000更新。
同じseed2026101314、初期Adam/EMA、LR1e-4→1e-5、depth1..6、段重み.5**level。
入力を縮小してからノイズ。推論clip・再帰feedback・Krita I/Oは変更しない。
新しいlinear一条件のみ学習。CZ clippedを対照として再利用し、CZ intervalも文脈比較する。
初期重み/config・データ・sampling plan・各6,000入力hashの一致を検証する。
CZの結果を見た後に定めた追試で、独立確認とは呼ばない。

評価もCZと同じseed2026101315・4split×64・3mask×2draw、4,608行・16等倍シート。
対照と初期値の計3,072出力はCZ保存結果と完全一致を要求する。灰色/粒状感、枝、端点、
近接線、白紙/黒、部分適用を確認し、断面欠損率だけでは採用しない。
GPU最初6更新は事前atol1e-7/rtol1e-5、全状態復元と6raw/EMA exportは完全一致。
CPUは境界の値/微分・AA勾配・正しい白黒・保護・実更新を検証。
固定opacity・有限候補の合成診断であり、実際の下描きへの汎化は未確認。
