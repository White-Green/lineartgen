# CZ: 誤った飽和に候補lossの修正勾配を流す

CYでは許容候補に対する欠損候補画素の67～72%が白側に飽和し、clipを通るfamily costの
直接勾配が遮られた。枝32画素中27、融合部の余分な黒133画素中120にも確認。
rangeguardは別途作用するので全lossが0とは言わない。この診断だけで因果は確定しない。
以前の[単一教師interval L1](source-interval-l1-plan-20261010.md)とは異なり、有限許容候補の
現在のlossで同じ機構を検証する。clipを外す推論やSTEは使わない。

## 固定条件

CY profile raw6000を共通初期値にし、16ch/15,809params、同じ512合成例、fresh Adam/EMA。
各6,000更新、batch4、LR base5e-5*sqrt(4)=1e-4→cosine1e-5、depth1..6・各段0.5**level。
縮小してからGaussian、whole .1/.5・partial .5、両条件同じ入力/ノイズ(seed2026101314)。

- clipped: CY profileと同じCharbonnier family＋.25経路断面＋geometry/rangeguard。
- interval: family pixel costのみ、物理出力から許容clipping区間までのCharbonnier距離にする。

物理opacity a=(1-z)/2、候補opacity y=(1-clean)/2に対し、y=0なら残差relu(a)、
y=1ならrelu(1-a)、内部AAならa-y。sqrt(residual²+.05²)-.05を使う。
[-1,1]内で従来pixel costと同じ。正しい白/黒の範囲外余裕は追加で罰せず、
誤った側に飽和した画素には実際の導関数で戻す力が届く。
分母・mask・候補集合・係数は変更しない。経路断面/geometryはclip後のまま。
全段/画像/断面で合計したcostから一つの候補を選ぶ。推論・再帰feedbackのclipは維持。

## 評価

CYと同じ4split×64例、3masks×2noise(seed2026101315)、初期値＋最終raw6000の4,608行。
旧指標と全経路full_指標、16等倍シートで、枝/近接線/細線/角/端点/紙面/黒/保護を確認。
特にold_train24、old_probe36、new_probe36を確認。平均距離の改善だけでは成功としない。
灰色の再発、線の太り、余分な接続の増加にも注意する。

CPUで飽和両側の値と勾配、正しい白/黒/AAのゼロ勾配、範囲内の値の一致、保護、
対照gradientの一致・実更新を確認。最初6更新GPU再現atol1e-7/rtol1e-5、保存復元・12export完全一致。
固定opacity・有限候補の合成診断。Krita・配布・独立確認seed・封印データは変更しない。
