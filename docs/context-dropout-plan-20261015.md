# ET: 同じ入力・目的関数で再帰条件dropoutを比較する

ER/ESでEQ mixedのwide61/62の余分な線が粗い条件により増幅されると確認した。条件の位置情報保持や倍率変更だけでは改善しなかった。学習中に粗い条件を省く経験を加え、誤った粗い形状に引きずられにくくできるか調べる。曲線37やscene7の問題にはこれだけで十分とは考えない。

EQ mixed raw18k（16,402パラメータ）を同じ初期値としてcontrol/dropoutを各6,000更新する。B4、base LR1e-4×sqrt(4)=peak2e-4→min2e-5、warmup0、clip1、EMA.999。optimizer/EMAを両方同条件で初期化。3,000でstate保存し、スケジュール/optimizerを維持して6,000へ継続する。0/6/3000/6000のraw/EMAを保存（16export）。

入力と教師はEQ mixedの続きのsampling step18000–23999（seed2026101440）。512合成geometry、独立opacityの重ね合わせ、幅4–32、D4、粗化、strength maskを同一とする。再帰深度1–6・全段loss・.5^level重み・family選択を維持。dropoutのみ各画像・各粗→細接続の条件を確率.5で零にする。独立乱数[2026101505, local_step,0x435458]、空間全体共通、残す値の倍率変更なし。粗い段の出力とそのlossは常に計算し、切った接続だけを通る勾配を止める。推論は両モデルとも通常の全接続。学習後の閾値/平均処理なし。

CPUテストでall-keepの既存forward一致、切った接続から細段への勾配0と粗段自身の勾配維持、mask再現を確認。実モデルGPU preflightで深度1–6のall-keep control gradientを既存objectiveと比較（既定ROCm envelope atol5e-6/rtol1e-4、差とexactを記録）、depth1両arm一致、両arm入力/教師一致を確認。6更新replay、checkpoint復元と16exportの一致、有限値、source/code hashを検査。

評価はEQと同じ既知入力、深度3のstandard/stressと深度6の512scene、width sweepで9,516出力。参照はEQ mixed raw18k。固定34等倍ページを全て見る。wide副線/点の減少とscene1融合・分岐の維持が主仮説。curve37とscene7の残存、幅の中間域/太線・blank/fill・partial保護の退行も見る。MSEで採否を決めず、単一seedの診断とする。予約fresh16例や製品は変更しない。
