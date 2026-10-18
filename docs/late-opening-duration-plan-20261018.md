# HM: 後半openingの1200→6000継続比較

HL1200では、22508パラメータの縮小枝モデルの中心線・塗り・短い尾を保ち、openingが80/87.5pxの余分な外側線を減らした。ただし36/37/scene2の灰色、scene7の弱い接続、太い短線の直交片は残った。短期で部分改善があったため、別計画として継続時間を比較する。

HLの対照/opening両方の最終raw1200、AdamW moments/step1200とEMAを引き継ぎ、各6000まで4800新規更新する。optimizerをリセットせず、HLのconfig.steps/decay_steps=1200とconstant LR2e−5（base1e−5×sqrt4）をそのまま使う。実行上の最終stepだけ6000にする。B4/acc1、clip1、EMA.999、深さ1〜6cycle、全段loss .5^level、dropout.5、distance/geometry.1/sign.25/globalL4、retained教師、最終段opening0/.1は不変。モデル定義・推論後処理・データ分布を変えない。

HH/HI/HJのstep1201〜6000を同順に利用する。新規4800の内訳legacy3000/GL1200/GV600、既存1200と合わせ3750/1500/750。source/candidate/mask/axis/dropoutを過去HHのpayloadと完全照合する。学習journalはHLの1200行を継承し、新規更新と区別する。1200/1206/1236/2000/3000/4000/5000/6000の全状態とraw/EMA合計32exportを監査（うち4exportは継承1200の再保存）。両条件の新しい先頭36更新を独立再演しmetric/全state完全一致を確認する。

CPUは継承scheduleの終点以降も一定であることと、実際の1200prefix/残り4800順序を検査し、既存opening/retained目的のテストも行う。GPUでは元HLコードからのfull-state復元との完全一致、全6段×3家族18batchの初期出力一致と有限非零勾配、直接HLを継続した6更新/全state完全一致、最後のbatch payloadを確認する。

評価は最終raw6000固定。中間checkpoint/EMAを品質で選ばない。参照initial=HL opening1200、対照winner=HM control6000、consistent=HM opening6000。9854出力・原寸76ページ（通常、劣化、512scene、太さ、FU/GK/GO、8方向、HC/HF部分/移動）をすべて確認する。教師3ページと1114例のactive-ink監査はhash付き再利用で、新しい目視には数えない。80/87.5pxの外側線だけでなく中心保持、36/37/scene2の灰色、scene7接続、60/62直交片、塗り、短い尾を確認する。MSEを採用基準にしない。

別batch/コンパイル/保存参照の比較はHL同様、実行前のatol/rtol各1e−5。geometry anchorも同条件で、非完全一致数と最大差を保存する。入力・保護画素・保存state・自己36更新は完全一致。失敗や中断を隠さず、未保存更新が出た場合は復旧前に記録する。記述的幾何測定を実行前に固定し、実出力と併せて判定する。fresh16、本番重み、Kritaには手を加えない。
