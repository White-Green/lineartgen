# EO: 入力の保存精度を学習に含める

ENは近接線を整理したが、既知scene1のfloat入力で改善した連続性が8bit入力では保たれなかった。モデル容量やgeometry lossを変更せず、合成float入力だけでの学習と保存精度の差を検証する。

EI group6 raw18,000からquant群を6,000更新する。完了済みEN merge群を対照として再利用する。初期値、optimizer/EMA reset、512形状、sampling seed2026101502、幅拡張、rough、D4、mask、Gaussianキー、教師候補、lossはENと同一。batch4、基準LR1e-4×sqrt4=peak2e-4、cosine最小2e-5、warmup0、再帰1–6段、全段loss.5^levelを維持する。

roughと幅拡張の後、例ごとに確率.5で8bit画像へ丸めてfloatへ戻す。独立乱数キー[seed,step,item,0x51383030]を使用し、その他のsamplingを変えない。元のfull-resolution入力の保存を模擬するため、8bit化は縮小前に行い、Gaussianノイズは従来通り縮小後に加える。ノイズ画像の量子化や推論後の閾値処理は行わない。

元geometryとmerged geometry、候補許可フラグは元のfloat sourceからENと同様に固定する。各stepで候補payloadのhashがENと一致することを検証する。保護領域はその例の保存後入力を保持する。8bit化しないbatchでは入力全体がENと完全一致することを確認する。全例を常時8bit化せずfloatも残すので、異なる入力精度への適用性を検証できる。

事前確認は丸め誤差上限、範囲・dtype・冪等性、既存512px入力のuint8列との完全一致、全6段で元/merged教師とmask一致、保護領域保持、縮小後Gaussian残差一致、非変更例一致。モデル/loss/D4/幅拡張の検証はENの同一ソースの試験を引き継ぐ。学習時の6step再現・全状態復元・raw/EMA8export照合も行う。3,000更新ずつ2区間で実行し、最終rawだけを比較する。

評価はENと同一の9,516出力、事前固定34ページとscene1/7の追加等倍比較。標準/stress/幅摂動は従来の入力を保持し、512pxはfloat/uint8両方を確認する。初期とEN対照の保存予測一致を確認する。8bitでの途切れ減少、floatの退行、太い単線の改善維持、部分編集を重視し、MSEや画像全体のRMSだけで判定しない。予約済み新規確認と製品は変更しない。
