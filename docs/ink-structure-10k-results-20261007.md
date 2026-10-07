# Ink-structure loss: 10k continuation results

2026-10-07、control / neighbor_guard / ink_guardの10k fine-tuningと評価を完了した。
**にじみは減ったが、線の保持・分離の固定採用条件には両候補とも届かなかった。**
この2候補を採用せず、別seed・新しい合成確認入力への進行も保留する。
閾値を緩めたり、未達の条件を除いて採用扱いにはしない。

## 比較条件と監査

共通100k EMAからのfine-tuning。最初の2k screenからmodel / optimizer / EMA全状態を正確に継続。
train seed 20261007、batch16、sqrt(batch)適用済みpeak LR4e-5、元の10k cosine clock。
126,765 parameters、再帰1–6、各段loss規則、縮小後noise、Krita入出力は同一。
学習sourceの固定commitは `06562d5c590c34032bf9dc9827f75df451eebb3e`。
実行root: `runs/ink-structure-20261007-continuation`。

10kで全optimizer counter、状態の有限性、EMA/raw exportとcheckpointの全配列一致を監査済み。
固定dev 1,562条件 × 3モデルをEMA/rawそれぞれ評価。
55固定パネルの1/2/5回推論、40白紙条件の1/2/5回推論、24補助geometryも評価した。
予測・定義可能な指標は有限、保護領域は完全一致。これらは画質合格を意味しない。
clean実線画への合成劣化を含むが、人が描いた実下描きでの検証ではない。

## EMA主評価

| 方法 | real strong strokes halo改善 | geometry strong strokes halo改善 | 最終基準未達項目 |
| --- | ---: | ---: | ---: |
| neighbor_guard | 16.68% | 17.57% | 42 |
| ink_guard | 10.33% | 11.44% | 20 |

改善率は同時点control比。未達項目数には同じケースを含む条件層の重複があり、独立失敗数ではない。
全条件を混ぜた平均で保持の悪化を隠さず、固定した各domain・劣化・strength・partial/whole条件を判定した。

neighbor_guard:

- real strong blur / toneのink deficitが+1.306 / +1.361ポイント。
- geometry strong toneはink deficit +2.274ポイント、cut +4.375ポイント、active F1 −1.475ポイント。
- geometry strong blurのprofile missing +1.241ポイント。
- 一部のmild条件では誤接続・切断も増えた。

ink_guard:

- geometry strong blurのprofile missing +0.896ポイント、strong toneのink deficit +0.604ポイント。
- geometry mild blurのbridge +4.167ポイント、mild / strong strokesのbridge +2.083ポイント。
- real mild blur / toneのactive F1は−2.158 / −2.190ポイント。
  特にpartialの低下は、編集領域に教師inkがないケースでF1が1→0になる影響が大きい。
  mild blur case447が該当し、平均的な線の4%消失と解釈してはいけない。
  白紙への偽陽性という実際の別問題であり、固定判定から除外はしない。

source単位cluster bootstrapのhalo絶対差95%区間:

| 方法 | real strong strokes | geometry strong strokes |
| --- | --- | --- |
| neighbor_guard | −.02507 [−.02942, −.02085] | −.03688 [−.04143, −.03213] |
| ink_guard | −.01553 [−.01895, −.01233] | −.02401 [−.02716, −.02068] |

realはhalo定義可能46条件・12source、geometryは128条件・32kind/variant。
学習seed間の不確実性や新しい実下描きへの汎化の区間ではない。
rawのhalo改善はneighbor 16.55 / 17.45%、ink 10.33 / 11.39%。rawでも40 / 16項目未達。

## 保存画像の確認

EMAのgeometry24ケース、real details479/480/640を確認。
rawはgeometry16ケースとreal480。0/640/1295/1391の1回対5回推論、白紙partial case7の5回推論も確認。
`review.json` に実際に見た画像のSHAを保存した。

strong blurでの直線・細線・曲線・分岐の消失、strong strokesでの灰色の縁、平行線の結合、
意図したgapの閉鎖は残る。5回反復でも誤結合は修復されず、さらに黒くなる場合がある。
太い線の中心は復元するが、保持lossを足しただけで線の滑らかさや分離が解決したとは言えない。

目視した白紙strength0.5はほぼ白い。一方、全白紙条件ではstrength0.7・5回推論の偽inkが残る。
EMAで最大平均opacityはcontrol .00903、neighbor .00676、ink .00820。
各モデルの最大ink pixel数は6,189 / 4,266 / 5,301。白紙完全保持ではない。

## 次の仮説: 出力のhard clipで失われる勾配

保持lossの係数をさらに増やす前に、出力段の切り詰めを調べた。
`models/output.py` は `clip(raw, -1, 1)` を返す。範囲外のrawに対する勾配はゼロ。
したがって、誤って白に飽和した画素では、出力画像上で正しい方向のloss勾配があっても、
その画素の出力headを通る勾配は途絶える。他画素・再帰経路を通じた共有パラメータ更新は別に存在する。

固定55パネル + 事後に選んだ失敗8ケースで、initialと10kの3モデルのclip直前値を測った。
全252推論で通常出力と完全一致し、3再帰段を含む756レコードを保存した。
この選択集合は全devを代表する推定ではなく、故障原因を調べるための診断である。

- 強いblurの直線807、細線903、曲線999、角1095、分岐1191では、controlの教師線画素の
  約93〜100%が白側のclip範囲外。両保持lossでもほぼ同じ状態が残った。
- 調べたgeometry strong blur 15ケースでは、ink不足量の約52〜53%が白飽和画素にある。
- initialにも同じ現象があり、今回の追加lossだけが発生源ではない。
- geometry strong toneの不足に占める白飽和の割合はcontrol約4%、両候補約7〜8%。
  clipだけですべての失敗を説明できるわけではない。

`posthoc-saturation/results.json` に比率、モデル・source SHA、出力同値検査を記録した。
次は学習入力でも頻度を確認し、forward値を保ったまま飽和域の勾配を通す小規模比較を行う。
これは未実施の改善仮説であり、修正すれば画質が改善するという証明ではない。
[straight-through推定の原論文](https://arxiv.org/abs/1308.3432)は、硬い非線形演算を通す
代理勾配の考え方の参考になる。今回のclipへの適用は独自の実験で、真の微分とは区別する。
