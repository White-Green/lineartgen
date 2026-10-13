# 線画品質の現在地（CY完了）

**経路ごとの断面lossを加えても、目立つ枝欠損と近接線の融合が残った。**
16ch・512例で各6,000更新を比較し、断面平均と一部の欠損率の小改善に留まる。未採用。

追加診断で、欠損候補画素の67～72%が白上限を越え、clip後のfamily lossから直接修正されない
状態だった。次は誤った飽和にも勾配を流すinterval Charbonnierを比較する。
推論のclip・画像I/O・再帰は維持する。成功はMSEではなく許容線画の形と実画像で判断する。

- [CY: 経路別断面と飽和の診断](path-profile-results-20261013.md)
- [CW/CX: 候補固定×誤差関数](robust-family-results-20261013.md)
- [CV: データ数比較](family-expansion-results-20261013.md)
- [CU: 一つの清書を選ぶloss](plausible-family-results-20261013.md)

固定opacity・有限候補の合成診断を継続中。Krita・配布・封印データは変更していない。
