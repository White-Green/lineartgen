# 代理勾配と背景の局所誤差の比較

直前の [2k比較](output-gradient-results-20261007.md) では、contrastありの代理勾配は線を消し、
contrastなしのink保持lossは細線欠落を減らす一方で背景の点状汚れを増やした。
そこで次の3条件を同じ100k EMAから新たに比較する。

| 条件 | 切り分ける変更 | 対応する完了済み対照 |
| --- | --- | --- |
| ste_no_contrast | 元のlossからcontrastを外し、clipに代理勾配 | hard no_contrast、ste_control |
| ste_neighbor_guard | 既存のneighbor保持lossに代理勾配 | hard neighbor_guard |
| ste_ink_paper | ste_ink_guardに背景の局所最大誤差を追加 | ste_ink_guard |

2kで固定screenと画像確認を通る条件だけを、最大2条件、主halo改善の合計順に10kへ進める。
10k cosine clock・同じseed20261007・実効batch16・peak LR4e-5・warmup100を維持。
縮小後ノイズ、合成下描き、再帰1〜6段、最高段の直接lossと小さい段への再帰勾配も変更しない。
パラメータ数126,765、モデルforward、Krita入出力に変更はない。

## 背景の局所誤差

教師opacity≤0.02かつ編集可能な画素について、教師を超える予測opacityの二乗誤差を作る。
8×8の重ならないタイルごとに最大を取り、編集可能な背景を含むタイルだけで平均する。
画像ごとに正規化してbatch平均する。端の不完全なタイルも1個として数える。
これにより、広い白地の平均に埋もれる単発の誤inkも評価できる。
元の画素loss等は残るので、最大以外の画素の学習を打ち切るわけではない。

教師の線や保護領域の誤差をpoolへ入れない。近傍背景と遠方背景の両方を対象にする。
連続なアンチエイリアス教師がそのまま正解で、正しい出力のlossと勾配は0。
二値化や線の位相保持を保証する項ではない。

係数は評価画像を使わず、固定した12学習batch、再帰1〜6段を各2回、実際の4×4勾配蓄積で校正した。
ste_ink_guardのparameter勾配に対し、中央値0.1倍・最大0.5倍を上限として小さい方を採る。
得られた係数は0.3397133594097619。係数選択でmodelやoptimizerを更新していない。
他のloss係数を再調整せず、採用基準も緩めない。

## 比較・保存

rootは `runs/paper-guard-20261007-screen`。
`plan.json` を校正前に保存し、`calibration.json` に12batchのloss、parameter勾配normとcosineを記録する。
入力・初期値・設定・学習コード・最終batch hashを照合し、完了済みhard/STE対照を再利用する。
これは独立した対照再学習や別seed確認ではない。

`compare_paper_guard_study.py` は2k時点で6つの既存対照と新しい3条件を統合し、
EMA主判定・raw感度・最終採用条件の診断を分けて記録する。
複数rootの画像を組み合わせる際もtarget/source/noisy/initial画素が完全一致することを確認する。
10k対照が存在しない条件との10k比較は作らない。
採用には実際の出力、反復・白紙、別fine-tuning seed、新しい合成条件での検証が引き続き必要。
