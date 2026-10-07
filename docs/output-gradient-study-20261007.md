# 出力clipの勾配を通す比較実験

10kの保持loss比較は、haloを10〜18%減らした一方で、細線・線同士の分離の採用条件に失敗した。
調べたstrong blurの細線等では、教師線画素の約93〜100%が白側のclip範囲外にあり、
その出力headを通る勾配がゼロになっていた。初期100kモデルにも同じ現象がある。
詳細は [10k結果](ink-structure-10k-results-20261007.md)。

学習用にも、事前固定した12batch・再帰1〜6段で同じ診断を行った。
最高解像度の編集可能な教師線1,616,469画素のうち87,126画素が白側に飽和。
ink不足量の11.56%がその画素にあり、batch別では5.61〜43.36%。
これは少数の診断batchでの値で、学習集合全体の推定ではない。
他の画素や再帰経路を介した共有パラメータの更新までゼロという意味でもない。

## 変更するもの

`output_clip_gradient="straight_through"` をUNetに追加した。
forwardは従来どおり正確な `clip(z, -1, 1)`、backwardは `dz_out/dz=1` の代理微分にする。
クリップの真の微分ではない。対象は全再帰段で、段間lossの重みは変えない。
[Bengioらのstraight-through推定](https://arxiv.org/abs/1308.3432)を考え方の参考とするが、
このモデルで画質が改善することは別途実験で確かめる。

画像上の誤差がある飽和画素にも修復方向の勾配を渡せる。正しい白へ飽和している画素は、
画素MSEの誤差がゼロならその項の勾配もゼロのまま。
残差headのnoise=0ではrawへの微分は引き続き0で、保護領域の値を保つ。
再帰を通る代理勾配は実際の合成関数の微分ではなく、これも比較対象の変更に含める。

モデルのparameters・channels・演算のforward・Krita入出力は同じ。
ONNX書き出しではモデルを複製し、同値な通常Clip経路だけを出力する。
元の学習モデルの設定を変更しない。PythonやKrita側の修正は不要。

## 比較と固定条件

| 新しい条件 | 対応する既存のhard clip条件 | loss |
| --- | --- | --- |
| ste_control | control | 元のtone MSE、identity、edge、contrast |
| ste_ink_guard | ink_guard | contrastなし、領域loss、ink保持loss |

全条件は同じ100k EMA起点、train seed20261007、batch16、peak LR4e-5、warmup100、
10k cosine clock。再帰1–6、段loss倍率、sqrt(batch)LR、縮小後ノイズ、劣化生成・入力順を維持。
loss係数を再校正しない。まず2kで比較し、固定screenを通る条件のみ全状態を保持して10kへ進める。

既存hard条件の完了済み2k/10k軌跡を再利用する。
新コードのhard条件で元の6batch GPU更新を再生し、control / ink_guardともmodel・optimizer・EMAが
元の保存状態と**全配列完全一致**することを確認した。新たな独立seedの対照とは扱わない。
比較時にも元source・設定・入力・最終clean batch・initial出力・ケース条件・モデルSHAを照合する。

採用条件は直前の実験から変更しない。2k screenでは主haloの一方で5%以上改善し、
指定14群でink deficit +2ポイント、missing/cut/bridge +3ポイント以内。
10kの最終条件は両主haloで10%以上改善し、線の保持・幅・位置・遠方背景・F1の厳しい条件を全て判定する。
採用には実際の画像、raw感度、反復、白紙、別fine-tuning seed、新しい合成入力の確認が必要。

## 検証・再現

- 代理勾配の向きと正しい飽和白のゼロ勾配、noise=0の保護画素勾配を検査。
- 同じ重みで1/2/5回再帰推論、保存復元のforward値はbitwise一致。
- 学習12batchの全再帰段でもhard / surrogateのforward・clip直前値が完全一致。
- 2条件×6段の実効batch16 GPU更新で有限なloss/gradientを確認し、全状態の保存復元が完全一致。
- 代理勾配を使う実更新も、中断・再開ありとなしでmodel / optimizer / EMAが完全一致。
- 小型UNetのONNX exportはdynamic形状でJAXとの許容誤差内一致を確認。

root: `runs/output-gradient-20261007-screen`。
`training-saturation.json`、`preflight/results.json`、`hard-baseline-replay-audit.json` が実測記録。
`compare_output_gradient_study.py` が外部hard対照を統合する。
自動watcherの2条件だけの集計には対照がないため、それだけで採否を決めない。

準備時点では学習後の品質は未確認。飽和域の勾配を通せたことだけでは採用しない。
