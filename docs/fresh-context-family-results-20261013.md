# DI: 新しい初期値・同容量の参照範囲比較

各18,000更新と評価を完了し、標準8枚・劣化8枚・512px6枚をすべて等倍で確認した。
参照範囲31pxのdilatedは薄い/粗い黒ベタの内部と一部の線の滑らかさを改善したが、
灰色側線、交点の切れ、薄い黒ベタの輪郭リングは残った。両構成とも製品採用しない。

同じ初期値から、16ch・3残差block・15,809パラメータのplain/dilatedを学習。
単一段の参照範囲は15x15/31x31、外側の画像再帰は両者に維持。
同じ512合成形状、透明度/ぼかし/揺れの拡張、2pxの複数許容清書を使用。
全体で一候補を選ぶlinear Charbonnier＋.25断面＋.05range、再帰1–6、各段loss、sqrt(batch)LRを維持。
DDの学習済み重みは参考評価のみ。DDとの違いを初期値だけの因果比較とは扱わない。

- old_train24の短い枝、old_train30/new_train30/old_probe30/new_probe30の交点は両構成で不安定。
- old_probe36はplainが比較的明瞭な一本の曲線を出し、dilatedは灰色が目立つ。曖昧な複数線を一本にすること自体は失敗と数えない。
- faint/roughの黒内部はdilatedがかなり濃くなり、roughのリングも弱まった。faintの明るいリングと内部のむらは残った。
- soft/roughの近接曲線の灰色側線は残った。通常の黒ベタにも細い周辺の模様があり、dilatedで目立つ。
- 512pxではscene1の平行曲線とscene4の長い線がdilatedで滑らか。一方scene2の近接曲線は灰色の不規則な領域が増えた。
  scene5の側線、scene7の縦線/交点の途切れは両方に残る。部分適用内でも同様。float/uint8は見た目が近い。

全体適用.5での共通許容族の断面距離 / 切断率。候補依存の接続問い合わせなので分母は一定とは限らない。

| 条件 | plain断面 | dilated断面 | plain切断 | dilated切断 |
|---|---:|---:|---:|---:|
| new_probe | .195685 | .190490 | .032041 | .025884 |
| old_probe | .193616 | .191587 | .034242 | .021238 |
| base | .177630 | .176153 | .028381 | .013323 |
| faint | .248372 | .216346 | .060668 | .021983 |
| dark | .199737 | .209785 | .014795 | .009619 |
| sharp | .235808 | .213873 | .030109 | .019698 |
| soft | .193249 | .241816 | .021645 | .005820 |
| rough | .235045 | .225468 | .062435 | .029816 |

数値commit `4ae4ef0`。CPU23テスト、修正後の関連4テスト再確認、Ruffを通過。
実際の入力勾配でも参照範囲の差を確認した。全6深さの初期出力と初期パラメータは完全一致。
最初6更新の再実行状態差は両者1.49e-8で事前固定atol1e-7/rtol1e-5を通過（bit exactとは主張しない）。
6/6000/18000全状態復元と16raw/EMA exportは完全一致。18,000回の入力/候補/annotation hashもarm間一致。
学習driverは保存監査・training_complete後にexit143となり、未実行だった評価のみ別driverで完了した。
学習結果を選び直したり、更新を重複実行したりしていない。

標準/劣化9,216行と追加512px128出力。保護領域は完全一致。
追加Gaussianを変えた標準/劣化それぞれ768ファイル対、512px64モデル/encoding対は完全一致。
source-onlyなので独立な品質標本とは数えない。入力uint8化による出力RMSの平均はplain .001062、dilated .000822。
学習step4106時点の直近600更新の平均update時間はplain .098秒、dilated .463秒（準備/hashを除く）。
このROCm環境の観測で、推論速度の比較ではない。

合成のみ。ONNX/Krita GUI未検証。dilation>1のexportは明示的に拒否しており、製品/配布は変更していない。
参照範囲だけでは連続性が解決しないため、次は端点・交点を局所領域としてlossに明示する比較を行う。
長い経路の平均に局所的な切れが埋もれる可能性を検証する。原因の確定ではない。

- [計画](fresh-context-family-plan-20261013.md)
- [完了記録](../runs/fresh-context-family-20261013/completion.json)
- [薄い入力](../runs/fresh-context-family-20261013/stress/native-review/faint-whole05-0.png)
- [交点](../runs/fresh-context-family-20261013/evaluation/native-review/new_probe-whole05-1.png)
- [512px](../runs/fresh-context-family-20261013/diagnostics/scene512/native-whole-6-7.png)
