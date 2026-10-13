# DK: 全解像度のsource正規化

18,000更新、標準/劣化9,216行、512px128出力、runtime段数96出力を完了した。
標準8枚・劣化8枚・512px6枚・runtime6枚、補足oracle2枚をすべて等倍確認。
薄い線・薄い黒ベタと輪郭リングは大きく改善した。一方、近接線の灰色帯は残り、一部では悪化した。
製品採用はまだ行わず、改善した濃度処理を次の研究の出発点とする。

全解像度のinkを33x33局所peakで正規化し、peak alphaを.4へ揃えてからbox縮小する。
特徴とactive residual anchorの両方に使い、t=0では元sourceを完全に保持する。
同じ16ch・3残差block・15,809パラメータ、DIと同じ初期値・18,000batch・optimizer/LR・lossで学習した。
raw対照はDI plain raw18000を再利用し、重複学習したとは数えない。
初期パラメータは同一だが、初期関数はrawとnormalized sourceで意図的に異なる。

- faintの細線は濃くなり、faint/rough黒ベタはほぼ不透明で、強いリングもほぼ消えた。部分適用内でも同様。
- base/faint/darkで清書の濃さと太さが揃う。darkの分岐にrawで見られた厚い側線も減った。
- sharpの細線6番は逆にrawより灰色になった。soft/roughの近接線には灰色の余分な線が残る。
- old_probe36や512px scene1/2/5では、normalizedの灰色帯がrawより目立つ箇所がある。
- 512px scene4の長い線は滑らかになった。scene7の左側の交点もrawよりつながるが、上の接点・下の曲線との交点付近の小さな切れは残る。
  scene7の中央縦線gapは意図的なので欠陥と数えない。[評価の補足](scene512-interpretation-20261013.md)。

全体適用.5の共通許容族の断面距離/切断率。候補依存の問い合わせで分母は一定とは限らず、採否をこの数値だけでは決めない。

| 条件 | raw断面 | normalized断面 | raw切断 | normalized切断 |
|---|---:|---:|---:|---:|
| new_probe | .195685 | .200068 | .032041 | .021330 |
| old_probe | .193616 | .197112 | .034242 | .009272 |
| base | .177630 | .179694 | .028381 | .009719 |
| faint | .248372 | .179704 | .060668 | .009719 |
| dark | .199737 | .179694 | .014795 | .009719 |
| sharp | .235808 | .244825 | .030109 | .021384 |
| soft | .193249 | .259103 | .021645 | .003842 |
| rough | .235045 | .245022 | .062435 | .025996 |

数値commit `bc06593`。CPU22テスト、Ruffを通過。初回のeager/JIT比較で1.49e-8の丸め差があったため、
その比較のみatol2e-7として全22テストを再実行した。保護・保存復元の完全一致条件は維持。
GPU全6深さの初期出力は期待値と完全一致。rawの6更新再実行は重み最大1.49e-8、EMA5.08e-13、
normalizedの6更新再実行全状態は1.49e-8で既定atol1e-7/rtol1e-5を通過。
6/6000/18000全状態復元と8 raw/EMA exportは完全一致。
全18,000回の入力・候補・annotation hashはDI raw journalと一致。
再利用したrawと参考dilatedの評価出力も、標準/劣化各1,536ファイル・計6,144モデル対でDIと完全一致。

追加Gaussian2drawは標準/劣化各768ファイル対、512px64モデル/encoding対で完全一致。
source-onlyなので独立な品質標本とは数えない。入力uint8化による出力RMS平均はraw .001062、normalized .001393。
float/uint8の見た目は近いが、正規化で量子化への感度は少し増える。

128px停止・512px3段・strength.5・1passの追加比較では、元の126,765パラメータ/100k EMAは
この合成下描きで多くの線を消し、残る線にも粒状の灰色領域が出た。両15,809パラメータ研究モデルは
線を明瞭に追従する。これは構成・学習・入力処理全体の比較で、パラメータ数だけの因果比較ではない。
元モデルは実際の入力倍率1.1、研究モデルは実際の合成学習domainである1.0を使う。
出力thresholdなし。保護は各モデルに渡したencoded inputと完全一致。
研究configにはdata.image_scale=1.1が継承されており、将来の配布では実際の学習domainと整合させる必要がある。
normalizedのONNX exportは明示拒否している。Krita GUI・実際の配布物の実行試験ではない。
またnormalized anchorはt=0を保持してもt→0+の連続性を保証しない。

候補選択の追加診断では、非定数/非wide108例中、raw101例・normalized100例で既に中央offsetが最小だった。
段ごとの画素項の最小offsetが違う例は37/40例あるが、因果は未確認。
「最終出力が外側候補ばかりを選ぶ」という説明はこの結果では支持されない。
中央候補固定のcurriculumはまだ実施しない。

次は許容清書の線数を見直す。現在は位置を変えられても、近接した二本を一本へ整理する候補がない。
見た目では許容している整理をlossでは罰しているため、入力が曖昧な近接線に限って、
一本の明瞭な線へまとめる候補も許す比較を検討する。灰色帯の原因と確定したわけではない。

- [計画](full-source-normalization-plan-20261013.md)
- [完了記録](../runs/full-source-normalization-20261013/completion.json)
- [薄い入力の改善](../runs/full-source-normalization-20261013/stress/native-review/faint-whole05-0.png)
- [残る近接線の灰色](../runs/full-source-normalization-20261013/diagnostics/scene512/native-whole-2-3.png)
- [元の配布モデルとの比較](../runs/full-source-normalization-20261013/diagnostics/runtime-comparison/native-whole-2-3.png)
