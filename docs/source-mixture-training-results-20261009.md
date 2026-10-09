# 実clean線画でのblur/mixed追加学習 — 2026-10-09

**どちらも採用しない。** mixedはmatched blur対照より重ね描きのにじみを10.24%減らしたが、F1改善は+.013（事前条件+.02未達）。細線とblur復元を悪化させ、追加学習前のモデルを総合的に上回らなかった。

同一のsource-signal raw4kから、891学習用clean source＋20%オンライン図形で4,000更新ずつ追加。`pixel`はblur確率.8、`mixed`は確率.8でtone/blur/strokesを均等選択。画像・mask・Gaussianの乱数、モデル130,605パラメータ、loss、sqrt(batch)LRは固定。Burn式のscale→再帰1–6を維持。段loss0。明示的one_passは学習forwardを変えない。[固定計画](source-mixture-training-plan-20261009.md)。

| 開発評価・全域強度.5 | 追加前initial | blur/pixel | mixed |
| --- | ---: | ---: | ---: |
| 実線画blur F1 | .944971 | .934999 | .918956 |
| 同・にじみ | .011933 | .025589 | .026631 |
| 実線画strokes F1 | .829934 | .825265 | .838266 |
| 同・にじみ | .123276 | .149988 | .134628 |
| 図形blur F1 | .905006 | .856270 | .773047 |
| 図形blur 切断率 | .007292 | .101042 | .155208 |
| 図形strokes F1 | .711129 | .694559 | .693438 |
| 白紙最大opacity | .040913 | .109194 | .100620 |

実画像各corruption familyは24条件、図形は64条件の平均。複数crop/変形が同じsourceに由来するため独立標本数ではない。cleanとtoneのF1には小幅な改善があるが、blurとの両立に失敗。mixedのにじみは初期モデルよりむしろ増えており、対照比の改善だけを採用理由にしない。

固定local図形ではPROBE blur.5のF1がpixel .838229に対しmixed .530924、濃度不足率.204785→.360357。PROBE細線のF1はmixedで0、alpha.5基準の切断率1。幅誤差は小さくても線が薄くなればよい復元ではない。local保持条件26項目、広域の事前チェック121項目中40項目に失敗した。途中1kのlocal失敗39項目から一部改善したが、最終基準を通過していない。

[実線画の比較](../runs/source-mixture-training-20261009/development/native-review/real.png)、[図形blur](../runs/source-mixture-training-20261009/development/native-review/blur-05.png)、[重ね描き](../runs/source-mixture-training-20261009/development/native-review/strokes-05.png)を含む、最終local8枚＋広域6枚を等倍で実際に確認した。mixedでは細線・曲線・細部が薄く、強い重ね描きは太さが残る。近接平行線は融合し、隙間も十分に分離しない。線周囲や編集境界に薄い副輪郭があり、白紙にも微小な端部残留がある。部分編集の保護画素は全条件で厳密一致。

最終primary128、local broad912、広域4,698条件（1,562開発case＋4白紙×initial/pixel/mixed）を評価。44個の実評価sourceはすべてvalidationで、training/final-testとの重複0。depth1–6の更新数は678/654/668/671/680/649。全段loss設定、データ再開一致、両条件の教師/mask対応を確認した。6/1k/4kの全状態復元とraw/EMA計8書き出しが完全一致。initialの広域1,566条件のmetricsは過去のsource-feature-broadと完全一致。各形状8回のGaussian drawは同じ出力なので独立8標本とは扱わない。

数値コード3e931df。初回c32d1f5はlossログキーの欠落で**最初のoptimizer更新前**に停止し、run一式を保存したうえで同じ初期値・seedから修正再開した。32件の関連テストを通過。途中1k画像の目視はPROBE blur.5の1枚のみ。[最終監査](../runs/source-mixture-training-20261009/final-audit.json)。

補助診断では、異なるclean線幅が8bit化後に同一のblur入力になる例を確認した。[識別可能性の記録](blur-opacity-ambiguity-20261009.md)。これは今回の全性能低下の原因を確定したものではない。

次は事前に固定したmixed raw4kを共通初期値とし、元画像条件のpatch識別器あり/なしを比較する。[次の計画](../../jax-source-patch-prior-20261009/docs/source-patch-prior-plan-20261009.md)。教師画像・crop分布の点検も行う。配布モデル、Krita UI、封印評価は変更していない。
