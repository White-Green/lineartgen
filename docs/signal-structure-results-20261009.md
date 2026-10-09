# 元画像特徴でのregion・接続loss比較 — 2026-10-09

両追加lossを不採用とした。にじみと一部の隙間は改善したが、濃度・線幅を保つ条件を通過していない。

| raw4k | pixel | region.1 | region.1+connection.1 |
| --- | ---: | ---: | ---: |
| TRAIN blur halo | 0.031929 | 0.020881 (34.60%減) | 0.021078 (33.99%減) |
| PROBE blur halo | 0.028251 | 0.020204 (28.49%減) | 0.020196 (28.51%減) |
| PROBE blur.5 全域 F1 | .830925 | .822991 | .821674 |
| 同・濃度不足率 | .179243 | .211795 | .215753 |
| PROBE細線 幅誤差px | .555487 | .545255 | .394291 |
| PROBE gap 切断率 | 0 | .166667 | 0 |
| PROBE gap 誤接続率 | .666667 | .333333 | .333333 |
| PROBE gap 中央opacity | .487409 | .364535 | .267568 |
| 保持条件の失敗数 | — | 21 | 31 |

失敗数は重複する比較条件の数。接続lossはPROBE gap中央濃度をregion比26.60%減らしたが、TRAIN側では逆に増えた。regionはPROBE gapの切断も増加。各モデルのprimary fit合格は両splitとも0/32。8回のGaussian drawは全モデルで同じ出力になり、独立標本は各split4形状である。

[事前計画](../../jax-signal-structure-20261009/docs/signal-structure-plan-20261009.md)どおり、同一初期値・新規形状・4,000更新で比較した。130,605パラメータ、再帰1–6、段loss0、sqrt(batch)LR、縮小後Gaussianを維持。対照は完了済みsource-signalを厳密再利用。1,000時点も保持条件16/18項目失敗で、途中値を選んでいない。

最終の8枚を等倍設定で確認し、数値から選んだ白紙の最悪端部1枚も確認した。細線や曲線は整うが薄さが残り、PROBEの隙間も完全には開かない。白紙は広域では白く見えるが、編集マスクと画像下端の交点に微小な残留がある。最大opacityはpixel .040913、region .179714、structure .249070で、完全白とは言えない。保護領域は全条件で厳密一致。

数値コード295e1b4を固定。対照と実学習の6更新再現、1k/4k全状態復元、候補raw/EMA計8exportの一致を確認。各endpointの対照primary64・broad456出力/metricsが元モデルと厳密一致。最終primary192・broad1,368条件を評価。中間画像はPROBE blur.5の1枚のみ目視し、最終とは区別した。

[最終比較画像](../../jax-signal-structure-20261009/runs/signal-structure-20261009/broad/step-4000/probe-blur-0.5-native.png)、[監査](../../jax-signal-structure-20261009/runs/signal-structure-20261009/final-audit.json)。

追加lossの結果とは独立に、配布モデルを含む6,568条件の広域・反復診断を開始した。元画像特徴での局所濃度分布loss比較も固定して順番待ち。source特徴が反復中の状態を使わない点は別途診断中で、Krita互換の品質を確認した意味ではない。配布モデル・封印評価は変更していない。
