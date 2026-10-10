# 線画復元実験の現状 — 2026-10-10更新

**採用条件を満たす新モデルはまだ得られていない。** 現時点で最も有望なのは、追加Gaussianではなく元の下描きからCNN特徴を取る130,605パラメータの候補E。配布中100kモデルは置き換えていない。

候補Eは、既存コーパスをぼかした入力の強度.5・全域平均で、配布モデルに対してhaloが76.86%少なく、F1が.854720→.944971、濃度不足が.168443→.048584だった。ただし学習履歴の異なるモデルの観測比較であり、太い重ね線・近接平行線・隙間は未解決。
[広域診断と画像](../../jax-source-feature-broad-20261009/docs/source-feature-broad-results-20261009.md)。

| 完了した追加比較 | 結果 |
|---|---|
| F: region / region+connection loss、4k | halo約28–35%減、濃度などの保持条件21/31項目失敗。不採用 |
| H: 局所濃度分布loss、4k | halo約65–77%減るが、PROBE濃度不足.179→.375、38項目失敗。不採用 |
| I: 明示的1回復元とBGRA/ONNX | 実BGRA84条件＋seed変更28条件、最大差1階調、保護画素一致。実装契約は検証済み、画質採用とは別 |
| J: コーパス＋20%図形のblur対mixed拡張、各4k | tone改善はあるが、強いblurの細線保持が悪化。広域40項目失敗。不採用 |
| K: source条件付きpatch判別器、warmup250＋各2k | halo減少は対照比blur3.11%／strokes0.73%。広域165チェック中17項目失敗、最終14枚確認。不採用 |

詳細: [F](../../jax-signal-structure-20261009/docs/signal-structure-results-20261009.md)、[H](../../jax-signal-transport-20261009/docs/signal-transport-results-20261009.md)、[I](../../jax-one-pass-restoration-20261009/docs/one-pass-restoration-results-20261009.md)、[J](../../jax-source-mixture-training-20261009/docs/source-mixture-training-results-20261009.md)、[K](../../jax-source-patch-prior-20261009/docs/source-patch-prior-results-20261009.md)。

教師画像の点検で、高倍率画像自体に幅広いグレー領域がある例を確認した。256学習クロップでは、倍率別の教師で白紙39枚、元解像度に切り替えると0枚だった。幅広い中間グレーの画素割合も減った。少数の点検結果であり、コーパス全体の比率や出力問題の単独原因を確定したわけではない。
[教師点検](../../jax-native-target-training-20261009/docs/native-teacher-audit-20261009.md)。

**L完了・不採用:** 元解像度教師ではコーパスblur/strokesのhaloが対照比61.83%/54.57%減少。
ただし黒い内部まで薄くする副作用があり、保持条件70/165項目失敗。部分編集の黒い領域では
平均alphaが1から.65/.62まで低下した。最終14枚を確認した。
[結果](../../jax-native-target-training-20261009/docs/native-target-training-results-20261009.md)。

**M完了・不採用:** 同じ元解像度教師で画素lossの強度重みを均一にすると細線は濃くなったが、
コーパスblur/strokesのhaloは対照比19.05%/12.65%増えた。黒い内部のalphaも.68程度にとどまり、
保持条件53/165項目失敗。最終14枚を確認した。
[結果](../../jax-source-flat-loss-20261009/docs/source-flat-loss-results-20261009.md)。

**進行中N:** 元解像度教師に白紙・黒ベタ・塗りつぶした図形・太い線を追加する比較。
期待比率はコーパス70%・従来の細線図形20%・新しい合成教師10%。
逆強度重みと均一重みの2条件を同じ初期値・同じ入力列から各4k更新し、L/Mの対照と比べる。
旧166バッチ、元の図形、置換しないコーパス画像、再開後の入力、全再帰段の準備配列が一致し、
23テストを通過。数値コードa1f7aa7で固定し、学習開始済み。効果はまだ未検証。
[固定計画](../../jax-ink-area-coverage-20261010/docs/ink-area-coverage-plan-20261010.md)。

下描きはすべて合成。評価の`real`は既存コーパス画像という意味で、自然な下描き／清書の対ではない。コーパス生成スクリプトには画像生成モデルとSwinIR拡大が含まれる。正確な細線幅は、ぼけと透明度の組合せから一意に決まらない場合も実証した。[8bit入力の曖昧性](../../jax-source-mixture-training-20261009/docs/blur-opacity-ambiguity-20261009.md)。

再帰1–6、段loss倍率0、sqrt(batch)LR、縮小後Gaussian、保護画素の一致を維持。source特徴モデルは追加Gaussianのseedを変えても同じ出力なので、反復drawを独立標本に数えない。Iの1回復元モードは新しいruntime metadata形式2を要し、配布パッケージは未変更。

独立学習seed・封印した新条件の確認・Windows/Krita GUI検証は未実施。繰り返し見ている開発出力での改善だけを最終確認とは扱わない。十分な最小モデルサイズも未確定。
