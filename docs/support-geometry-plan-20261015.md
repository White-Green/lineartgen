# EV: 入力supportだけで融合教師を許可する比較学習

EUの288条件監査で固定8px上限を外すと14条件が追加され、probe37（間隔8.053524px）も許可されると分かった。これは既知例から得た探索仮説であり、fresh確認ではない。教師選択の固定上限だけを変更し、線の断片化が改善するか調べる。

初期値はEQ mixed raw18k（group6、16402 parameters）。ET controlの同じ初期値、同じoptimizer初期化、seed2026101440、親sampling step18000–23999を再使用する。B4、base LR1e-4 × sqrt(4)=peak2e-4→minimum2e-5、warmup0、clip1、EMA.999、6000stepを3000ずつ継続。再帰1–6、全段loss係数.5^level、幅4–32px拡張、独立opacity重ね合わせ、D4、粗化・部分mask・縮小後ノイズは全て維持する。context dropoutは行わない。

新しいsupport armだけを6000step学習する。対照群は完了済みET controlを再利用する。最初の6stepの実パラメータ・optimizer・EMAを再生してET checkpoint6と比較し、既定ROCm許容誤差atol5e-6/rtol1e-4を検証する。6深度のall-keep gradientは元merged objectiveと完全一致を確認する。全6000stepで入力と元teacherのdigestをET journalと比較する。新しく許可されなかったbatchでは候補payloadも完全一致を要求する。旧許可を取り消さず、新許可は間隔8px超のparallelに限る。27個の既存融合fieldはcache生成時に完全一致を確認する。

merge geometryのMAX_SEPARATIONだけを無限大にする。同じ17断面/65点/中央min/peak>=.02/ratio>=.6/fraction>=.75/valid>=8を使用する。二本と一本の各9ずらし候補から、全段で一貫したfamilyを一つ選ぶ。元候補を残す。入力に白い隙間がある二本を無条件に融合しない。

保存は0/6/3000/6000step raw/EMA（新規8export、対照の既存8export）。評価はEQ初期/ET control/support raw6000の9516出力、既定34等倍ページ（通常12、粗化10、512px複合6、幅sweep6）。ETの初期・control保存出力と完全一致を確認し、モデル状態、input/teacher/code hash、有限値・範囲・保護領域不変・網羅性を確認する。

判定はprobe37の断片が連続した明瞭な線になること、scene1の融合・分岐が維持されること、広い線の副線・scene7の余計な切れを悪化させないこと。通常曲線、交差、意図的gap、blank/fill、部分編集を同じページで確認する。MSE順位では採用しない。これだけでwide副線やscene7が解決すると仮定しない。改善しても広い視覚検証を通るまではKrita更新や未使用16例の確認をしない。
