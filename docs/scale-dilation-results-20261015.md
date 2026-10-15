# EW: 再帰段ごとの dilation 診断（2026-10-15）

固定した EQ mixed 18k と ET dropout 6k に対し、既知の6入力を float / uint8、深さ6で評価した。72出力、16枚の等倍比較ページを確認した。新規学習・採用・予約済み新規入力の評価は行っていない。

変更は畳み込みの dilation のみ。元の1/4/16、各段で `max(1, d // 2**level)` とする physical、マップ寸法から上限を設ける bounded を比較した。重み、正規化、再帰状態の受け渡し、出力変換は同一。小さいマップでは大きい dilation の中心以外のタップが全てパディング領域に出ることを単体検証したが、それだけで不良出力の原因が確定したとは扱わない。

physical と ET dropout の組み合わせで wide61 の長い副線がかなり薄く短くなる。一方、薄い副線と端部の破片は残る。wide62 の周囲の点も多少減るが主線・端部が太くなる例がある。bounded の改善は小さい。probe37 の断片化、scene7 の余分な切れ目は全方式に残る。scene1 の濃い融合線・接続した分岐・上部の短線は維持する。wide58 は概ね良好だが physical で端が少し膨らむ。

CPU 3テスト（4.28秒）と Ruff が通過。GPU 評価は17.379秒、終了コード0。legacy 24出力は元の推論と完全一致、EQ mixed の12出力は ER 深さ6と完全一致。深さ1の12比較も元実装と完全一致。モデル状態の不変性と全出力の有限性・範囲・ハッシュを確認した。

固定重みでの構造変更は学習時の分布を変えるため、これを学習済み改善として採用しない。次は EQ mixed raw18k から physical を6,000更新学習し、ET dropout を対照として、入力・教師・dropout・学習率・更新回数を揃える。パラメータ数は16,402のまま。EV の support-only 条件は引き継がない。

証跡: `runs/scale-dilation-20261015/completion.json`、`evaluation/visual-review.json`、`diagnostics/decision.json`。凍結コミット `7e251e4`。
