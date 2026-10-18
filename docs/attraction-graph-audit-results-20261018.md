# HT: projected anchor 表現監査結果

学習比較へ進める程度の表現性は確認した。40既知入力・全6段240教師を生成し、fineの12原寸ページすべてを確認。正解座標/幅/存在/辺を与えた診断であり、学習済み性能ではない。

通常24例では近傍anchorが多数あっても線は太らず、曲線・交差・gap・taper・塗りがほぼ元の不透明パスを保つ。fineの全40例で元pathの未表現数0。0.25/0.5px近接2本のfixtureは旧cell方式でcoverage差/ink約0.502だったが、projectionは数値上ほぼ0かつgraph2成分。0.2px短線も点に潰れず描画された。これは原寸では十分判別できないため数値検証を併記する。

限界もある。0.25/1px gapは別graphを保つが端が後退し、coverage差/inkが.00481/.00244になる。3本の非常に近い平行線は最短pathの領域がgridに載らず表現を落とし得ることをテストで確認した。交差を別pathとして描くため、graph component数は見た目の接続数ではない。通常24例のcoverage差/ink最大はprobe48の.00346。粗い段の教師はfinite/統計確認のみで、fineと同等品質とは主張しない。

5 semantic tests pass (.92s)、CLI/lint/diff pass。prepare2.180s、run18.284s、初回終了0。HQのCPU参照を再利用し、以前のDQ参照許容値失敗は変更していない。新規学習0、fresh16未使用、本番/Krita/export変更なし。

次は同じ22725パラメータのCNNで投影offsetを学習する。HRの学習前初期値と同じ1200入力系列を使い、存在/辺の正負別平均BCEとcount正規化の2条件を事前固定する。教師が既知なら描けるという今回の結果と、実際に入力から予測できるかを区別する。

Artifacts: `runs/attraction-graph-audit-20261018`、frozen `6e54953`。
