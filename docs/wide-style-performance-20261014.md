# DMの実行前性能確認

参照範囲87pxの比較を完遂するため、計算上等価な畳み込みを検証した。
モデル容量・参照範囲・教師・lossの変更による速度比較ではない。

初回4963d51の標準dilationでは1群1更新が約.81秒だった。
340更新で停止し、journalと6更新checkpointを元worktreeに保存した。完了実験には数えない。
dilation16だけを剰余batchのstride1畳み込みへ変換した第2試行0e5cf74では約.46秒だった。
旧実装6更新とのraw/EMA差は最大8.95e-8で基準を満たしたが、dilation4の勾配計算も遅く、473更新で保存・停止した。

GPUで単一畳み込みのforward・入力/重み勾配を比較したところ、256pxではdilation4の直接計算.1419秒に対し
剰余batch方式.00431秒。出力最大差3.34e-6、重み勾配最大差1.03e-7。
両者ともfloat32なのでbit exactとはしない。事前のatol1e-5/rtol1e-4を満たした。
dilation16でも同じ変換を検証済みで、末尾をpaddingして逆変換後にcropすることで非整除サイズも扱える。

最終試行ではdilation>1に同じ変換を使い、dilation1は元の経路をそのまま呼ぶ。
CPU33テスト（29.12秒）とRuffを通過。2/3/4/8/16の出力・両勾配、bias、RF87正規化モデルの実更新・再開・samplerを含む。
本学習とは別の使い捨てモデルで18更新を試走し、compile後の12更新平均は.1030秒、深さ6は約.109秒だった。
データ生成・転送・hash・保存を含む総時間ではない。初期モデルと本学習checkpointはこの試走で変更していない。

旧試行の場所:

- `../jax-wide-style-context-20261014/runs/wide-style-context-20261014/attempt-status.json`
- `../jax-wide-style-context-replay-20261014/runs/wide-style-context-20261014/attempt-status.json`

各runのdiagnosticsに演算ベンチマークを保存した。最終worktreeの`diagnostics/update-performance/results.json`は
18更新の時間と使ったコードhashを記録する。この速度確認は画質改善の証拠として扱わない。
