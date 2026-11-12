# 大目標への関係：学習済みの成分判断をKrita APIへ同じ出力で接続する

このブランチは配布前のruntime準備であり、十分な学習済み候補の採用・配布ではない。
研究側の方針は`jax-component-learning-20261109/docs/component-readout-runtime-preparation-20261112.md`。
I本系列の6k継続学習とは独立して作業した。現在の同梱assetsは以前のpreviewのまま。

## 追加した読込形式

- 既存のformat1・`base_component_mean`は従来どおり動く。
- format2・`base_component_mean_max_mlp32`は、CNN139,186とreadout1,089の計140,275パラメータ。
  最終段stateの成分平均/最大32値を32→32→1 GELUへ通し、元の平均logitへ8倍して加える。
  閾値0、成分接続、描画、タイル、1〜6段再帰、BGRA入力は評価時と同じ。
- manifestの`readout`はローカルbasenameとSHA-256を持ち、CNNと別に検証する。
  `cnn_parameters=139186`、`readout_parameters=1089`、全体`parameters=140275`を検査する。
  NPZはhidden_kernel(32,32)、hidden_bias(32)、kernel(32,1)、bias(1)のfloat32有限配列だけを許す。
- JAX/Flaxをruntime依存へ追加しない。Krita pluginの入力や生成ログは変更していない。

## 完了したローカル検証

13テスト通過。別成分や非占有セルからの特徴混入、空成分、ゼロreadoutの旧判定一致、
位置/幅/edgeの不変、誤った配列名・shape・dtype・非有限値を検査した。
一時fixtureのI3k RAWを用い、256pxと1024pxの計3つの既存dev出力alphaを完全一致させた。
256pxでは1〜6段の全headを研究用ONNX/NumPy実装と完全一致させ、公開APIの出力も一致した。
blank、strength0、黒白不透明保護、固定seed、manifest不整合の拒否、JAX/Flax未importも確認した。
さらに従来同梱モデルの1〜6段と512px場面を保存済みBGRA参照へ完全一致させた。

根拠は研究側の共有runs内：
- `component-readout-runtime-functional-20261112/verification.json`
- `component-readout-runtime-probe3000-20261112/verification.json`
- `component-readout-runtime-legacy-parity-20261112/verification.json`

この3k fixtureは配布assetsへ入れず、wheel/ZIP/CI artifactも作っていない。
今回の全段照合はONNX/NumPy参照で、最終候補の全段JAX照合を済ませたという意味ではない。
Windows/DirectML・実Krita GUIは未検証。最終候補の十分な学習・未使用確認後に、全チェックを再実施する。

この準備のコード・旧manifestは検証runにhash登録した。最終assetsを配置するときはこのcommitから
別の配布worktreeを作り、準備時のソースと結果を保持する。main/releaseと既存previewは未変更。
