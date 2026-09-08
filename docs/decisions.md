# 技術選定（2026-09-08）

## パッケージ管理: uv

既存 Python 開発と連続性があり、venv・Python 版・extras・lock・build を扱えるので uv を採用した。
学習を Python 3.12、Krita runtime を Python 3.10 以上に分け、各 project に lock を置く。
uv workspace は全 member の Python 要求の共通部分になるため、runtime を workspace member にしない。
root は runtime をローカル path dependency として参照する。

資料: [Projects](https://docs.astral.sh/uv/guides/projects/)、
[Workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/)。

## 学習: Flax NNX + Optax + Orbax + Grain

Flax NNX は通常の Python class と状態を持つ Module を使い、JIT と自動微分を組み合わせる。
Burn の Learner のように学習全体を一つの API に委ねる構成ではない。
このプロジェクトでは `Trainer` に蓄積・評価・ログ・保存を集め、数値処理を `steps.py` に分けた。
recursive U-Net、複数 loss、ノイズ領域の試行錯誤を追いやすいことを優先した。

Keras 3 の JAX backend と `fit()` も候補になる。標準的な教師あり学習なら便利だが、
今回は再帰と複数解像度の勾配を明示したいので NNX を選んだ。
TensorFlow の導入は不要。データ列は Grain、optimizer は Optax、保存は Orbax、
TensorBoard ログは TensorBoardX を使う。

資料: [NNX basics](https://flax.readthedocs.io/en/latest/nnx_basics.html)、
[Flax data loaders](https://flax.readthedocs.io/en/latest/guides/data_loaders.html)、
[Orbax checkpoint guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html)、
[Keras custom JAX training](https://keras.io/guides/custom_train_step_in_jax/)。

## 大画像: pyvips と任意の tiled TIFF

32K PNG 全体を NumPy 配列として読み込む方法は crop 学習に不要なメモリを使う。
pyvips で必要領域を読み、長い実験では一度 tiled TIFF に変換する。
PNG の sequential decode はランダム crop の時間まで一定にはしない。
TIFF 変換もデータセット全体の複製容量と初期処理時間を使うため、任意の prepare コマンドにした。

資料: [pyvips introduction](https://libvips.github.io/pyvips/intro.html)、
[binary installation](https://libvips.github.io/pyvips/README.html)。

## 推論: ONNX Runtime

JAX は学習側で使い、Krita には ONNX と NumPy / ORT を配る。
JAX の Windows GPU 対応を Krita 配布の要件にしなくて済む。
現状の U-Net は通常の Conv / Pool / Relu / Resize なのでこの分離に適している。
複雑な制御フロー・新演算・dynamic shape の変換は自動的に保証されない。
実際に jax2onnx 0.16.1 の Pool / Resize の可変形状制約に当たったため、
対象を絞った ONNX graph 変換と export ごとの数値比較を入れた。

Windows では既存の ZIP 配布と複数ベンダーの GPU に合わせて DirectML を使う。
公式資料では DirectML は sustained engineering 扱いで新規 Windows 用には WinML も案内される。
この既存 Python plugin では ORT の DirectML provider を最小の変更で組み込める。
DirectML の要件に合わせて opset 20、sequential execution、memory pattern 無効化、
同じ session への推論直列化を設定する。GPU 速度は Windows 実機で別途測る。
現在は各解像度を NumPy と ORT 間で渡すため、転送コストの改善余地がある。

資料: [jax2onnx API](https://enpasos.github.io/jax2onnx/user_guide/api/)、
[known limitations](https://enpasos.github.io/jax2onnx/user_guide/known_limitations/)、
[DirectML provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)。

Python package の Requires-Python だけでは対象 wheel の存在は保証されない。
実際に ORT 1.24.3 は Python >=3.10 という metadata に対して cp310 wheel がなかった。
runtime の Python 3.10 用依存は、wheel を確認した CPU 1.23.2 / DirectML 1.22.0 に固定した。
NumPy と ORT は CPython minor ごとのバイナリなので配布 ZIP も minor 別になる。

資料: [ORT 1.23.2 files](https://pypi.org/project/onnxruntime/1.23.2/#files)、
[DirectML 1.22.0 files](https://pypi.org/project/onnxruntime-directml/1.22.0/#files)。

## AMD GPU 学習

ホストは Radeon 8060S (`gfx1151`)、ROCm 7.2.1。
AMD の 7.2.1 公式互換表は JAX / plugin 0.8.2 を記載している。
PyPI にこの版の plugin がなく、最新の同版 JAX / jaxlib / rocm7-plugin 0.11.1 を
隔離 venv で実機検証したところ、GPU 認識・畳み込み・逆伝播・保存が成功した。
この実測構成を `rocm` extra に固定した。公式の認定組み合わせを意味するものではない。
TheRock の新しい SDK 配布へシステムを更新する必要はなかった。

資料: [AMD JAX compatibility](https://rocm.docs.amd.com/en/docs-7.2.1/compatibility/ml-compatibility/jax-compatibility.html)、
[ROCm plugin 0.11.1](https://pypi.org/project/jax-rocm7-plugin/0.11.1/)、
[TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)。
