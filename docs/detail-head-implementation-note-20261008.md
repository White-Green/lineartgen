# Detail branch implementation and validation

The optional `model.detail_channels` is0 by default. Nonzero values require a
source-conditioned UNet. The branch reads the noisy image, noise level, encoded
original rough guide, first pre-pooling encoder features and final decoder
features, then adds its scalar output before the existing residual conversion.
The tested plan uses raw guidance and floor0. There are no new runtime inputs.

All270 tests pass in375.49 seconds (three existing warnings). Five focused tests cover: inherited tensors and recursive outputs match at zero
initialization for widths8/16; serialized tensors restore exactly; the last
projection receives gradient immediately and hidden layers after one update;
invalid configurations fail; and a nonzero branch exports to ONNX and matches
JAX through actual145×129 BGRA input/output at1/2/5 denoising iterations, with
exact protected pixels and at most one byte of quantization difference. This is
a fixture parity test, not candidate quality or Windows DirectML testing.

The preflight performs12 candidate GPU updates (two arms, depths1–6) and restores
full state exactly. Every branch kernel and bias changes after these updates.
A six-update unchanged control reproduces the historical model, optimizer and
EMA state bitwise. Initial full-corpus quality equality and actual-run resume
checks are also required by the runner.

The extra parameter counts are1,313/3,777 on top of126,828. Counting convolution
multiply-accumulates for one512×512 UNet pass gives a guided baseline of
2,330,984,448. Width8 adds339,738,624 (+14.575%); width16 adds981,467,136 (+42.105%).
This excludes pooling/activation/resize costs and recursion and is not a measured
latency. The calculation is saved with per-block counts in
`runs/detail-head-fitting-20261008/static-compute.json`. A useful quality result
would still require measured inference performance.

Numerical source is frozen at `223b28ee4f1349cc76a84e812f8c3075d7f87769`, with218
source files,5 candidate settings and4 replay settings. Both archives preserve
that commit. The finite fitting experiment remains governed by the predeclared
[plan](detail-head-fitting-plan-20261008.md), not these implementation checks.
