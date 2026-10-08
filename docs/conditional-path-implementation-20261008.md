# Conditional-path implementation and validation

The experimental implementation supports explicit remaining progress and paired
rough-to-clean intermediate states. Defaults preserve the legacy numerical path.
The independent broad detail-head training continues in its existing checkout.
No new learned candidate or production model has been accepted.

`model.progress_conditioning=true` requires a guided residual UNet. It adds
bias-free, zero-initialized1×1 projections of `1-u` to the first encoder and
inherited output head, and to the detail head when present. Construction occurs
after inherited layers. The full width16 model adds23 parameters, for130,628 in
total. Ordinary source guidance remains the original rough input. At the first
state, the progress projections vanish even after learning.

`diffusion.training_path` selects `rough` or `bridge`. Both progress-conditioned
arms share exactly the same strength, mask, Gaussian and progress draws. Progress
uses a separate folded key, with .5 starting-state probability and otherwise
uniform[.05,1]. Paired fixed-source training now passes the clean teacher and
rough source separately; the former is used to build targets and intermediate
training states, never as the guide. Each local arm prepares its own configured
path. An explicit warm-start method list avoids accidentally treating the new
arm names as random initializations.

The JAX and NumPy samplers provide progress to each recursion stage and keep the
existing finite update. The ONNX graph has an optional fifth `progress` input,
advertised by metadata. The runtime supports both old3/4-input graphs and this
5-input graph. `LineartModel.infer` and the Krita BGRA interface are unchanged;
deploying a progress-conditioned model would require its accompanying runtime.

All **279 tests pass**, including9 new tests, in599.06 seconds on CPU. There are
three inherited deprecation warnings. New tests cover the exact initial state,
paired path difference, protection at coarse mask boundaries, oracle trajectories
and progress in both samplers, inherited tensors/function/serialization, actual
loss gradients through each new projection, rejected invalid configurations,
and nonzero-projection ONNX/BGRA agreement at1/2/5 iterations. The separate targeted
run passed9 tests in45.15 seconds. Source/test hashes are recorded and unchanged.
The new preparation/evaluation/rendering helpers pass Ruff and compile checks;
actual preparation verifies copied data and that paired configs differ only in
`diffusion.training_path`.

The fixed local study is prepared under
`runs/conditional-path-fitting-20261008/`, with a historical width16 replay root.
Before actual training, GPU updates at all six recursion depths, exact historical
replay, initial forward comparisons and the first-update restore remain required.
Its evaluator saves and verifies individual complete blocks, checks all expected
conditions and preserves arrays, per-case metrics, summaries and hashes. It
re-evaluates the original local width16 checkpoint at1/2/5 iterations and requires
its one-step arrays to equal the historical arrays. Raw five-step comparisons
follow the predeclared plan; no EMA or iteration-count substitution is allowed.

Ten final native sheets are specified in the renderer: five main sheets, two
blur repeat sheets, two blur partial-mask sheets and a blank repeat sheet. Rendering
does not mark inspection complete. These use stored float inputs; actual BGRA
quality review, broader testing, independent seed and sealed new draws would
still be needed for any advancing candidate. GPU preflight and learned-output
quality have not yet been established for this study.
