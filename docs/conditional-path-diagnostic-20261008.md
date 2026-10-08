# Conditional path: training and repeated inference

The current synthetic rough training distribution differs from the intermediate
states of an ideal, perfectly restoring sampler. A CPU diagnostic confirms this
algebra against the existing NumPy runtime. This is a reason to test a different
training path, not evidence that the mismatch causes the learned gray fringes.
The two ongoing broad detail-head trajectories remain unchanged.

## Current objective

Let `r` be the original rough source, `c` the masked clean target, `s` the initial
noise-strength map and `e` independent Gaussian noise. All arrays below are at
one pyramid level: form image/mask pyramids first and generate independent noise
at each resolution. Set the target equal to the rough source on protected pixels
before forming its pyramid, as in the existing training implementation.

Current training samples `x = sqrt(1-t) r + sqrt(t) e` and supervises `c`.
With the exact rough guide available, `x` supplies no additional information
about `c` conditional on `r,t`. Under idealized unconstrained pixel MSE, the
optimal prediction is therefore `E[c | r,t]`. This does not prove the actual
network ignores its noisy input: it has finite capacity, clipping, recursive
conditioning and additional regularization. Nor does it prove the corruption is
ambiguous; the known-operator inverse diagnostic shows much of the fixed float
blur information is recoverable.

## A path compatible with the current sampler

Write `u` for remaining progress, from 1 at the first forward pass to 0 at the
final state. The current sampler uses `t=s*u`. If its predicted clean image were
exactly `c` at every step, its state would be

```
x(u) = sqrt(1-s*u) c + sqrt(u)*sqrt(1-s) (r-c) + sqrt(s*u) e.
```

At `u=1` this is exactly the current noisy rough input. At `u=0` it is `c`.
Keeping a prediction fixed while moving from `u` to `v` gives the existing update

```
x(v) = sqrt(v/u) x(u)
     + [sqrt(1-s*v) - sqrt(v/u)*sqrt(1-s*u)] predicted_c.
```

The difference from current training at the same effective strength `t=s*u` is

```
delta(u) = [sqrt(u)*sqrt(1-s) - sqrt(1-s*u)] (r-c).
```

It is zero for clean identity pairs and at the starting state; it need not be
small for rough inputs during later iterations. The proposed training change
would sample this intermediate path using the paired teacher, while inference
would still receive only the original rough input and its mask. Progress must
be supplied separately: effective strength `s*u` alone cannot distinguish the
starting strength from progress. No upscaled clean prediction would replace the
rough input or its guide.

For fixed `r,c,s` and interior `u`, the path derivative simplifies to

```
dx/du = x/(2u) - c/[2u*sqrt(1-s*u)].
```

Thus a clean-target regressor can parameterize the conditional vector field.
Its pixel MSE differs from velocity MSE by the positive factor
`1/[4u²(1-s*u)]` when `s,u` are conditioned on. This gives the same ideal
conditional mean optimum, not the same optimization weighting or finite-model
solution. Existing edge/contrast terms further change that ideal objective.
Zero-noise coordinates remain fixed; the formula is used on active coordinates.
The endpoint velocity is singular, while the existing finite update has a
well-defined endpoint. These qualifications matter for any implementation.


The existing pixel term also balances tones using teacher-dependent weights, so
its unconstrained optimum is generally a weighted conditional mean. Moreover,
only the highest resolution has direct loss (stage multiplier0); smaller states
receive gradients through recursion. The per-level oracle identity therefore does
not establish a correctly learned joint probability flow across the full pyramid.
Both choices are preserved in the paired probe. A successful comparison would be
empirical evidence for this path under the existing objective, not a proof of the
flow-matching assumptions for this implementation.

This is a repository-specific conditional Gaussian path, not an implementation
of a published Schrödinger bridge or a demonstrated optimal transport solution.
Related primary work provides the general motivation: I²SB learns restoration
between informative degraded and clean endpoints using analytically sampleable
intermediate distributions ([ICML paper](https://proceedings.mlr.press/v202/liu23ai.html)).
DDBM similarly models bridges between paired endpoint distributions
([paper](https://arxiv.org/abs/2309.16948)). Conditional flow matching explains
how regressions on sample-conditioned paths yield their marginal vector fields
under its stated assumptions ([Theorems 1–3](https://arxiv.org/html/2210.02747v2)).
None of these papers validates this model's line quality or its few-step sampler.

## Executed diagnostic

The saved plan precedes execution. Use the existing 57 training and 57 local
probe pairs, strengths 0/.1/.5, whole and partial masks, three pyramid levels and
1/2/5 sampling steps. The partial mask cuts at rows103/columns171 to exercise
mixed coarse-resolution boundary cells. An oracle callback returns the known
masked target at every scale; it never loads learned weights.

All 2,052 actual runtime invocations (16,416 forward callbacks) match the derived
path within maximum absolute float32 error **7.15e-7**. Final targets and
protected pixels are exact. All 10,260 evaluated state differences match the
formula; clean-pair differences are zero. An independent float64 central
difference at progress .2/.5/.8 agrees with the derivative within **1.59e-9**.
The runtime source, input arrays, plan and as-run script hashes are retained.

For nonblank local probe examples at full resolution, whole masks and progress
.5, the mean absolute state shift on target-ink pixels, expressed in ink units,
is .1525/.1838 for tone at starting strength .1/.5, and .1818/.2190 for blur.
These describe a counterfactual perfect sampler, not measured learned-model
trajectories or quality gains. No model, production configuration or fresh
confirmation sample was changed; no visual acceptance judgment is made here.

Artifacts: `runs/oracle-bridge-diagnostic-20261008/`. The tracked helper
`scripts/diagnose_oracle_bridge.py --root NEW_ROOT` expects the saved plan in a
new root and refuses to overwrite existing results. Its numerical body matches
the executed script; only root selection and overwrite protection were added.

## Controlled next comparison, if needed

Compare legacy rough-only states and this path with identical rough/clean pairs,
initial strengths, progress draws, masks, Gaussian samples, model conditioning,
optimizer schedule and losses. Retain the original legacy trajectory as an
external reference: changing progress sampling also changes the distribution of
effective strengths, so a simple old-versus-new comparison would be confounded.
Keep the first-state samples represented and evaluate 1/2/5 inference steps
separately, with thin lines, cut/bridge errors and protected pixels included.
The one-step ideal MSE ambiguity at `u=1` is not removed by this change.

Any actual training experiment needs its own frozen plan, function-preserving
initialization checks, resume tests and JAX/ONNX/runtime agreement, followed by
native learned-output review. This diagnostic neither initiates that experiment
nor bypasses the campaign's independent-seed and sealed-confirmation gates.
