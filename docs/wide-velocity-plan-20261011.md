# BO: capacity of the noisy-input velocity CNN

BM region balancing and BN equal spatial-scale supervision did not resolve
fixed geometry. AR previously tested width for the source-only one-pass model,
not the noisy-feature velocity model. Test this capacity question separately.

Return to the exact BC point-MSE protocol (scale_loss_multiplier0). Double
channels4/10/20/40 to8/20/40/80, detail16 to32, final decoder2 to4, hidden head3
to6. Parameters130628 to517911. All other model fields and inference are fixed.
Embed AZ interval36000 weights with the existing widen_unet, seed2026101113.
Inherited output channels initially ignore extra input channels; new channels
keep random kernels so new paths can learn. Fresh Adam/EMA, seed2026101093,
batch4, baseLR5e-5*sqrt4=1e-4 constant,6000updates, recursion1–6, same4 TRAIN
and4 unfitted PROBE512px antialiased geometry pairs. Frozen BC MSE6000 control.

Same point velocity target, highest-stage supervision and gradients through all
recursive insertions, Gaussian after resizing, source/progress conditioning,
25%tau0 otherwise sin²(pi*U/2). No region/interval weighting, auxiliary image
loss, target permutation, new data, or inference-grid change. Same updates/data
but greater compute; not a matched-FLOP comparison.

Before updates, replay8 narrow initial and8 final control32-step outputs bitwise.
Check expanded initial vs narrow at all6 depths,8 cases, whole/partial96 examples
for both velocities and recursively inserted endpoints: absolute tolerance1e-5.
Convolution order changes, so widening agreement is not bitwise. Initial primary
32-step integration agreement must be within2e-4; record actual errors. These
tolerances are fixed before training, not relaxed after seeing results.

Test learned nonzero guide/progress projections through widening, recursive
velocity/endpoints/protection and nonzero gradients into newly added paths.
Verify actual update and full Adam/EMA restore. Retain states0/6/2000/6000,
8 raw/EMA exports, first6 replay and all state/export equality checks.

Primary finalraw6000. Same32-step sin² Euler, depths1/3/6 and4 fixed fresh noise
draws on8 cases plus initialdepth3/draw0:200 rows,52 prediction blocks,
88 fit criteria per model and64 retention checks vsBC. Replay all25 frozen BC
blocks bitwise. All5 TRAIN/PROBE native/overview and partial native sheets are
reviewed. No checkpoint/draw/EMA selection or threshold relaxation. Protected
float pixels exact, finite outputs required.

No deployment or production/plugin/default changes. A fitting pass still needs
varied synthetic training, independent/sealed confirmation and actual runtime
parity. One width comparison cannot identify the minimum adequate model size.
