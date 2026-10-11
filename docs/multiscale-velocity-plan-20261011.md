# BN: direct velocity supervision at every recursive scale

BM's region weighting did not resolve fixed geometry. Return to BC point MSE
and isolate spatial supervision: change scale_loss_multiplier0 to1. At each
stage, use its own clean pyramid target minus its own independently noised
initial state, not a resized fine-scale velocity. Average per-active-pixel MSE
per image, then equally over stages. No interval relaxation, region weighting,
extra image loss, architecture, noise distribution, pairing or sampler change.
This is a weighting experiment, not correction of an established code bug.
No stop-gradient in the recursive insertions.

Same BC AZ interval36000 initialization, fresh Adam/EMA, seed2026101093,
fixed4 TRAIN plus4 unfitted PROBE512px antialiased geometry pairs. Model130628,
channels4/10/20/40, detail16, noisy features/base, source/progress conditioning.
6000updates, batch4, baseLR5e-5*sqrt4=1e-4 constant, cyclic depth1–6. Frozen BC
exact MSE6000 is the control. At depth1 the objectives agree. At larger depths,
normalization reduces the relative contribution of the largest stage; this is
part of the declared intervention. No search over stage weights.

The initial z=s+.25*strength*Gaussian is generated after resizing, independently
at each scale. Same25%tau0 and sin²(pi*U/2) other times. Coarse teachers are area
averaged antialias coverage, not binary masks. Prepared input arrays and RNG
draws remain identical to BC. Main-stage teacher/velocity remain bitwise exact.

States0/6/2000/6000;8 raw/EMA exports; all state and export equality checks;
first6 update replay. Before training replay8 BC control plus8 initial outputs.
At final evaluation replay all25 BC control blocks bitwise. Primary raw6000,
32-step sin² Euler grid, depth1/3/6,4 fresh noise draws,8 cases plus initial
depth3/draw0:200 rows,52 physical/clipped blocks,88 fit criteria per model,
64 F1/deficit retention checks. No checkpoint/draw/EMA selection or threshold
relaxation. Inspect all5 train/probe native/overview and partial-mask sheets.

Tests cover exact main-stage preparation, derivative targets at every scale,
resize-before-independent-noise distinction, original objective/gradient
equivalence at multiplier0 and depth1, nonzero gradient through all recursive
stages, protected pixels, and real CNN update/full optimizer+EMA restoration.
No adoption, production/plugin/default change. Any fitting improvement still
requires varied synthetic training, independent/sealed confirmation and actual
runtime parity before deployment.
