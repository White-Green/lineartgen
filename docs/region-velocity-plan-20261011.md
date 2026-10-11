# BM: region-balanced exact velocity targets for antialiased geometry

BC's point-velocity output-gradient audit assigned98.1% absolute output-gradient
mass to paper. This is not a parameter-gradient norm or causal proof. Interval
velocity reduced that imbalance but still failed geometry. Test exact MSE with
explicit region normalization, avoiding one-sided white/black velocity freedom.

Return to BC's exact initial AZ interval36000 weights and original6000-update
geometry protocol, with fresh Adam/EMA. Freeze BC point MSE6000 as control.
Only the objective changes: partition active clean pixels into paper(y>=1),
antialias(-1<y<1), opaque ink(y<=-1). Compute velocity squared error averaged
within each nonempty region per image, then equally average the present regions
and batch. Empty regions contribute neither error nor denominator. All-protected
images give zero loss and gradients. No clipping of targets/residuals, additional
image loss, tone weighting, dynamic coefficient, or sampler change. This is not
assumed to preserve an unweighted conditional flow density; quality is measured.

Use the same4 fixed TRAIN and4 unfitted PROBE512px AD pairs. Model130628,
channels4/10/20/40, detail16, original noisy features/base and source/progress
conditioning. Seed2026101093, batch4, baseLR5e-5*sqrt4=constant1e-4, warm0,
6000updates, cyclic recursion1–6, highest-stage velocity loss with gradients
through smaller clipped endpoint insertions. Same z=s+.25*strength*Gaussian
after resizing,25%tau0 otherwise sin²(pi*U/2), fixed pairing, no OT permutation.

Predeclare states0/6/2000/6000,8 raw/EMA exports, six-update exact replay, all
state/export checks. Before training reproduce8 frozen BC MSE primary draw0
outputs and8 initial outputs bitwise. Evaluate all25 frozen final blocks again
at the end (all depths/draws and partial mask). Primary finalraw6000, no selection.

Before and after training audit exact-MSE versus region-MSE absolute output
gradient mass on16 predetermined training-path batches at depth3, keys0–15.
Report per-region pixel counts, loss and output-gradient mass; do not describe
it as parameter-gradient norm or infer causality from that decomposition alone.
These diagnostic paths are training examples, not generalization trials.

Same200 rows/52 physical+clipped blocks and5 sheets as BC/BF: two final models,
8 cases, depths1/3/6,4 fixed fresh noise draws, plus8 initial depth3/draw0.
32-step sin² Euler grid unchanged. Depth3 has88 fit conditions per model,
64 F1/deficit retention checks against BC MSE. Inspect all4 draws in train/probe
native and overview sheets, plus protected-left partial native at depth6.
Prepared-float source preservation must be exact. No threshold relaxation.

Tests cover counts, equal region contributions under equal residual, absent
classes/all-protected cases, finite inward gradients for both velocity signs,
actual CNN update and full optimizer/EMA restore. No production/plugin/default
changes, no adoption. A pass would still need varied synthetic training,
independent/sealed confirmation and actual runtime parity.
