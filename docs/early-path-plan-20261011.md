# BZ: increase early continuous-time coverage after BX6000

BX noisy passes all24 shape criteria but violates five weak-noise mode-frequency
guards. BY finds a positive initial bias and delayed response near tau .002–.01.
Test extra training coverage there, retaining the existing probability of exact
tau0. This targets the poorly fitted part of the path, not the teacher posterior
itself; no oracle targets or model-specific output corrections are introduced.

Restore BX noisy's actual6000 model, Adam moments/count and EMA, then copy all
state into control/early arms. Continue each to12000 with batch16, constant
LR1e-5 (base2.5e-6*sqrt(16)), no warmup. Control uses the original continuation
stream. Early shares every label, teacher, Gaussian, strength, mask and depth,
but independently selects half the nonzero tau values and replaces tau with
.05*tau^2. Keep the other times and all exact-zero examples bitwise unchanged.
Thus the time distribution is25% zero,37.5% original continuous uniform and
37.5% on (0,.05) with a quadratic transform, in expectation. This changes the
loss weighting over time without importance correction; not an unbiased estimate
of the old time-averaged objective. Do not describe it as changing only variance.

Selection key is the original seed2026101088 folded with global step and
0x4541524C. Continue global steps6000..11999, depth=step%3+1. Reconstruct each
selected intermediate state from the same downsample-then-noise initialization
and same resized teacher. Keep protected source exact. Same134688-parameter
Fourier-modulated CNN, noisy anchor, raw interval endpoint loss/max(u,1/32)^2,
top-stage gradient through clipped recursive inserts. No other loss or noise
changes. Final raw12000 only; no checkpoint selection.

Copy the parent6000 checkpoint privately for restore; bind all files. Verify
full parent state equality, two independent copies, constant LR and the first6
original control updates. Save/fully restore6000/6006/9000/12000,16 raw/EMA
exports. Journal all6000 batches: hashes, label/depth counts, identical tau0
count, selected count and counts in (0,.01). Compare control batches with the
original function. Tests cover identity when unselected, same noise/targets,
protected pixels, exact-zero preservation, early-time bounds, independent arm
updates, full optimizer restore and matching next update.

Evaluate initial BX6000 and both final arms with unchanged uniform Euler1/8/32,
12 conditions,64 Gaussian draws/group. Reproduce all36 BX noisy physical/clipped
blocks and metrics bitwise; save108 blocks. Primary32 has36 guards/model:
nearest-mode RMS<=.15, >=80% within.20, nearest-A fraction.1–.9. No relaxed
criteria. Inspect all three native sheets, fixed first-four draws per group.
Finite/protected checks remain mandatory. This is repeated development on two
memorized shapes; no independent seed, new teachers or sealed confirmation.
No production model, plugin, runtime default or deployment changes.
