# CC: observed-source spatial weighting for sparse line endpoints

CA's modulated model still loses curves/junctions despite a clean distant
background. Test whether uniformly averaged endpoint errors devote too little
training weight to the small observed line region. This is an optimization
hypothesis, not an attribution of all failures or a new model-size conclusion.

Start both134688-parameter arms from the actual CA modulated6000 model/Adam/EMA
state. Continue1200 updates to7200, constant LR1e-5=5e-6*sqrt(4), batch4, no
warmup. Preserve BC/CA's four512px train pairs and four unfitted probes, depth1–6
rotation, step6000..7199 seed2026101093 data/noise/times,25% tau0 and sin² times.
Noise .25*t remains independent after resizing each scale. Both arms see exactly
the same prepared arrays, loss only at top stage with gradients through smaller
clipped endpoint inserts, and32 uniform Euler steps at inference.

`control`: CA's unmodified mean active interval endpoint squared error divided
by max(u,1/32)^2. `band`: half that mean plus half the source-band weighted mean
of the same per-pixel errors. Define source ink=clip((1-source)/2,0,1), and band
as its9x9 SAME max pool, intersected with active pixels. Normalize its weighted
error by band mass separately for each image. If the band is empty, use the
ordinary mean for both halves. Retain the ordinary loss with coefficient.5 so
unobserved background errors are still penalized. No added geometric target,
loss on generated samples, learned critic, thresholding or renderer change.

Weights depend only on the observed source and edit mask, not the teacher,
Gaussian sample, evolving state or time. For each fixed observed condition they
are positive constants multiplying individual pixel risks, so their unconstrained
pointwise minimizers are unchanged (also for the interval risk). This differs
from BM's teacher-region balancing; it is not a proof that a finite shared CNN
will improve. The small fit diagnostic determines whether to test broad data.

Freeze code/plan/data/checkpoint hashes before updates. Private parent checkpoint
copy bound to originals. Save/fully restore6000/6006/6600/7200 including Adam/EMA;
16 raw/EMA exports. Replay first6 updates exactly, covering all6 depths. Record
all1200 input hashes. CPU tests cover source-only weights, scaling and blank
fallback, protected pixels, equal pointwise-risk minimizers, actual independent
updates and exact next-update restore. No learning-rate arm or checkpoint search.

Evaluate initial CA modulated6000, control7200 and band7200 on the same depth1/3/6,
four draws and8 cases:288 rows,75 blocks including the partial depth6 batch.
Replay all25 initial CA physical/clipped blocks and metric rows bitwise. At
primary depth3, keep88 fitting guards/method unchanged; retain64 F1/deficit
checks each for control vs initial, band vs initial and band vs control (192).
Inspect fixed train/probe native/overview sheets with all four draws and the
partial native sheet with one fixed draw per four train cases. All values finite
and protected pixels exact. Final raw7200 only; no use of EMA for selection.

No changed probes, sealed confirmation, independent training seed, production
model, plugin, runtime defaults, export or deployment. CB's noise assignment
experiment proceeds separately; its results do not alter this plan.
