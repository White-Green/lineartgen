# AX: Training-time coverage near both boundaries

Declared after AW. Conditional minibatch OT improves shape error but weak-noise
outputs still collapse to one mode and strong-noise outputs retain background
speckle. Test whether coverage near the beginning and end of the transport path
limits fitting. Change only the distribution of nonzero training times:
`tau_new = sin(pi*tau_old/2)^2`. The existing25% atom at time0 is preserved;
the continuous component becomes the arcsine distribution, emphasizing both
boundaries. The unweighted velocity objective now averages under this new time
distribution. This is intentional time weighting, not an unbiased estimator of
the previous uniform-time objective. No loss coefficient is tuned.

Reuse AW's exact inputs, labels, conditional assignment, Gaussian noise, masks,
strengths and teacher targets. Regenerate only bridge states and progress at
new times. Audit192 indexed minibatches. Unit tests verify exact preservation
of non-time data, the interpolation/derivative, protected pixels, distribution,
unchanged optimizer update on equal prepared data and full-state/next-update
restore. Keep the old generation outside the new JIT to preserve rounding.

Everything else unchanged: fresh common random130,628-parameter CNN,
6,000updates, seed2026101088, batch16, sqrt(batch) LR, warmup100/cosine,
recursion1–3/top-stage loss with smaller-stage gradients, same teacher-free
Euler sampler, final raw6000. No AW fitted weights. Retain states6/2000/6000,
6raw/EMA exports and all180 output blocks (144 reused controls plus36 new).
Evaluate the same12groups×64draws×1/8/32steps. Apply unchanged36 checks to
boundary32 (RMS<=.15, fraction within.20>=.8, mode-A .1–.9), finite physical
outputs and exact protected pixels. Inspect3 fixed sheets, first4 draws.

No toy-model deployment or claims of ordinary-lineart generalization. No
plugin or production recursion changes. A passing toy would require broader
synthetic-data training and independent confirmation before actualBGRA/ONNX
quality verification.
