# AT: can the actual recursive CNN learn a two-mode conditional path?

Declared after AS failed its fixed fitting gate. This is a small mechanism study,
not a broad restoration candidate or a reason to change the229 adoption checks.
AQ's analytic oracle knows both clean endpoints; AT instead learns from random
paired examples and samples without either teacher being supplied to the CNN.
The older conditional-path study used a different square-root path, teacher
dependent tone weighting and model initialization, and failed badly. AT is a
new linear-path endpoint-regression diagnostic, not a reproduction of that study.

Use AQ's exact64px ambiguity: A is a10px solid vertical bar, B two3px bars with
a4px gap. Radius2 dilation makes their rough supports identical; sigma1.5 blur
and opacity.3 produce the common informative source. Both teachers are equally
probable. This is deliberately non-injective and cannot reveal the hidden user's
intended teacher. No frequency claim about real sketches follows.

Use existing130,628-parameter UNet with channels4/10/20/40, detail16, post-pool
skips, noisy features, source guidance, explicit progress and sampled-image
residual anchor. Fresh common random weights (seed42), fresh AdamW/EMA. Separate
`direct` and `bridge` arms share every teacher, mask, strength and Gaussian draw.
16 examples/update, strengths .1/.5 and whole/right-half edits. Direct trains
only z=source+.25*strength*epsilon at time0; bridge trains
x_tau=(1-tau)z+tau*teacher, with25% time0, otherwise uniform[0,1). Noise is drawn
after each pyramid reduction. Both regress the clean endpoint using physical
preclip MSE averaged over active pixels per image, no teacher-dependent tone,
opacity or shape auxiliaries. This has the conditional-mean optimum at each
time; it is time-weighted velocity regression, not the old balanced bridge loss.

Exactly6,000 updates/arm; seed2026101088, batch16, baseLR5e-5 times sqrt(16)=2e-4,
warmup100/cosine to1e-5, EMA.999. Cycle recursive depths1–3 because64px with this
UNet reaches16px at depth3. This toy size limit is explicit: production512px
depth1–6 and its loss conventions are not changed or validated by this study.
Only top-stage loss with gradients through smaller predictions. Retain full
states6/2000/6000 and all12 raw/EMA exports; fixed final raw6000, no model selection.

Sampler: independent initial z at each pyramid scale; evaluate the learned
endpoint at every scale recursively and Euler-update
x += (endpoint-x)/(1-tau)/N. Assign source exactly at every protected pixel and
step. Internal time is distinct from edit strength. Clip only final display
outputs to[-1,1]; report unclipped excursions. Inference sees source, x, strength
and time, never teacher A/B or the oracle. Candidate32 evaluations versus direct1
is a deliberate additional inference cost, not a matched-latency comparison.

Final evaluation seed2026102088,64 independent Gaussian draws/group for12 groups
(whole/partial × strength.1/.5 × depth1/2/3), shared across all methods and step
counts1/8/32. Save all72 output blocks and metrics. Primary bridge32 checks:
all finite, protected exact, and in each of12 groups mean nearest-endpoint RMS
(normalized by half-endpoint-difference norm) <= .15, at least80% draws within.20,
and each endpoint selected by at least10% and at most90% of draws. Also report
teacher-averaged MSE, which can prefer a gray conditional mean to crisp sampling.
Direct and bridge1/8 are descriptive comparisons, not fallback selection.
Review three fixed native sheets (one/depth), first four samples of each group,
with teacher A/B, rough, direct1, bridge1 and bridge32. No cherry-picking.

Tests before training cover initial path equality, downsample-before-noise,
independent Gaussian draws, teacher-free sampler interface, oracle finite-step
behavior, protected pixels, actual gradients and full-state/next-update restore.
Check identical initial parameters and no cross-arm parameter sharing.
Even a complete pass proves only learnability on these two memorized shapes;
it requires a separate broad protocol to test new geometry, real corpus, actual
BGRA/ONNX, independent seeds and runtime cost. No toy weight is deployable, no
plugin/runtime/default changes, no sealed confirmation or final-source data.
