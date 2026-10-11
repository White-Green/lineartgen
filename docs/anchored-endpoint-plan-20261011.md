# BV: noisy versus source anchor in a bounded endpoint flow

BU's global variance preconditioning amplified background artifacts. This
separate diagnostic tests an existing one-pass design idea in endpoint flow:
predict the clean endpoint as an adjustment to the informative original source,
instead of forcing the network to cancel a direct skip of every noisy pixel.
It is a hypothesis motivated by visible noise, not a demonstrated explanation.

Two fresh arms share AV's exact CNN parameters and data. Both CNNs see evolving
x, edit strength, original source s, time u, and the smaller predicted endpoint.
Read raw head F via the already-tested optional accessor. `noisy` forms
D_raw=x+sqrt(strength)*F; `source` forms D_raw=s+sqrt(strength)*F. At each scale
the clipped endpoint enters only larger scales. This changes the anchor only
between the two new arms. Their initial parameters match; functions do not.
No source-only feature substitution and no clean teacher at inference.

Use interval endpoint MSE (one-sided residual at exact white/black, point error
at antialias values), divided by max(u,1/32)^2, averaged over active pixels per
image. This is endpoint interval supervision with capped velocity weighting;
it is not equivalent to unweighted endpoint MSE or an exact probabilistic
flow objective after clipping. No teacher tone balance, shape auxiliaries or
global variance preconditioning. Capping prevents arbitrarily large late-time
weights; the cap matches the smallest remaining time in the32-step sampler.

At inference v=(clip(D_raw,-1,1)-x)/max(u,1/32); reuse AV's uniform Euler sampler
at1/8/32 steps. At these step counts u never falls below1/32, so the inference
denominator is exact. Do not silently use this recipe at higher step counts.
Protected pixels retain s at every stage and integration update. The normal
serialized model configuration remains AV's residual/noisy configuration;
the source anchor is explicit in this custom training/sampling wrapper. These
exports cannot be deployed with ordinary runtime dispatch. Do not relax the
existing config validator that forbids source anchor plus progress there.

130628 parameters, no added weights or layers. Fresh Adam/EMA,6000 updates per
arm; original AV batch16,seed2026101088,64px common rough source for the two
binary bar teachers,depth1–3,25%tau0 otherwise uniform, strengths.1/.5 and
whole/right-half masks. Gaussian is generated independently after resizing.
BaseLR5e-5*sqrt16=2e-4,warm100/cosine to1e-5. Retain the frozen AV6000 raw control
and replay its36 physical/clipped output blocks and metrics. AV comparison
changes loss, sampler endpoint clipping and anchor recipe; it is contextual,
not a pure anchor ablation. The controlled contrast is noisy versus source.

Tests verify default noisy endpoints against original AT at all three depths,
the exact depth1 anchor shift, bounded teacher-free sampling/protection, interval
gradients including genuine AA, capped finite time weights, gradients through
recursion, real updates and full-state/next-update restore. Before GPU training,
check unchanged initial parameters and noisy endpoint replay. First6 updates
replay independently. Save full states0/6/2000/6000 and16 raw/EMA exports.
The6000-row sampling hash journal must equal BU exactly. Final raw6000 only.

Reuse12 toy groups×64 draws,steps1/8/32,108 output blocks across AV/noisy/source.
At primary32 retain36 guards per method: normalized nearest-teacher RMS<=.15,
at least80% within.20, nearest-A fraction.1–.9. Review all three native sheets
with fixed first-four outputs/group/method. Neither label frequencies alone nor
white background alone establishes success. No changed thresholds, new shape
confirmation, sealed conditions, production/plugin/runtime/default changes.
