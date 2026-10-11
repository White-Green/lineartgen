# BU: source-centered preconditioning of the velocity network

BS failed to turn faint noisy strokes into connected lines. This small mechanism
study tests an optimization issue before adding another image prior: the
unconditioned CNN must infer the time-dependent signal/noise cancellation from
raw image inputs and linear time biases. That is a hypothesis, not an established
explanation of BS. It is not solved by BD's projection or AT's endpoint loss.

EDM motivates separating network input/output scaling and loss weighting
([Karras et al., section5](https://papers.nips.cc/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf)).
The following coefficients are our derivation for this project's source-centered
linear bridge, not EDM's denoiser formula or a claim of published lineart results.
Let q=y-s, n=.25*strength*epsilon, u=1-tau, delta=x-s=tau*q+u*n,
v=q-n. Under an independent zero-mean variance model Var(q)=a² and Var(n)=b²,
set a=.5,b=.25*strength,D=tau²*a²+u²*b². Then

```
cin  = 1/sqrt(D)
cskip = (tau*a² - u*b²)/D
cout = a*b/sqrt(D)
v_theta = cskip*(x-s) + cout*F_theta(cin*(x-s), strength, source, remaining, insert)
```

For that variance model the normalized residual target has unit variance and
zero covariance with normalized delta. These assumptions do not assert that
lineart corrections are Gaussian, independent across pixels or zero-mean.
The coefficients are deterministic inference-time reparameterization, with
no teacher inputs and no division by remaining time. They stay finite at both
time endpoints for positive edit strength. Protected pixels use dummy finite
coefficients and are explicitly preserved; their velocity and loss are zero.

Keep the existing130628-parameter recursive CNN and read its raw head output F
before the old sampled-image residual anchor/scale. An optional accessor leaves
the default forward and serialized configuration unchanged. Input is normalized x-source;
source guidance, strength, progress and clipped physical endpoint insert remain
available. Smaller predictions feed only larger scales. No learned parameters
are added, no new Fourier embedding, normalization layer or architectural width.
Common initial CNN parameters are copied from AV's seed42 initial export, with
fresh Adam/EMA. Parameter equality does not mean equal initial velocity functions.
These exports require the custom preconditioned sampler; ordinary runtime
dispatch is deliberately not claimed to implement it. A preparation-only trial
using output_mode=direct was rejected by the existing progress-condition config
validator before any update. Keep that failure; do not relax the validator.

Train two arms: `physical` minimizes the original active physical velocity MSE;
`normalized` minimizes error²/cout². The latter changes time/strength weighting,
not teacher-dependent tone weights. Both use exactly AV's64px two-teacher common
source,16-item batches, masks, strengths.1/.5, Gaussian keys and time draws
(25%tau0,otherwise uniform),6000 updates,baseLR5e-5*sqrt16=2e-4,warm100/cosine
to1e-5,depth1–3. Retain AV's frozen point-velocity6000 control and reproduce all
36 physical/clipped output blocks bitwise at inference1/8/32 uniform Euler steps.
Using this original sampler isolates preconditioning from BA's later grid change.
The primary contrast physical versus normalized has equal initial functions.
Comparison with AV changes input/output preconditioning together; no attribution
to one coefficient alone. No interval supervision or image auxiliaries.

Audit algebra against direct covariance/variance calculations, protected pixels,
real gradients and next-update/full-state restore. Replay the first6 updates
from fresh states. Save model/Adam/EMA at0/6/2000/6000 and all16 raw/EMA exports.
Final raw6000 only; no checkpoint selection or new training seed search.

Evaluate12 groups (depth1/2/3,whole/right-half,strength.1/.5),64 Gaussian samples
each using the old EVAL_SEED. At steps1/8/32 save all108 candidate/control blocks.
At primary32, retain36 guards per arm: nearest-endpoint normalized RMS<=.15,
at least80% of draws within.20, and A fraction within.1–.9. Review first four
32-step outputs for every model/group in all three native sheets. Report both
modes and background artifacts, finite physical outputs and protected identity.
Passing would establish only this two-shape mechanism, not broad lineart quality
or independent confirmation. No production/plugin/runtime-default changes.
