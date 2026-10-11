# BX: time-dependent feature gains throughout the endpoint CNN

BV/BW produce clean backgrounds but retain mixed shapes and output-frequency
bias. Their time conditioning adds a linear tau projection only at the first
encoder, head and detail branch. Test whether directly modulating every block
with nonlinear time features improves this controlled ambiguity problem.
This is a hypothesis, not a diagnosis that the old conditioning caused failure.

Retain both noisy/source bounded endpoint recipes from BV, including raw interval
endpoint error divided by max(u,1/32), top-stage loss and gradients through
clipped recursive inserts. Keep all inherited CNN weights and time projections.
Append a zero-initialized 1x1 projection in every encoder, decoder, head and
detail block. Its inputs are tau=1-u and sin/cos(pi*tau*[1,2,4,8]); its outputs
are per-channel gain and bias. After the first convolution and inherited bias,
apply h*(1+gain)+bias before ReLU. Resize time maps by nearest neighbor to each
block's resolution. No teacher or image features enter these projections.
Only the first convolution per block is modulated. No normalization layer.

The optional ModelConfig.progress_modulation='fourier_film' defaults to 'none'
and requires existing progress conditioning. Existing source-anchor restrictions
remain; custom BV wrappers define actual residual anchors. Zero projections
preserve the initial function (predeclared atol2e-6, rtol0; report actual maximum).
Existing parameters are bitwise identical; parameter count130628 ->134688 (+4060).
This is not a parameter-matched comparison; a gain would require later ablation
to distinguish nonlinear time features, multiplicative conditioning and size.

Both fresh arms train6000 updates with the exact BV step-indexed batch16 labels,
Gaussians, masks, strengths and tau, depth1/2/3 rotation, seed42 inherited weights.
Fresh Adam/EMA. Base LR5e-5*sqrt(16)=2e-4, warmup100, cosine6000 to1e-5.
Same top-stage interval endpoint objective; no extra loss, teacher, OT or
noise changes. Save and fully restore0/6/2000/6000 states,16 raw/EMA exports;
replay the first6 full updates and all6000 input journal records against BV.

Evaluate final raw6000 only, uniform Euler1/8/32, same12 groups and64 Gaussian
draws/group as BV. Reproduce both BV controls'72 physical/clipped prediction
blocks and metrics bitwise. Save144 blocks total; primary32 has36 guards/method,
RMS<=.15, at least80% within.20 and nearest-A fraction.1–.9, unchanged.
Inspect six native sheets (anchor x depth) with fixed first-four draws against
the matching BV control. Protected pixels must be exact and all outputs finite.
No checkpoint selection or switching primary grid after seeing results.

Tests cover old default behavior, zero-initialized function preservation,
time-dependent gain/bias before activation, valid configuration, modulation
gradients, full optimizer restore/next update and export/import. This toy uses
only two memorized teachers and repeated development noises. No new shapes,
independent seed or sealed confirmation; no candidate solely from toy success.
No production model, plugin, runtime-default or deployment changes.
