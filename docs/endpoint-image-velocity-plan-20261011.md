# BQ: supervise starting endpoint images as well as point velocities

Point/interval velocity, region reweighting, equal stage supervision, capacity
and initial noise amplitude have not yielded a geometry fitting pass. Test a
direct image constraint using the AD objective which helped one-pass fitting.

Return to BC's130628 model, AZ interval36000 initial weights, Gaussian coefficient
.25, same4 fixed512px TRAIN/4 unfitted PROBE pairs, point velocity MSE. Same fresh
Adam/EMA,6000updates, seed2026101093, batch4, baseLR5e-5*sqrt4=1e-4 constant,
cyclic recursion1–6, highest-stage loss with gradients through smaller insertions,
25%tau0 otherwise sin²(pi*U/2), same random draws/teachers and32-step sin² Euler.
Frozen BC MSE6000 control. No architecture, data, pairing or sampler changes.

Add0.1 * mean_over_batch[1(tau==0) * AD_image_loss(endpoint,target,strength)].
Endpoint is x+(1-tau)*predicted_velocity, physical before clipping. Compute each
image's AD loss separately, then batch mean with zero for non-start examples;
do not renormalize by the number of selected examples. Expected atom exposure
is25%; non-start samples have unchanged point MSE. Use AD's exact saved loss
configuration: interval L1, tone power.5, preclip opacity deficit.3, gap.1,
paper-band.1, identity.1, edge.1, active contrast.01, noise floor.05. Same
min_noise.001. This changes the fitting objective, not a guaranteed preservation
of conditional flow density. Pixel losses can encourage mode collapse for
ambiguous sources; this fixed-pair test does not establish mode coverage.

Before updates reproduce8 BC final and8 initial outputs bitwise. Same states
0/6/2000/6000,8 raw/EMA exports, first6 replay and exact full-state/export checks.
Tests require coefficient0 original loss/gradient equivalence, unchanged loss
and gradients without tau0, zero auxiliary gradients on non-start examples,
batch rather than selected-subset normalization, inward halo/deficit gradients,
actual CNN update and full optimizer/EMA restoration. Save individual loss terms.

Primary finalraw6000, no checkpoint/draw/EMA selection. Depth1/3/6,4 noise draws,
8 cases plus initialdepth3/draw0:200 rows,52 physical/clipped blocks,88 fit criteria
per model,64 retention checks. Replay all25 frozen BC final blocks bitwise and
review all5 native/overview/partial sheets. No thresholds are relaxed. Protected
floats exact and outputs finite. No deployment, plugin/default change or adoption.
Varied synthetic training, ambiguity/mode coverage, independent/sealed confirmation
and actual runtime parity remain necessary for a candidate.
