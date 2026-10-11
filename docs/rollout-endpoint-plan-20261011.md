# CG: train on the model's own intermediate image

CD/CE/CF did not repair missing strokes by changing inference count, resolution
order or time weights. Compare teacher-created and model-created midpoint inputs
with identical two-pass training cost. This tests exposure to the model's own
errors, not another scalar loss coefficient or an established causal diagnosis.

Both arms start from the actual CA modulated6000 model/Adam/EMA; continue1200
updates at constant LR5e-6*sqrt4=1e-5, batch4, depth1..6 rotation and the same
step-indexed CC source/target/Gaussian initializations. The old random time draws
are discarded, not used as supervision times. At u=1, compute bounded endpoints
D0 at every resolution from original noisy source states z. The second pass
uses u=.5 and either xmid=.5*z+.5*clean (`teacher`) or xmid=.5*z+.5*D0 (`rollout`).
The latter is exactly the first update of uniform two-step inference at every
resolution. It receives no teacher input. Gradients propagate through that
midpoint and all clipped smaller inserts; no stop-gradient. Protected pixels
remain original source. Recompute smaller-to-larger endpoints on the second
pass, without any larger output feeding smaller models.

Loss = .5*(intervalMSE(rawD0,y) + 4*intervalMSE(rawDmid,y)), averaged over active
pixels per image at the largest stage. This retains the original endpoint
risk at its two chosen times, u=1/.5. Both arms have the same times, two model
passes, teacher target and weighting; the midpoint generation and its derivative
are the intervention. A positive result would not distinguish input-distribution
change from differentiating through the first prediction. This is a fixed-pair
fitting diagnostic, not a production change to multi-stage supervision. Training
on a single teacher per rough image is not a proof of valid multimodal transport;
the ambiguous two-mode problem would require a separate coupling check.

No model parameters, sampler algebra, source/noise definition or masks change.
Noise remains independent after each source resize. Save/fully restore states
6000/6006/6600/7200 and16 raw/EMA exports, replay six updates. Tests check the
midpoint against a manual Euler step, teacher independence of rollout states,
nonzero gradients through the first prediction, teacher-mode isolation from
that derivative, protected pixels, actual full-state restore and next updates.

Evaluate final raw7200 only and frozen CA initial6000, same eight AD512px pairs,
depth1/3/6, four old Gaussian draws, partial depth6. Primary2 inference steps,
secondary32 (report both, no swapping primary). Each has288 rows,75 blocks,
264 fitting and192 retention checks: teacher/initial,rollout/initial,rollout/teacher.
At32 reproduce all25 original CA physical/clipped blocks and96 metric rows
bitwise. Primary2 has no historical reference, so report no fabricated replay.
Review five sheets per count,10 total: train/probe native/overview include four
draws; partial one draw per four train cases. All finite/protected identity.

No new shapes, seed search, checkpoint selection, sealed confirmation,
production/plugin/runtime-default/export/deployment changes. Passing fixed
pairs alone cannot establish practical quality.
