# Original-rough output anchor, fixed paired comparison

Hypothesis: the noisy residual skip x_t + sqrt(t) F asks a small network to
cancel sampled Gaussian pixels while also recovering fine rough strokes. A
source anchor rough + sqrt(t) F removes that mandatory cancellation from the
output skip. The network still sees the same noisy image, t, original rough
guide and recursive smaller prediction. No clean target enters inference.
This is a different parameterization, not proof that the input is uninformative.

Compare pixel (noisy anchor) with source (original-rough anchor), both with the
existing sqrt-tone pixel loss, preclip opacity deficit .3, edge .1, identity .1,
contrast .01. No region, connection or local transport term is added here.
130,605 parameters, same [4,10,20,40] guided UNet with width16 detail branch.
Keep residual sqrt(t) scale with floor0. Source anchor at t=0 uses the incoming
image to preserve protected pixels exactly, including repeated sampling.

Both inherit the same preclip-opacity.3/raw1200 parameter tensors except that
the last projections of the main and detail heads are reset to zero in BOTH
arms. Hidden pretrained features remain identical. Optimizer and EMA start
fresh. Thus the control initially outputs clipped noisy input, and the source
arm initially outputs rough source; initial functions deliberately differ.
The paired reset control is required: the earlier procedural pixel model with
its nonzero denoising head is not this experiment's comparison control.

Use exactly the procedural seed2026100951, four new512px geometry targets per
update, nine kinds with5% blank, width.6–6, supersampling4, and80% blur/opacity
augmentation (sigma0–2, opacity.25–1). Independent Gaussian noise is applied
AFTER pyramid downsampling. Same strength/mask/identity distribution, depth1–6,
stage multiplier0, batch4, LR1e-4*sqrt(4)=2e-4, warmup100, cosine4000 ending1e-5,
EMA.999. Arm update order rotates. Primary endpoint raw4000, intermediate1000;
no endpoint switch, postprocessing threshold, or evaluation seed tuning.

Reuse the fixed64 primary case/draw conditions and456 broad conditions per
model: both local development splits, clean/tone/blur, strengths.1/.5, whole/
partial, depth3 one iteration. The candidate needs >=10% blur-halo reduction on
each split, no stratum mean ink-deficit increase>.005 or F1 decrease>.005,
no primary case width-error increase>.10px or cut/bridge increase>.01.
Inspect all8 final sheets (nine geometry rows at both strengths/splits plus
blank and protected boundary). Blank success is not restoration success.
This is a development screen. Repeat-stroke rough, real-clean-derived synthetic
rough,1/2/5 iteration BGRA quality, independent training seed and untouched
sealed confirmation remain necessary after a promising screen.

Tests cover physical-anchor algebra, trainable/zero protected gradients,
shared parameter identity, source provenance, zero-head recursive sampling,
serialization and guided ONNX/Krita BGRA contract. Initial snapshot/evaluation
runs before both preflight and actual first6 updates to match execution order.
Require bitwise first6 replay and full checkpoint roundtrip. If repeat training
is not bitwise stable, stop and record it explicitly rather than relabeling it
exact; checkpoint serialization itself always requires exact identity. Freeze
code/config/input hashes before GPU updates; retain raw/EMA snapshots and full
state at1000/4000. No production model or plugin changes during this study.
