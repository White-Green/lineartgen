# CI: a small full-resolution CNN inside the recursive pyramid

Compare fresh seed42 models: a 134688-parameter Fourier-FiLM U-Net and a
15809-parameter local CNN (width16, three residual blocks, two3x3 convolutions
per block,3x3 stem,1x1 head, SiLU). Local features never pool or stride. Both
retain the external small-to-large image recursion and independent Gaussian
noise added after each source reduction. Local input concatenates noisy image,
source, noise strength, and nearest-upsampled coarse prediction; white image
values are centered to zero. Fourier-FiLM acts on stem and each block's first
convolution, with initially zero modulation projections. No normalization.

Both models start with all final image heads zero (including U-Net detail head),
so their initial raw heads and anchored outputs are exactly equal. Random
hidden weights need not match. Fresh Adam and EMA; no pretrained weights in
either trained arm. Both use CH's two-step rollout objective: half the initial
endpoint interval MSE plus twice the midpoint interval MSE, plus.1 times AD's
full image loss on the second raw endpoint. Gradients flow through the first
predicted midpoint and smaller-resolution outputs. No teacher coarse input.
Top-stage fitting loss remains a temporary diagnostic; production per-stage
loss is unchanged. Local width nesting is not implemented in this pilot.

Train1200 updates, batch4, constant LR5e-5*sqrt4=1e-4, no warmup, depths1..6 in
rotation. Same four AD synthetic rough/clean512px pairs, step-indexed noise
seed2026101093 and steps0..1199 for both arms. Retain unused old random-time draws
in the input preparation but use only u=1/.5 in this objective. Save full states
0/6/400/1200 and16 raw/EMA exports; six-update replay, exact restore, next-update
unit test, parameter count, zero-initial-function equality, gradient flow,
protected identity, model serialization and local-stage translation checks.
Record per-update wall times (including JIT; summarize warmed steps separately).
Parameter count does not establish faster execution.

Final raw1200 only. CH image8400 is a frozen quality reference with a different
training history, not a matched architecture control. Primary comparison is
local vs fresh U-Net. Bundled changes include pooling, channels, activation,
residual blocks, input centering and capacity; no claim of isolating pooling.
Primary2 and secondary32 inference steps; each count includes288 metric rows,
75 output blocks,264 fit checks,192 F1/deficit retention checks. Reproduce all50
CH physical/clipped blocks and192 metric rows bitwise. Same thresholds, AD
crops, fixed four Gaussian draws, depth1/3/6, all finite and protected exact.
Review all10 sheets: whole train/probe native/overview with four draws each,
and partial native with one draw per training case. Explicitly inspect gray
backgrounds and echoes, which F1/deficit guards alone do not cover adequately.

This is a fixed-pair fitting pilot, not independent confirmation. No new
training seed, sealed holdout, actual Krita/BGRA, ONNX support, plugin change,
production default, deployment or adoption. Any continuation or numerical
change after results requires a new frozen plan/worktree.
