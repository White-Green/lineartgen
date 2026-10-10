# BC:512px antialiased geometry velocity fitting

Declared after BB's fresh-noise toy pass. The two-template result does not test
curves, antialias coverage, corners, crossings, taper or full recursion. Use the
established AD four-pair fitting set: TRAIN999/1095/1287/1383 (curve/corner/
parallel/gap), PROBE807/903/1191/1479 (straight/thin/junction/taper). Copy the
exact512px clean, synthetic strong-blur rough, strength.5, probe metadata and
predeclared160px crops. The target identity oracle must pass. Synthetic only.

Both arms start from identical AZ interval36,000 parameters, with fresh Adam
and EMA equal to the initial model. This tests adaptation from a learned toy
prior, not from-scratch superiority. Model remains130,628 parameters, channels
4/10/20/40, detail16, with source/progress conditioning and noisy features/base.
Train6,000 updates, batch4 (all four pairs per batch), seed2026101093, base5e-5
scaled by sqrt(4) to constant1e-4, warmup0. Cycle depths1–6, top-stage loss only
with gradients through smaller predicted endpoints. Preserve noise generation
after resizing: z=s+.25*strength*epsilon at each scale.

For each update draw fresh Gaussian noise and time:25% exactly tau0, otherwise
sin²(pi*U/2). Use the same prepared batch for both arms. Teacher/source pairings
remain fixed: these observations are different, so the two-identical-source
toy's target permutation is inapplicable. Point and interval velocity objectives
are imported unchanged from AZ; only the objective differs between arms. The
interval loss does not alter antialias targets. This custom objective bypasses
generic image_loss, and the custom32-step boundary-grid sampler is unchanged.

Retain states0/6/2000/6000 and16 raw/EMA exports; replay the first6 updates and
all depths exactly. Final raw6000 fixed. Evaluate both models on all8 cases at
depths1/3/6 and4 fixed fresh noise draws (seed2026102093). Primary depth3 applies
AD's22 fitting criteria separately to each draw:88 checks per arm, without
loosening thresholds. Depths1/6 are recorded diagnostics, not a substitute for
a primary failure. Require interval's F1 drop<=.005 and ink-deficit increase
<=.01 versus matched MSE for each of8 cases and4 draws:64 retention checks.
Unfitted probes are never added to training or used for checkpoint selection.

Keep200 metric rows:192 final evaluations plus8 initial-weight depth3/draw0
references. Save52 physical/clipped prediction blocks (50 full-mask,2 partial).
Also run both final models on all4 training sources with the left half protected
at depth6,32 steps; require exact source values there. This is an inference
invariant check, not evidence of partial-mask training or boundary quality.
Inspect5 fixed sheets: train/probe native crops and full-image overviews with
all4 primary draws per arm, plus a native partial-input sheet. Do not select
only favorable draws. All outputs must be finite.

A successful fixed-pair fit is not broad generalization. More varied synthetic
training, held-out teacher identities, independent training seed, sealed conditions
and actualBGRA/ONNX quality remain necessary. No Krita/plugin/default change.
