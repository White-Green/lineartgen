# Full-resolution detail head: controlled fitting plan

Declared before any candidate trajectory update or candidate quality evaluation.
The completed fitting and floor probes leave a possible architectural restriction:
the last decoder has two channels, and the inherited output head has widths
3→3→3→1, even when encoder widths are doubled. Its 196 parameters may limit
precise local restoration. This is a hypothesis, not an established cause.

Add a parallel, three-convolution detail head before the existing residual output
formula/clipping. Its inputs are the noisy image, noise map, downsampled original
rough guide, first encoder's pre-pooling features, and final decoder features.
It never receives the clean target or a guide derived from a prediction. Hidden
width is fixed to either8 or16, in independent arms `warm_detail8` and
`warm_detail16`. The final kernel and bias start at zero; other new layers have
the normal random initialization. All inherited parameter tensors are exact.
This preserves the initial function while providing a trainable local path.

For the baseline widths [4,10,20,40], the branch input has9 channels and adds
1,313 or3,777 parameters (128,141 or130,605 total). Both arms use raw guidance,
the original residual scale (floor0), post-pool backbone, conventional clipped
MSE, identity .1, edge .1 and contrast .01. No new loss or corruption change is
combined with this comparison. The added full-resolution compute cost must be
measured if the branch becomes a useful candidate.

Copy the exact57 training pairs and57 already-observed local probe pairs from
`jax-learnability-20261008/runs/learnability-20261008`. Use its unchanged indexed
batches, masks, independent Gaussian noise, depth1–6 cycle, stage weighting0,
sqrt(batch) LR scaling, batch4, baseLR1e-4, warmup50, cosine1200 schedule, EMA.999,
seed2026100801 and common100k initialization. Exactly1,200 updates per arm;
evaluate at0/400/1200 with the inherited strengths .1/.5 and fixed partial masks.
Replay the historical `warm_hard` first6 GPU updates and verify the full state
bitwise before reusing that control's completed trajectory. Check candidate GPU
updates at all6 depths and exact checkpoint resume before the actual training.

Raw final output is primary, EMA diagnostic. To advance to a broader study an
arm must, against `warm_hard` at1,200: gain at least .03 mean blur F1 in BOTH
train and local probe; lose at most .01 clean F1 in each; and increase mean halo
by at most .01 in every clean/tone/blur × split. Retain all per-condition fit
failures. Inspect five fixed native sheets: raw train blur .5, raw probe blur .5,
raw probe tone .1, raw probe clean .1, EMA probe blur .5. Use first variants of
all9 geometric kinds, target-centered160px crops without resizing or contrast
adjustment, columns target/source/control/detail8/detail16.

Both numerical and visual support are required to advance. If both pass, prefer
the smaller arm if its train and probe blur F1 are within .01 of the wider arm;
otherwise compare their summed blur F1. No extension or hidden-width sweep.
This is a local diagnostic, not an adoption test or untouched confirmation.
Any broader candidate still needs the campaign's strict retention/topology gates,
actual synthetic BGRA quality review, a new training seed and sealed new draws.
Production weights and Krita interfaces remain unchanged.
