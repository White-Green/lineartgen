# Shared fitting with local connection queries

Frozen before updates. Synthetic data only; this is a mechanism screen, not
adoption or the sealed confirmation. Two independent 130,605-parameter models
start at the same preclip-opacity .3 raw1200 checkpoint. Each model sees all four
previous TRAIN cases (straight38, thin39, corner41, gap45) in every batch. The
corresponding four existing PROBE cases are evaluation only. This changes the
previous specialist setup; only the paired pixel arm is the causal control.

Both use batch4, base LR1e-4 scaled by sqrt(4) to2e-4, warmup50,
cosine2000 to1e-5, EMA.999; primary raw final2000, intermediate600 diagnostic.
Fresh Gaussian draws use the unchanged2026100923/per-update/per-pair schedule,
after downsampling. Strength.5 whole image; recursive depths cycle1–6 and
scale_loss_multiplier remains0. Inference API and architecture are unchanged.
The clipped pixel loss, sqrt tone balance, preclip opacity-deficit.3,
identity.1, edge.1 and contrast.01 remain identical.

The treatment adds0.1 times the equally weighted mean of connected-query deficit
and disconnected-query capacity squared. Eight positive queries span20px of
known strokes; the one gap query has anchors8px inside its two caps. Each query
uses a64px patch, 3x3 anchors and eight-neighbor vertex opacity capacity. Maximin
search reaches a fixed point or the full V-1 simple-path bound. A selected
winning-path bottleneck receives the gradient; ties select one path. The teacher
supplies its actual antialiased capacity. Positive capacity is a lower bound;
disconnected teacher capacity is zero. No teacher geometry enters inference.
Connected and disconnected classes are normalized separately, preventing the
single gap query from being diluted among all512x512 pixels.

Known limits: local patches exclude external detours. A scalar connection can be
fixed with a tiny cut while most gray gap remains; full-gap opacity and actual
native crops must therefore improve. Strong pixel/ink supervision remains needed
to avoid cutting a correct stroke instead. Width is not supervised by this term.

Primary evaluation: raw depth3, eight existing diagnostic noise draws per case
per split (32 TRAIN+32 PROBE conditions per arm), all locked before training.
Fit screen: F1>=.95, ink deficit<=.05, halo<=.02. To advance this term, require
at least25% reduction of mean gap-middle opacity on both TRAIN and PROBE versus
paired control, no more than.005 mean F1 drop or.005 ink-deficit increase per
case/split, width-error increase<=.10px and no increase in cut rate. Inspect
replicate0 and worst-treatment-F1 native160px crops on both splits. Passing this
screen would require broader noise/mask/shape/data evaluation and independent
training seed; it does not permit adoption. Failing it triggers a revised
hypothesis, not checkpoint/threshold cherry-picking.

Before updates: independent CPU widest-path oracle and gradient tests, teacher
identity/antialiasing, cuts, faint bridges, detours, ties, empty anchors and
protected-gradient tests; source/config/input hashes and source archive. GPU
six-update preflight covers all depths, finite gradients and exact full-state
save/restore. Actual first six updates must exactly replay preflight. Final
model/optimizer/EMA restore must be exact. Both arms have identical initial
inference; the code must audit local teacher-query validity.
