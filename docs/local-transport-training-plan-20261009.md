# Local opacity transport training screen

Freeze before updates. New hypothesis, not adoption of an earlier failed arm.
The smooth-absolute distribution term selected the sharp candidate in all six
shifted-line oracle controls at coefficient3 while preserving antialias values.
The controls do not establish actual conditional ambiguity, model performance,
or geometry preservation; shuffled pixels can share the same distribution.

One treatment repeats the exact4000-step procedural pixel control with only
`3 * local_opacity_transport(physical_preclip, composite_target, editable_mask,
tile=16, distance='smooth_absolute')` added. Keep opacity-deficit.3, sqrt tone
weights, identity.1, edge.1, contrast.01. No region/connection term is added.
Physical white may overshoot white, but gray/black quantiles are finite targets;
this deliberately penalizes overblack physical values even if clipped black is
correct. Correct finite AA targets give zero loss/gradient. No new model parameters.

Same130,605-parameter initial preclip-opacity.3 raw1200, seed2026100951, fresh
geometry and blur/opacity augmentation, all noise/mask draws, batch4,
sqrt(batch)LR peak2e-4, warmup100, cosine4000 to1e-5, EMA.999, recursive depths1–6,
stage weight0, and downsample-before-noise. Model input, Gaussian sampling,
source guide, inference format and runtime are unchanged.

Reuse the procedural study's pixel arm only after its fixed4000 endpoint is
complete. Bind its source, configs, final full-state restore and all baseline
snapshots before any treatment update. Reproduce its first six updates with exact
model/optimizer/EMA equality to the saved preflight. At0/1000/4000, recalculate
all64 diagnostic baseline predictions and require exact equality. Copy raw/EMA
baseline snapshots byte-for-byte and bind their hashes. This saves a duplicate
4000-update control run while retaining a matched data/optimizer comparison.

Primary raw4000; raw1000 intermediate only. Evaluate the same64 diagnostic
conditions and57 pairs per split at strengths.1/.5, whole/partial, depth3, one
iteration. Require mean blur halo reduction>=10% in each split; per family,
strength, mask and split require ink deficit increase<=.005 and F1 drop<=.005;
four-case path checks require cut/bridge increase<=.01 and width error increase
<=.10px. Inspect the fixed all-nine-shape native sheets for both strengths/splits,
blank and partial boundary sheets. A local numeric pass without satisfactory
native output does not advance. Any promising result still needs broader
synthetic displaced strokes, actual BGRA/repeated inference, independent
training seed and the unopened sealed confirmation before adoption.

Six-step GPU treatment preflight covers all depths and finite gradients;
save/restore every model/optimizer/EMA state exactly and replay the actual first
six updates. Keep full1000/4000 checkpoints and raw/EMA snapshots. No coefficient,
checkpoint or iteration switch based on the evaluation results.
