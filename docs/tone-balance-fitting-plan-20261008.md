# Controlled tone-weight fitting comparison

Declared before implementation tests or updates of these candidates. The broad
detail-head comparison is complete and rejected. The independent conditional-path
study is running; this comparison does not depend on or alter its results.

The [analytic diagnostic](tone-balance-diagnostic-20261008.md) shows that inverse
frequency tone weighting favors broad dark predictions under uncertain line
positions. It does not prove this causes learned halos. Compare only the pixel
term's teacher tone weights, preserving the other loss terms and architecture.

Use the completed local width16 model's architecture,130,605 parameters, original
rough guidance and ordinary rough training path, without progress conditioning.
Each candidate starts from the original100k EMA with zero added outputs, not
from the locally fitted endpoint. Reuse exactly the57 train and57 already-seen
probe pairs, seed2026100801, batches/masks/independent Gaussian draws, recursion1–6,
stage weights0, batch4, baseLR1e-4 scaled by sqrt(4), warmup50 and1,200-update cosine
clock to1e-5, EMA.999. Keep identity.1, edge.1, contrast.01 and noise floor.05.

The historical `warm_detail16` uses the existing normalized tone weights `w`.
New `warm_tone_sqrt` uses `sqrt(w)` normalized to mean one per image. New
`warm_tone_plain` uses exactly unit weights. Both configurations keep
`balance_tones=true` and select `tone_balance_power=.5/0`. Power1 preserves the
old numerical expression exactly. No regional, contrast or boundary coefficient
changes are included. The raw checkpoint is primary; EMA is diagnostic.

Before updates require core regression tests, exact historical six-update GPU
replay and full restore, six-depth candidate GPU updates/full restore, and exact
initial inference. Restore the first actual update. Both candidates complete
exactly1,200 updates; evaluate0/400/1200, with1/2/5 inference steps. The original
width16 reference is reevaluated and its one-step predictions must match its
historical arrays. Save whole/partial results at1,200 and protect zero-mask pixels.
Do not select another checkpoint or switch primary mode/iteration count.

Advancement is a local screen only. At raw one-step, require at least10% relative
halo reduction for **blur at strength.1 in both train and probe**, compared with
the historical width16 raw1,200 endpoint. Across each clean/tone/blur family,
split, strength.1/.5 and whole/partial mask, require F1 loss no greater than.005,
ink-deficit increase no greater than.005, and halo increase no greater than.01.
Record aggregate means, absolute fit failures and blank/repeat behavior as well.
Actual native output must support reduced gray fringe without missing thin lines
or new incorrect joins. If both pass, prefer lower mean blur/.1 halo across the
two splits; use higher mean F1 only to break a tie. Neither result constitutes
adoption or an independent generalization test.

Inspect twelve fixed native sheets with160px teacher-centered crops: train and
probe blur at.1/.5 raw (four), probe tone/.1 raw, probe clean/.1 raw, probe blur/.1
EMA, probe blur/.1 and.5 repeat1/2/5 (two), probe blur/.1 and.5 partial (two), and
blank whole/partial at both strengths with repeats. Use the first variant of
each of the same nine shapes, retaining fused parallels, gaps and thin strokes.
Main columns are target, rough, historical width16, square-root weights, unit
weights, all at one inference step. These float-input panels are not BGRA API
quality checks. Any future broad comparison, independent seed20261005 and sealed
new synthetic draws remain separate and necessary before adoption.
