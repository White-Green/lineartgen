# Original rough-source conditioning: 10k result

Both candidates improve rough-stroke reconstruction substantially, but fail the
unchanged final retention criteria. No adoption, second fine-tuning seed or
fresh-condition evaluation is justified for these checkpoints.

The root is `runs/source-guide-20261008-continuation`; numerical source
`6194d8d61ccf0ff293c526e3fa93b27d94747a15` has the same Python as the 2k run
(`aba09b3`). All 194 source files and 14 settings were sealed. Both selected
model/optimizer/EMA states and counters were transferred exactly at 2k; the
original 10k cosine clock was retained. Actual GPU six-depth preflight and 243
tests were inherited with explicit provenance because no numerical code changed.

The first transfer attempt stopped before touching checkpoints because a
supplemental-validation index was absent. Its partial directory and log were
retained. Creating an index of the already sealed, actual pre-training audits
allowed an exact retry; no new validation result or earlier timestamp was
invented. Training then ran from 04:23 to 09:50 UTC on 2026-10-08; all numerical
evaluations finished around 09:55. The watcher startup race was avoided by waiting
for the controller status file.

| Method, EMA | Real strong-strokes halo reduction | Geometry strong-strokes halo reduction | Final failures (EMA / raw) |
|---|---:|---:|---:|
| source_ink_guard | 29.135% | 25.399% | 29 / 26 |
| source_control | 20.251% | 16.440% | 25 / 28 |

Comparisons are against the completed conventional control at the same step,
seed and clean/noise draws. Real inputs are real clean lineart synthetically
corrupted, not human rough sketches. Failure counts contain overlapping strata.
Raw halo gains are 29.435%/25.964% and 20.665%/17.140%, respectively.

For guide+ink guard, strong-blur geometry missing fraction improves by 12.867
percentage points, ink deficit by 5.978 pp and profile position MAE by 0.346 px.
Strong-strokes geometry missingness improves 3.743 pp, width MAE 0.465 px and
active F1 5.802 pp. Guided control shows similar smaller gains. Neither arm has
a final far-background opacity or bridge-regression failure.

Remaining failures concentrate in faint or mild corruption: real strong-tone
ink deficit increases 2.040 pp (guide+ink guard) / 1.845 pp (guided control), with
F1 dropping 1.484 / 1.227 pp. Geometry strong-tone cuts increase 2.031 / 2.448 pp.
There are also mild blur/tone ink and cut failures, low-strength strong-strokes
cuts, and a low-strength strong-tone width failure for guide+ink guard. Improved
averages on strong blur do not cancel these retention regressions.

Eight fixed native-pixel sheets were inspected. Real corrupted line contours are
visibly more coherent with fewer gray marks. Thin blurred shapes still disappear
or become fragmentary, parallel strokes merge, and intended gaps close. Repeated
inference can sharpen these wrong connections. Blank outputs are much cleaner:
maximum active mean opacity is .0001985/.0002411 for EMA and .0001698/.0002059
for raw, in table order. Full sheets, exact crops and hashes are retained.

Eight metric files are complete and all defined scalar metrics finite. Protected
regions remain exact. Actual 10k full states are finite, with raw/EMA inference
exports exactly matching restored parameters. `review.json` and
`completion-audit.json` record these checks and the rejection. Production weights
remain unchanged.

The next prospective test changes synthetic corruption coverage. Current training
uses clean or displaced/widened/blurred strokes; it does not explicitly allocate
examples to opacity-only and blur-only corruption. A fixed mixture of those
three families, keeping the same total augmentation probability, can test whether
the guide has learned to reject faint unthickened source lines. The architecture,
loss coefficients, Gaussian noise, recursion and optimization should remain
paired with these completed guided counterparts. This is a hypothesis, not an
established explanation or an adjustment to the quality gates.
