# Original rough-source conditioning: 2k screen (2026-10-08)

Continue `source_ink_guard` and `source_control` to 10k, in that order of the
predeclared summed-halo ranking. This is permission for a longer experiment,
not adoption. Actual native-pixel outputs still erase faint lines, merge some
parallel strokes and fill intentional gaps.

## Intervention and reproducibility

The fixed guide is the original rough source after synthetic sketch augmentation
and before Gaussian noise, provided at each recursion level and each denoising
iteration. It is never the clean target or an enlarged prediction. Two zero
initialized 3x3 projections add 63 parameters (126,765 → 126,828); existing weights
and the Krita BGRA API are preserved. See `source-guide-study-20261008.md`.

Root: `runs/source-guide-20261008-screen`. Numerical source:
`aba09b35a33300792a2f393b973a9c06d7f2398c` (194 files, 15 settings sealed).
All arms start at the same 100k EMA and use the same synthetic training draws,
fine-tuning seed 20261007, recursion 1–6, stage weighting, sqrt(batch) learning
rate, and original 10k cosine clock. Each has its own named source-disabled
counterpart: conventional control, hard ink guard, or interval ink guard.

Validation: 243 tests passed; 18 six-depth GPU preflight updates were finite with
exact full-state restoration. Four source-disabled baseline arms replayed 24
updates with bitwise-identical model, optimizer and EMA states. All 32 initial
guided/unguided forward pairs were bitwise identical; initial 1,562-case metrics
also match. Twelve training-only diagnostic batches were used without parameter
updates or coefficient tuning. Guide/no-teacher-leak and ONNX/BGRA tests cover
the new path, including fixed source guidance across repeated inference.

## Paired EMA screen versus conventional control

“Real” means real clean lineart synthetically corrupted, not human rough sketches.

| Method | Real strong-strokes halo reduction | Geometry strong-strokes halo reduction | 2k screen | Final diagnostics at 2k |
|---|---:|---:|---|---:|
| source_control | 8.520% | 7.539% | pass | 4 failures |
| source_ink_guard | 16.875% | 16.620% | pass | 5 failures |
| source_interval_ink_guard | 6.676% | 8.864% | pass | 52 failures |

These table comparisons are against conventional control. Only the corresponding
source-disabled arm isolates the effect of guidance for each loss configuration.
Failure counts include overlapping strata, not independent defects. All three
pass the unchanged permissive 2k screen; the fixed rule continues at most two.

For source_control, final diagnostics fail the 10% halo threshold and three ink
strata: real strong-tone partial (+0.547 pp), geometry mild-tone (+0.539 pp), and
its partial subset (+0.600 pp). For source_ink_guard, failures are real strong-tone
ink deficit (+0.714 pp; partial +0.970 pp), and geometry strong-tone strength 0.1
ink deficit (+0.856 pp), cuts (+1.042 pp), and active F1 drop (+0.511 pp).

Strong-blur geometry missing fractions improve by 2.025 pp for source_control and
1.969 pp for source_ink_guard. Their strong-tone missing fractions improve by
3.380 and 3.919 pp. Neither selected arm has a final-diagnostic far-background
opacity failure at this milestone. The interval-guided arm has 45 such failures
and ranks third; it is not continued.

## Visual and operational diagnostics

Eight native-pixel sheets were actually inspected: strong-blur geometry, both
strong-strokes strengths, raw strong blur, real EMA/raw details, repeated
one/five-step details, and partial blank five-step output. Their hashes and
observations are in `review.json`; source crops and render code are retained.
Thin strong-blur lines often disappear in all models, thick strokes retain gray
fringes, and five-step inference sometimes strengthens erroneous connections.
These common weaknesses are reasons to withhold adoption, not to call the models
visually clean based on an average metric.

Raw weights show greater halo reduction but worse retention: source_control fails
one 2k cut check, source_ink_guard fails four checks (ink/cuts), and the interval
arm fails one ink check. Final raw diagnostics have 31, 51 and 65 failures.
EMA was the primary selection mode before training; no raw substitution or
threshold adjustment was made. Longer training is not assumed to fix this.

Maximum active mean opacity across blank tests is 0.002270 / 0.001045 / 0.006787
for EMA and 0.000584 / 0.000382 / 0.001263 for raw, in table order. All eight
metric files contain finite defined scalars, protected-region errors are zero,
and exported raw/EMA arrays match the restored finite 2k checkpoints exactly.
The evaluation includes 1,562 development conditions, 55 fixed repeated-inference
panels, 40 blank conditions, and 24 supplemental geometry cases.

## Continuation decision

Restore the selected arms' complete 2k model/optimizer/EMA states into
`runs/source-guide-20261008-continuation`, preserving the 10k schedule. Evaluate
against completed conventional control and hard ink guard at 10k using the same
gates, EMA/raw outputs, partial masks, repeats and native panels. Only a passing
candidate proceeds to a second fine-tuning seed and sealed fresh synthetic
conditions. Those confirmations have not happened; production weights remain
the previously shipped model.
