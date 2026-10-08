# Ink-centered guide results

All four trajectories reached 2,000 updates. All ten predeclared EMA endpoint/blend
models fail the unchanged strict final gates. **None is adopted or advanced to
independent confirmation.** The fresh synthetic draws remain unopened.

Halo reductions below are against the matched conventional 2k control; negative
means worse. Counts include overlapping groups, not independent defects.

| Guide | Mixed parameter fraction | Real halo reduction | Geometry halo reduction | Strict failures |
| --- | ---: | ---: | ---: | ---: |
| ink | 0 | 13.68% | 11.83% | 10 |
| ink | 1/16 | 11.10% | 9.60% | 5 |
| ink | 1/8 | 8.47% | 7.34% | 5 |
| ink | 1/4 | 3.06% | 2.71% | 12 |
| ink | 1 | −33.20% | −26.56% | 67 |
| sqrt ink | 0 | 13.17% | 9.37% | 16 |
| sqrt ink | 1/16 | 10.40% | 6.93% | 13 |
| sqrt ink | 1/8 | 7.56% | 4.48% | 9 |
| sqrt ink | 1/4 | 1.69% | −0.52% | 18 |
| sqrt ink | 1 | −38.00% | −32.54% | 46 |

The ink-only strokes endpoint meets the two halo thresholds, but fails faint-ink
retention, missing/cut, bridge and width safeguards. Small mixed blends lose part
of the halo gain and still increase bridges. Square-root ink recovers some faint
fragments in strongly blurred examples, but does not restore clean continuous
lines. Mixed endpoints darken and widen incorrect joins. All ten raw models also
fail; raw weights were diagnostic and were not selected post hoc.

Eight sheets per encoding were actually inspected: EMA blur/.5, strokes/.1 and
strokes/.5, raw blur/.5, real EMA/raw, 1/5 repeats, and partial blank five-step.
These are native pixel crops without contrast or resize enhancement. The other
two rendered raw-strokes sheets are not included in the inspected list. Thin
strongly blurred lines often disappear. Parallel lines and intended gaps merge;
five-step inference reinforces the incorrect connections. Real-lineart eye
contours retain gray grain and shape errors even where fringe scores improve.
This is assistant inspection of fixed development cases, not a blind artist test.

Each evaluation root contains eight metric reports (19,495 predictions including
the inherited initial report), 10 exact model serialization records and 36 exact
GPU endpoint-forward comparisons. All defined scalar metrics are finite and
protected-pixel metrics are zero. The final audit verifies 209 pinned source
files, 15 settings and all saved model triplet hashes per root. Training roots
separately retain full checkpoint/raw/EMA audits at 2k. A bookkeeping assertion
initially counted all ten rendered sheets as eight; the final audit explicitly
lists only the eight actually inspected, without rerunning inference.

Artifacts: `runs/ink-guide-{ink,sqrt_ink}-evaluation-20261008`, their
`completion-audit.json` and `visual-review/step-2000/review.json`, and coordinator
`runs/ink-guide-plan-20261008/evaluation-workflow.json`. Numerical training source
is `d9f7c2130da3bf874de0f50e60b72a3bdf5a173b`; evaluation source manifests are kept
with the outputs. [Plan and pretraining validation](ink-guide-plan-20261008.md).

The next step is a training-only learnability diagnostic on a small, controlled
set of generated lines. Repeated global loss/guide changes have not resolved the
basic thin-line and separation failures. A controlled fit should separate
optimization/clipping limitations from insufficient representational capacity
and corruption ambiguity before another full comparison.
