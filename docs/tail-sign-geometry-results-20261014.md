# DX: RMS sign aggregation results

RMS aggregation largely removes a long duplicate edge on wide example7, but does not resolve the remaining quality failures. Side/end debris, rough-input gaps and visible fill-corner holes remain. Keep this as a research improvement and a starting point for further controlled diagnosis; no product adoption or reserved confirmation.

## Comparison and validation

Numerical commit e27b594; run `runs/tail-sign-geometry-20261014`. Followed the [plan](tail-sign-geometry-plan-20261014.md): same DU initialization,16 training shapes,25%base/75%rough augmentation shared exactly,6,000 updates per arm,16,064 parameters, six recursion depths, all-stage losses and sqrt(batch)LR. Only the sign-violation reduction differs between arms. Evaluate fixed raw6000, not an image-selected intermediate checkpoint.

Eight tests passed in36.75s; Ruff passed. All6,000 prepared input hashes matched. Every case received every depth213–302 times. Initial six-depth predictions and first-six replay metrics/states were exact, as were full-state restores and16 raw/EMA exports. Training1895.783s, evaluation7.857s. Parameter change RMS/max: mean.0186776/.1934427, RMS.0198524/.1974715. Last600 mean update seconds:.12454/.12636. Cross-objective loss values are not quality rankings.

All384 outputs generated and16 native pages inspected; protected values exactly retained. Per-page observations and hashes are in `evaluation/visual-review.json`; `completion.json` validates the main run and `diagnostics/integrity.json` records exposures and parameter changes.

## Visual findings

- Both arms learn mostly dark continuous narrow centers from wide bars. RMS suppresses the long duplicate edge on original example7 and weakens it on the rough variant. Several detached dots and faint side remnants persist on other wide examples.
- Rough wide0 remains broken: mean loses a lower stretch, RMS an upper stretch.4/6 retain dotted/weak lower parts. This prevents a claim of robust reconstruction even on these known shapes.
- Close curves8–11 are mostly dark and continuous in the original inputs, and mostly retained under rough corruption. Joined ends are not automatically failures: the user permits plausible clean fusion rather than exact teacher topology. Local width irregularities and short cuts remain, with some RMS regressions on rough10/11 and crossing12.
- Blank output remains clean. Fill15 still has prominent holes at its corners; RMS reduces some lower-left holes but worsens some upper/right holes. No opaque-fill improvement can be claimed overall.

The augmented6k fits are more robust than DW's fixed3k fits, but that cross-study observation also changes seed/steps. The within-DX result isolates mean versus RMS aggregation.

## Where remaining fill errors receive gradients

A post-review diagnostic uses already inspected original/rough examples7/15 at depth3 and the fixed RMS6000 checkpoint. Protocol/script/data/checkpoint hashes are bound in `diagnostics/sign-regions/plan.json`. The four reproduced predictions match the stored evaluation exactly; model weights remain unchanged. The plan's tolerance was1e-5 but the observed difference was zero.

Regions use the common union of candidate bands |target|<4 and unambiguous inside/outside signs across all nine allowed shifts. This deliberately excludes ambiguous translated stroke cores; it is not a complete line-recall metric. Original fill15 contains7,186 wrong-sign subpixels in the far interior versus819 in the near interior:89.77% of these false-white samples are far inside. Yet the far-interior errors receive only0.487% of the total fine-field gradient L1 mass, and2.489% of the sign-only gradient mass. Rough fill15 similarly has5,172 far versus1,444 near false-white subpixels, with far errors receiving0.219% of total gradient and0.663% of sign gradient.

Original wide7 has277 unambiguously false-ink far subpixels versus108 near; the far errors receive0.0153% of total gradient. These are local field-gradient proportions, not parameter-gradient feasibility claims or generalization results.

The inherited common near/far spatial weights1/.01 still apply inside the RMS sign term. Thus increasing attention to sparse violations did not eliminate the independent downweighting of errors away from boundaries. Next test near/far sign constraints with separate normalizers while keeping distance/derivative supervision, model, data and inference unchanged. This specifically targets opaque interiors and clean background, without imposing exact deep distance values again.
