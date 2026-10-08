# Partial-region summary correction

The conditional-path1,200 evaluation saved all5,472 finite predictions and their
per-case metrics, then stopped during summary construction.360 nonblank rows
have no teacher ink in the active region: the partial mask protects the actual
line. Their ink-deficit fraction is undefined (`None`), which the summary tried
to compare with a numeric threshold. Inference and saved model states succeeded.

The frozen training and inference evaluator files are unchanged. A separate
`complete_fitting_evaluation.py` audits all stored artifacts and finishes the
summary. Undefined retention has no deficit comparison; defined F1 and halo
checks still catch spurious ink in an empty active region. Existing thresholds,
conditions and mode/step selection are unchanged. The helper can also call the
frozen evaluator, handling only this exact known summary exception after saved
metrics pass completeness/hash checks. All other exceptions propagate.

Two focused tests pass in.89 seconds. Rebuilding the already successful400-step
summary matches every prior value and failed-fit record exactly. The1,200 summary
now completes; each aggregation audit binds the original metrics, frozen
inference script and separate helper hashes. The reference's original one-step
arrays must still match the historical arrays exactly. This is a bookkeeping
correction, not a new learned result or a changed quality gate. The queued
tone-weight study uses the same correction before any of its GPU updates.
