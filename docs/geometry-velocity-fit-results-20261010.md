# BC: 512px geometry adaptation failed

Both 6,000-update arms completed. The two-template success did not transfer to
four antialiased 512px training geometries: point MSE failed 52/88 fixed fitting
checks and interval velocity failed 58/88. Interval failed 46/64 retention checks
against MSE. Neither is an adoption candidate.

The initial model was AZ interval36,000, with fresh Adam/EMA, batch4, constant
1e-4 learning rate after sqrt(batch) scaling, and depths1–6. Both arms received
identical fresh Gaussian/path batches. Final raw6000 was evaluated at depths1/3/6
with four fixed draws on four training and four unfitted probe images. The
primary depth3 criteria and all predeclared thresholds were retained.

| Case | MSE F1 | Interval F1 | MSE ink deficit | Interval ink deficit |
|---|---:|---:|---:|---:|
| Curve999 (train) | .83058 | .77385 | .19400 | .26040 |
| Corner1095 (train) | .93009 | .84423 | .06640 | .19582 |
| Parallel1287 (train) | .93709 | .89350 | .05483 | .06806 |
| Gap1383 (train) | .99190 | .97881 | .08507 | .09849 |
| Straight807 (probe) | .44811 | .53598 | .64826 | .60666 |
| Thin903 (probe) | .55613 | .79075 | .28522 | .09818 |
| Junction1191 (probe) | .16507 | .12491 | .82491 | .88117 |
| Taper1479 (probe) | .70342 | .56409 | .41106 | .57515 |

Values average four primary draws; tests apply separately to every draw. Native
crops show broken/faint curves and corners, bridged gaps, near-erased junctions
and thinning/scratchy taper. Better white backgrounds alone are insufficient.

- [Training native crops](../runs/geometry-velocity-fit-20261010/evaluation/train-native.png)
- [Unfitted native crops](../runs/geometry-velocity-fit-20261010/evaluation/probe-native.png)
- [Training overview](../runs/geometry-velocity-fit-20261010/evaluation/train-overview.png)
- [Unfitted overview](../runs/geometry-velocity-fit-20261010/evaluation/probe-overview.png)
- [Partial-mask native crops](../runs/geometry-velocity-fit-20261010/evaluation/partial-native.png)
- [Final audit](../runs/geometry-velocity-fit-20261010/final-audit.json)

All five images were reviewed. Seventeen tests passed (24.88s); first six updates
cover all six recursion depths and replay exactly. Full optimizer/model/EMA
restores and 16 raw/EMA exports are exact. All 52 prediction blocks are finite;
protected prepared-float pixels are exact. Partial masks were inference checks,
not training examples or a claim of mask-boundary quality. Numerical commit:
c8226c59f32d036ae00e0c3d836f4ecac35988ca.

A posthoc, zero-update [loss-region audit](../runs/geometry-velocity-fit-20261010/loss-region-audit.json)
on 16 prepared training paths per arm found paper occupies 99.582% of pixels.
For MSE it accounts for 93.538% of squared error and 98.106% of absolute
output-gradient mass. For interval loss it accounts for only 14.024% and 9.493%;
antialiased edges account for 71.458% and 80.752%. This is an output-space
residual diagnostic, not a parameter-gradient norm or causal proof. It does not
support treating background dominance as the sole explanation for interval's
failure. Next inspect the unbounded intermediate trajectory: interval supervision
permits endpoint overshoot, which the current sampler feeds back unbounded.
No model, plugin, export or production default changed.
