# BF: fresh parameters were worse than the toy-initialized control

The fresh seed42 model completed6,000 updates under BC's exact configuration,
batches, noise, time distribution and interval velocity objective. It failed
70/88 fixed fitting checks versus58/88 for the frozen BC interval model, with
52/64 retention failures. Initialization from the two-bar toy is not the sole
cause of the geometry problem. Lifetime compute differs: the control has prior
toy training. This does not prove a model-size lower bound.

| Case | Fresh F1 | Fresh ink deficit | Fresh halo |
|---|---:|---:|---:|
| Curve999(train) | .45012 | .60016 | .00719 |
| Corner1095(train) | .50219 | .43243 | .00191 |
| Parallel1287(train) | .82995 | .18709 | .00457 |
| Gap1383(train) | .99147 | .24622 | .00035 |
| Straight807(probe) | .15059 | .84130 | .02904 |
| Thin903(probe) | .72327 | .24546 | .01254 |
| Junction1191(probe) | .05400 | .94272 | .00551 |
| Taper1479(probe) | .63410 | .51941 | .00316 |

Four-draw means at depth3; gates remain per draw. Curve, corner and junction
fragmentation is plainly visible. Reduced halo can reflect erased ink and is not
sufficient evidence of improvement. All five fixed sheets were reviewed:

- [Training native](../runs/fresh-geometry-fit-20261011/evaluation/train-native.png)
- [Unfitted native](../runs/fresh-geometry-fit-20261011/evaluation/probe-native.png)
- [Training overview](../runs/fresh-geometry-fit-20261011/evaluation/train-overview.png)
- [Unfitted overview](../runs/fresh-geometry-fit-20261011/evaluation/probe-overview.png)
- [Partial mask](../runs/fresh-geometry-fit-20261011/evaluation/partial-native.png)
- [Final audit](../runs/fresh-geometry-fit-20261011/final-audit.json)

Eighteen tests passed(26.71s), all25 frozen-control final blocks reproduced
bitwise, first six fresh updates replayed exactly, full states and8 raw/EMA
exports exact. All52 prediction blocks finite and prepared-float protected pixels
exact. Numerical commite9658f83a32f4f09ba83d581b206083205bd3a9c.
No candidate or production changes. Next test a geometry representation's exact
reconstruction of antialias coverage before attempting its supervision in a CNN.
