# BE: more source-start exposure did not repair geometry

Both actual BC interval6000 states were continued to12000, retaining Adam/EMA,
with common constant1e-5 LR. The control kept25%tau0; the focused arm used75%.
Actual exposures were5864/24000(24.43%) and18033/24000(75.14%). Both failed56/88
fixed fitting criteria. Focused failed20/64 retention checks against control.
No candidate; simply increasing observed-source training was insufficient.

| Case | Control F1 | Focused F1 | Control ink deficit | Focused ink deficit |
|---|---:|---:|---:|---:|
| Curve999(train) | .77727 | .75073 | .23379 | .25471 |
| Corner1095(train) | .82825 | .86066 | .20791 | .20108 |
| Parallel1287(train) | .90577 | .91177 | .03034 | .02843 |
| Gap1383(train) | .98788 | .99081 | .04269 | .03676 |
| Straight807(probe) | .50484 | .49740 | .64146 | .64900 |
| Thin903(probe) | .66079 | .64860 | .14388 | .13496 |
| Junction1191(probe) | .18711 | .19758 | .82946 | .81841 |
| Taper1479(probe) | .53765 | .52673 | .59499 | .60263 |

Four-draw means at the predeclared depth3; thresholds apply per draw. Native
images show ragged/thin curves, missing corner pixels, gap bridges, fragmented
junctions and overly thin taper. All five sheets were reviewed:

- [Training native](../runs/start-focused-fit-20261011/evaluation/train-native.png)
- [Unfitted native](../runs/start-focused-fit-20261011/evaluation/probe-native.png)
- [Training overview](../runs/start-focused-fit-20261011/evaluation/train-overview.png)
- [Unfitted overview](../runs/start-focused-fit-20261011/evaluation/probe-overview.png)
- [Partial mask](../runs/start-focused-fit-20261011/evaluation/partial-native.png)
- [Final audit](../runs/start-focused-fit-20261011/final-audit.json)

Nineteen tests passed(30.29s); an initial test-fixture return-value mistake was
corrected before freeze and its failure log retained. Initial full states equal
BC exactly, first six updates replay all six depths exactly, all full restores
and16 raw/EMA exports exact. All52 blocks finite and protected prepared-float
pixels exact. Numerical commit06254ae; no frozen numerical files changed.

A zero-update posthoc [one-step diagnostic](../runs/start-focused-fit-20261011/posthoc-start-endpoint.json)
used the same final weights, inputs and four primary noise draws. Curve F1 was
.85587(control)/.87317(focused), above the32-step .77727/.75073 but still poor.
Corner F1 was .94492/.94776, while unseen junction F1 was only .34053/.35465.
This neither passes the predeclared sampler gate nor establishes a new candidate.
It suggests both learned initial restoration and iterative transport need work.
The executable diagnostic was preserved as a text artifact without modifying
frozen numerical files. Next isolate the narrow vertical-bar initialization by
training the same geometry task from fresh parameters under BC's exact schedule.
No independent/sealed confirmation, deployment or default changes.
