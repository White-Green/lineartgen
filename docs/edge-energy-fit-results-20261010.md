# AS: edge normalization reduces fitted halo but does not solve restoration

Completed the fixed 1,200 updates, exact controls, state/export audits and both
final image reviews. The candidate fails one of 22 fitting checks: curved-line
F1 .959086584 < .98, versus AD .958878505. All eight AD retention guards pass.
The declared broad-training gate is unmet; this loss is not advanced or adopted.

For the fitted curve, halo .001312949→.000894730 (about 32% less), width MAE
.0968156→.0775825 px, coverage overlap .951465→.958452, but deficit
.00611634→.01044479. Fitted corner/parallel/gap pass their absolute checks, with
zero cuts/bridges, but corner and parallel coverage are slightly worse than AD.

Unfitted outputs remain inadequate: straight deficit .09123→.14271, thin F1
.98663→.97521, junction F1 .74044→.72323 with cut rate .26667, and taper deficit
.24186 with width MAE 1.17030 px. The taper still visibly has a gray uneven center.
The fitted native sheet looks broadly similar to AD; the quantitative halo gain
is not evidence of a general visual cleanup.

37 tests passed in 27.14 s. The new coefficient is the only config change. On all
eight real prepared tensors, direct image-space descent lowers the auxiliary,
protected and teacher gradients are zero, and additive gradient integration error
is at most 3.10e-10. Weighted auxiliary/base output-gradient norm ratios span
.192–.847; those are not parameter-gradient norms. All six initial depth outputs
are bitwise equal. Full model/AdamW/EMA restore at6/100/400/1200 and all six raw/EMA
exports are exact; every AD control output/metric is unchanged at0/100/400/1200.
Numerical commit `79711cae73dc0a83b5f1dbee1f47ec862386e800`.

- [Predeclared plan](edge-energy-fit-plan-20261010.md)
- [Final audit](../runs/edge-energy-fit-20261010/final-audit.json)
- [Fitted four examples](../runs/edge-energy-fit-20261010/evaluation/step-1200/train-native.png)
- [Unfitted four examples](../runs/edge-energy-fit-20261010/evaluation/step-1200/probe-native.png)

This rejects the specified coefficient/budget as a follow-up candidate. It does
not show that gradient supervision is always ineffective. No production/plugin
change, independent seed or sealed confirmation was performed.
