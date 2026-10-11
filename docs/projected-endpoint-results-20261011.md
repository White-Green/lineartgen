# BD: endpoint projection helps corners, does not solve geometry restoration

No updates. Projecting the predicted endpoint at each Euler step reduced interval
fit failures from58/88 to40/88, but17/64 retention checks failed against its
unprojected model. MSE changed52/88 to50/88, with5/64 retention failures. These
counts describe fixed checks, not an aggregate quality score. No candidate.

For interval, corner F1 rose .84423→.92766 and ink deficit fell .19582→.03765.
Curve F1 fell .77385→.75820. Unfitted thin-line F1 fell .79075→.59675 and its
ink deficit rose .09818→.25787. Junction F1 improved .12491→.22166 but the line
remains fragmented. Taper remains too narrow and scratched; the trained gap is
still bridged. Values average four primary draws; tests apply per draw.

The trajectory hypothesis was partly supported: on train depth3 draw0, original
interval input range progressed from[.32,1.62] initially to[-2.05,3.04] at grid
index24, with99.7% of pixels outside[-1,1]. Projection gave[-.80,1.09] there.
Initial Gaussian noise is unchanged and may legitimately exceed[-1,1]; this is
endpoint projection, not clamping of every intermediate image. Projection alone
cannot recover missing learned geometry.

Fourteen tests passed(5.19s), all52 original prediction blocks reproduced bitwise,
all102 blocks and392 metric rows audited, and all seven fixed sheets inspected:

- [Interval train native](../runs/projected-endpoint-20261011/evaluation/interval-train-native.png)
- [Interval probe native](../runs/projected-endpoint-20261011/evaluation/interval-probe-native.png)
- [Interval train overview](../runs/projected-endpoint-20261011/evaluation/interval-train-overview.png)
- [Interval probe overview](../runs/projected-endpoint-20261011/evaluation/interval-probe-overview.png)
- [Interval partial mask](../runs/projected-endpoint-20261011/evaluation/interval-partial-native.png)
- [MSE train native](../runs/projected-endpoint-20261011/evaluation/mse-train-native.png)
- [MSE probe native](../runs/projected-endpoint-20261011/evaluation/mse-probe-native.png)
- [Final audit](../runs/projected-endpoint-20261011/final-audit.json)

All values finite and protected prepared-float pixels exact. Numerical commit
6f88e5e33eb5f213d11b3726f5717304cf0833a7. The initial CLI freeze lookup failed
before freezing; the explicit action map was fixed before running numerical
comparisons. No frozen numerical files changed. No toy confirmation was launched
because the geometry gate failed. No production, plugin or default changes.
