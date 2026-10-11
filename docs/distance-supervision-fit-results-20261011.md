# BH: distance supervision improves some unseen geometry, with regressions

The corrected host-pyramid run completed1200 updates. Distance supervision failed
2/22 fitting checks(curve F1 and parallel-line bridging), versus1/22(curve F1)
for AD. Two of16 retention checks failed: trained-curve F1 and unfitted-straight
ink deficit. Not a candidate; no broad training gate is claimed.

Unfitted taper deficit improved .25450→.19220, halo .013237→.004737(about64% lower),
width error1.24490→1.04946px, while F1 .94650→.94434. The line is visibly darker
with less peripheral gray, but still too gray/narrow. Junction F1 improved
.74044→.78240, deficit .21137→.16055 and cut rate .33333→.2. However, trained
curve F1 fell .95888→.95106, parallel bridge rate rose0→.33333, and straight
probe deficit rose .09123→.12423. Exact distance point targets at saturated
near-boundary pixels may remove useful saturation margin; this is a hypothesis
for the next comparison, not a demonstrated cause.

This is the established130605-parameter source-feature/source-anchor one-pass
model, initialized exactly from R4000 with AD's data/config/schedule. It is a
separate diagnostic from the velocity experiments. It cannot represent multiple
conditional output modes merely because Gaussian inputs are generated.

All five final sheets were inspected:

- [Training native](../runs/distance-supervision-fit-20261011/evaluation/train-native.png)
- [Unfitted native](../runs/distance-supervision-fit-20261011/evaluation/probe-native.png)
- [Training overview](../runs/distance-supervision-fit-20261011/evaluation/train-overview.png)
- [Unfitted overview](../runs/distance-supervision-fit-20261011/evaluation/probe-overview.png)
- [Partial native](../runs/distance-supervision-fit-20261011/evaluation/partial-native.png)
- [Final audit](../runs/distance-supervision-fit-20261011/final-audit.json)

Twenty-four tests passed(19.78s), including exact legacy host preprocessing,
weight-zero loss/gradient equivalence, field gradients and actual update/restore.
All8 AD primary outputs replayed bitwise before and after training; first six
updates and every full restore are exact, as are8 raw/EMA exports. All16 prediction
blocks finite, protected prepared-float pixels exact. Numerical commitf1ae685.
The initial run stopped before any training because JAX arrays reached a NumPy
pyramid. The frozen failure and [repair evidence](../../jax-distance-supervision-fit-20261011/docs/distance-supervision-preflight-repair-20261011.md)
are retained separately. No production or plugin changes.
