# BI: retaining saturation removes the bridge, but hurts unseen lines

The1200-update margin variant failed1/22 fitting criteria(curve F1), versus2/22
for BH point-distance. The trained parallel bridge disappeared(.33333→0), curve
F1 improved .95106→.95709 and its halo fell .001225→.000919. All trained cuts
and bridges are zero. However7/16 retention checks against BH failed.

Unfitted thin-line F1 fell .98201→.72368 and deficit rose .01226→.08831. Straight
probe deficit rose .12423→.18799. Taper lost part of BH's improvement: deficit
.19220→.24318, halo .004737→.008332 and F1 .94434→.92058. Junction F1 fell
.78240→.74222; cut rate remained.2. Counts alone cannot describe this tradeoff.
No candidate. Longer equal-budget fitting is needed before attributing the
remaining curve error to representational capacity; AD itself needed longer
training and a final lower learning rate to pass its fixed fitting criteria.

Twenty-eight tests passed(23.92s);8 frozen BH controls replay bitwise before and
after training. First six updates, full-state restores and8 raw/EMA exports exact.
All16 output blocks finite and protected prepared floats exact. All5 sheets read:

- [Training native](../runs/distance-margin-fit-20261011/evaluation/train-native.png)
- [Unfitted native](../runs/distance-margin-fit-20261011/evaluation/probe-native.png)
- [Training overview](../runs/distance-margin-fit-20261011/evaluation/train-overview.png)
- [Unfitted overview](../runs/distance-margin-fit-20261011/evaluation/probe-overview.png)
- [Partial native](../runs/distance-margin-fit-20261011/evaluation/partial-native.png)
- [Final audit](../runs/distance-margin-fit-20261011/final-audit.json)

Numerical commitcfd34a2. Same R4000 init, data, config, radius8 and weight.1 as BH;
only saturation bounds change. This one-pass source model is separate from the
conditional velocity flow and does not restore multimodal sampling. No broad or
independent confirmation, deployment, plugin or default changes.
