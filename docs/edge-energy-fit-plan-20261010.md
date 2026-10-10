# AS: normalized interval edge energy on fixed fitting pairs

Declared after AR's final 14-image review, before this experiment's numerical
outputs. AR's 4x parameter count improved separation but left gray fine curves,
thick repeated strokes and 41/229 failed broad checks. AO proves that additional
optimization can fit four pairs with the small model, but not that the current
loss efficiently teaches profiles. Existing Sobel MSE is averaged over all pixels.
AS tests teacher-normalized spatial residuals at fixed 1,200 updates against AD.
This is a local fitting diagnostic, not evidence of broad or Krita quality.

For physical preclip opacity p=(1-z)/2 and teacher y=clip((1-target)/2), residual
e is max(p,0) on paper y=0, min(p-1,0) on black y=1, and p-y on fractional AA.
This is the signed residual to each teacher's valid clipping interval. Correct
AA and correct over-saturated endpoints have zero residual. For horizontal and
vertical adjacent pairs whose two pixels are editable, add squared differences
of e. Divide each image's sum by max(sum squared teacher differences,1). If the
teacher differences are all zero, divide by max(number of valid pairs,1) instead.
Average over images. Teacher, masks and denominator are stopped-gradient.

Add coefficient .1 to AD's loss. This is a spatial auxiliary, not normalized
pixel L1 and not a replacement for it. Constant residuals are in its null space;
base pixel/opacity losses supervise them. Spatial gradients need not point toward
the individual teacher at every pixel. Direct descent must reduce the auxiliary,
protected pixels and teacher must receive zero gradient. Check correct AA and
valid saturated endpoints, wrong-saturation correction on a stroke, 90-degree
symmetry, padding invariance for nonconstant teachers, zero-edge fields, exact
disabled-path compatibility, both generic trainers, and full-state/next-update
restore. On all eight real prepared tensors audit loss, gradient scale relative
to the base objective, and additive integration before learning. This does not
establish parameter-gradient scale or learned improvement.

Only loss.edge_energy_weight 0→.1 changes from AD's band arm. Same R raw4k initial
130,605 parameters, fresh AdamW/EMA, batch4, seed2026101066, constant LR5e-5 times
sqrt(4)=1e-4, no warmup, 1,200 updates. Keep physical interval L1, preclip opacity
.3, gap clearance .1, paper band .1, all other terms and exact input tensors.
TRAIN=999/1095/1287/1383, PROBE=807/903/1191/1479, 512px, strength .5, strong blur.
Depth cycle1–6, stage-loss multiplier0 with recursive gradients, downsample before
Gaussian, masks and source-only one_pass remain unchanged. No AO/AR weights.
Retain states6/100/400/1200, six raw/EMA exports and exact AD reference outputs at
0/100/400/1200. Initial all-six-depth outputs must agree within1e-5 (expect exact).

Final raw1200 only; keep AD's 22 fitting checks (F1>=.98, deficit<=.02, halo<=.003,
width MAE<=.2, cuts/bridges zero where defined) plus eight retention checks against
AD (F1 loss<=.005 and deficit increase<=.01 on each TRAIN case). Review both final
native sheets, including all four unfitted examples. Continuous coverage overlap
is descriptive, not a replacement gate. Only passing all30 checks allows a broad
follow-up; fixed-pair weights are never adopted. No coefficient/checkpoint/EMA
selection, independent seed, sealed confirmation, production or plugin changes.
