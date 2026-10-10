# AZ: Interval velocity loss results

Clear improvement in the ambiguous two-template toy, but not a complete toy
pass or an adopted model. From identical boundary-time30,000 full states,
6,000 more updates give12/36 failed criteria for ordinary MSE and1/36 for
interval error. All24 interval shape-fidelity criteria pass; one mode-balance
criterion fails: depth3,whole-image,strength.1 gives58/64 nearest-A (.90625),
above the unchanged.9 bound. Do not round this into a pass.

| Final raw36,000,32 steps | Mean RMS at strength .5 | Within RMS .20 at strength .5 |
| --- | ---: | ---: |
| Ordinary velocity MSE | .301002585 | 0 / 384 |
| Interval velocity error | .039535829 | 374 / 384 |

The high-noise mean shape error falls about87%. Interval group means across
all12 conditions range .00780–.06216. All groups have at least95.3% of draws
within .20. All three fixed native sheets show much cleaner white backgrounds
and solid black lines, with occasional grooved/incomplete bars still visible.
Protected halves are intentionally rough. Preclip values extend approximately
-3.06 to2.79; clipping is part of this custom toy renderer. This is not an
actualBGRA/ONNX test or evidence of ordinary-lineart generalization.

The interval residual is one-sided only for exactly saturated targets; correct
saturation has zero gradient and wrong saturation retains it. Intermediate
coverage keeps ordinary squared error. The endpoint equivalence for remaining
u>0 and behavior at protected pixels are tested. This objective changes the
flow's statistical target, so sharp results do not establish exact sampling of
the true conditional distribution or globally optimal transport.

42 tests pass in66.27s. Both model/Adam/count/EMA copies start exactly equal to
the actual30,000-step parent; LR schedules and first6-update replay are exact.
States30000/30006/32000/36000 and16 raw/EMA exports are verified.324 saved
output blocks are finite and protected-exact, including252 reused controls.
The final raw36,000 checkpoint and first4 displayed draws were predeclared.

- [Depth1](../runs/interval-velocity-fit-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/interval-velocity-fit-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/interval-velocity-fit-20261010/evaluation/depth-3-native.png)
- [Metrics](../runs/interval-velocity-fit-20261010/evaluation/metrics.json)
- [Final audit](../runs/interval-velocity-fit-20261010/final-audit.json)

Next test: keep both final weights fixed and compare uniform Euler times with
sin-squared time nodes, emphasizing the beginning/end at the same evaluation
count. Training uses boundary-focused times but uniform32-step inference can
skip the early weak-noise decision region. This is a solver hypothesis, not
permission to loosen the mode-balance criterion or a declaration of adoption.
