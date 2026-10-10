# AY: Matched continuation to30,000 updates

Completed both predeclared continuations. More training improves reconstruction
but neither arm passes all36 toy checks: uniform time fails15, boundary-focused
time fails17. No broad-data model or deployable artifact is established.

| Arm | Mean nearest-endpoint RMS, strength .5 at6000 | At30000 | Weak-noise whole-image A counts at30000 |
| --- | ---: | ---: | --- |
| Uniform time / conditional OT | .378466681 | .273110350 | 5 / 3 / 0 |
| Boundary time / conditional OT | .463701295 | .314370073 | 18 / 39 / 34 |

Each A count is out of64 at depths1/2/3. Mean error falls about28% and32%,
respectively. Boundary time retains both-mode coverage in all12 groups. Uniform
time passes all weak-noise RMS/resolution criteria, but still fails mode balance
for whole-image weak noise. Both arms fail RMS and resolution criteria at .5;
background speckle remains. The boundary arm's weak-noise depth2/3 whole-image
shape errors are .17039/.17624, also above the fixed.15 threshold. Do not rank
overall suitability just by the number of failed checks.

All three fixed native sheets were inspected: regularity and central gaps
improve compared with6000, but occasional gray gaps and faint background noise
persist. Protected halves remain deliberately rough. These are two memorized
teacher shapes with new Gaussian draws, not unseen-lineart generalization.

36 tests pass in54.27s. Actual parent model/Adam/count/EMA states restore
exactly. Configs record30,000 total steps and6,000 decay steps; learning-rate
schedules match the parents exactly, remaining at1e-5 after6000. The first6
additional updates replay with exact metrics and full state. Full checkpoints
6000/6006/12000/30000 and all16 raw/EMA exports are verified. All252 output
blocks are finite and protected-exact, including180 hash-bound reused controls.
Final raw30,000 was fixed for each arm before training.

- [Depth1](../runs/bridge-duration-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/bridge-duration-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/bridge-duration-20261010/evaluation/depth-3-native.png)
- [Metrics](../runs/bridge-duration-20261010/evaluation/metrics.json)
- [Final audit](../runs/bridge-duration-20261010/final-audit.json)

Next: a matched loss comparison from the boundary-time30,000 state, which
preserves both modes. Test interval velocity error: paper penalizes only
velocity too small to reach white; opaque ink penalizes only velocity too large
to reach black; antialias targets retain ordinary error. Compare against the
unchanged squared-error objective for equal additional updates. This abandons
exact pointwise flow-matching regression at saturated targets and must be
judged by the same output criteria, not claimed to preserve the exact target
probability distribution or optimal flow.
