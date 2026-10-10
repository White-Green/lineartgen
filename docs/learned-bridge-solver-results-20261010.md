# AU: increasing Euler steps does not rescue the learned field

The same AT bridge weights were evaluated at32/64/128/256 steps with exactly the
same12 groups and64 Gaussian draws/group. All32-step physical/clipped arrays and
metrics reproduce AT exactly. No neural updates, new draws, or weights were used.

| Steps | Failed toy checks of36 | Mean of12 group endpoint RMS values |
| --- | --- | --- |
|32|30|.564817|
|64|30|.556115|
|128|29|.565989|
|256 (fixed primary)|28|.586163|

At256, every group's mean RMS still exceeds.15 (range .168464–1.029864). Both
mode bias and visible paper artifacts remain in all three inspected native
sheets. Some low-strength groups improve, but higher-strength whole-image RMS
worsens at all three depths. Fewer failed thresholds do not mean better overall
quality: the error magnitude and sample appearance remain inadequate.

Even128→256 changes individual pixels substantially (maximum differences
.134–1.430 across groups), partly reflecting changed selected interpretations.
This is not a proof of full numerical convergence. It does show that increasing
this sampler's step count through256 is insufficient to recover clean modes from
the fixed learned field. It does not justify deploying AT or changing its gates.

All48 output blocks are finite, all protected pixels exact, and the three fixed
native sheets were reviewed. Every inherited numerical file is byte-identical to
AT and its previous16 tests remain applicable; AU adds actual32-step full-array
replay over all12 groups. Timing is descriptive and includes compilation where
present, not a warmed Krita/ONNX benchmark.256 recursive evaluations require
256×depth CNN calls. Numerical commit `c564aa3`.

- [Plan](learned-bridge-solver-plan-20261010.md)
- [Final audit](../runs/learned-bridge-solver-20261010/final-audit.json)
- [Depth1](../runs/learned-bridge-solver-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/learned-bridge-solver-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/learned-bridge-solver-20261010/evaluation/depth-3-native.png)

The endpoint-to-velocity conversion divides prediction error by remaining time.
A model whose endpoint residual does not shrink appropriately near time1 can
produce a poorly behaved field; additional steps alone cannot force that error
to vanish. This is a mathematical failure mechanism to audit, not yet a proven
explanation of every AT artifact. A direct velocity parameterization is a
separate possible training experiment, not implemented by AU.
