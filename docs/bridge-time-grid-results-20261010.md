# BA: Fixed-weight boundary-grid results

The interval model with32 boundary-focused Euler steps passes all36 predeclared
toy criteria. The same grid with the matched MSE weights still fails12/36.
This establishes a working learned mechanism on two memorized ambiguous shapes;
it does not establish general lineart restoration or a deployable candidate.

For interval-grid, all12 mean relative endpoint RMS values are .00695–.05121,
at least95.3% of draws per group lie within.20, and all A-mode fractions are
between.1 and.9. The previous failing depth3,whole-image,.1 condition changes
58/64→56/64 A, with the same weights and initial draws. Other groups remain
within their shape criteria. Criteria were not changed.

Three fixed native sheets were inspected. Clean white background and solid
black bars persist, unlike the matched MSE model's speckle. Protected halves
remain rough. A few individual imperfect outputs remain; the criteria concern
aggregate quality, not perfect output for every draw.

14 tests pass in16.99s. All72 physical/clipped uniform-solver control blocks
replay bitwise exactly. All396 saved/reused output blocks are finite and
protected-exact. No learning, checkpoint selection or EMA selection occurred.
The final audit's combined toy_passed flag includes the deliberately retained
MSE control, which fails; failures_per_arm records interval_grid=0,mse_grid=12.

- [Depth1](../runs/bridge-time-grid-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/bridge-time-grid-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/bridge-time-grid-20261010/evaluation/depth-3-native.png)
- [Metrics](../runs/bridge-time-grid-20261010/evaluation/metrics.json)
- [Final audit](../runs/bridge-time-grid-20261010/final-audit.json)

Next: hold both weights and the32-step boundary grid fixed, evaluate1,024 new
Gaussian draws in every condition with a separately declared seed, and keep
the same criteria. This checks sampling stability only, not an independent
training seed, new teacher geometry, the sealed broad-data conditions or actual
Krita quality. Those later stages remain necessary.
