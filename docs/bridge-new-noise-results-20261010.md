# BB: Fresh-noise sampling stability

The interval-grid candidate passes all36 unchanged toy criteria on1,024 fresh
draws per condition (12,288 draws total). The matched MSE-grid control still
fails12/36. Both weights, the32-step solver, thresholds and displayed sample
positions were fixed before evaluation; new seed2026103089 was used.

Across12 interval conditions, mean relative nearest-endpoint RMS is
.00749–.04795, the fraction within .20 is .95605–1.0, and nearest-A fraction
is .41113–.86133. Both modes persist, but their frequencies are not50/50;
this is not exact conditional-distribution recovery. All three fixed native
sheets were inspected. Backgrounds remain cleaner and bars more solid than
MSE; occasional imperfect gaps or small extra marks remain.

9 tests pass in2.42s, including metric aggregation and sample-count-invariant
gating.192 unique reproducible chunk keys are recorded; both models use the
same keys. All24 saved blocks contain1,024 physical/clipped outputs, are finite
and preserve protected pixels exactly. Model files are byte copies of BA and
hash-bound. No neural update or new checkpoint selection occurred.

This confirms stability across new Gaussian draws for the same two memorized
teachers. It is not an independent training seed, new geometry, the sealed
broad-data conditions, or Krita/BGRA/ONNX validation. No model was adopted.
The final audit's combined flag includes the intentionally retained failing
MSE control; failures_per_arm records interval_grid=0 and mse_grid=12.

- [Depth1](../runs/bridge-new-noise-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/bridge-new-noise-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/bridge-new-noise-20261010/evaluation/depth-3-native.png)
- [Metrics](../runs/bridge-new-noise-20261010/evaluation/metrics.json)
- [Final audit](../runs/bridge-new-noise-20261010/final-audit.json)

Next: expand the fitting diagnostic to antialiased curves, corners, parallel
lines and intentional gaps at512px, restoring full training recursion1–6.
Compare point and interval velocity losses from the same initial weights and
fresh optimizers on the established four-pair synthetic fitting set, with
unfitted straight/thin/junction/taper probes. This is a new geometry study, not
a relabeling of the completed two-template pass as broad generalization.
