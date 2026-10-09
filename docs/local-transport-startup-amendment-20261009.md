# Local transport startup discrepancy, 2026-10-09

The original six-update bitwise replay requirement failed before any trained
quality endpoint was inspected. The failed comparison and actual step6 checkpoint
are retained. This is an explicit protocol deviation, not an exact-replay pass.

A fresh six-update run without initial evaluation exactly reproduced preflight.
A fresh run including the actual snapshot/evaluation startup differed from both.
Snapshot/evaluation was checked to leave the complete model/optimizer/EMA state
bitwise unchanged. Explicit data/noise seeds and numerical sources are unchanged.
The exact hardware/compiler cause is not established; GPU floating arithmetic
and the rank loss's sensitivity near tied predictions are plausible causes.

Actual versus preflight: model max difference4.960224e-6, maximum per-leaf
RMS6.413699e-8; optimizer max3.739261e-6; EMA max2.980233e-8. Across all
reruns: model max1.832098e-5, maximum per-leaf RMS3.092443e-7; optimizer
max6.405823e-5; EMA max1.061708e-7. This warrants an explicit reproducibility
limitation, not silently rounding the result to exact or blaming serialization.

Continue this exploratory run from its original saved actual step6 state.
No numerical source, dataset, loss, learning-rate schedule, endpoint, quality
gate, control model, or checkpoint selection is changed. The initial restore,
final full-state restore and raw/EMA export comparisons still require exact
identity. Control six-update replay and all control predictions/metrics still
require exact identity. An encouraging treatment would additionally require an
independent training seed and reproducibility investigation before adoption.
The current model has no quality pass and remains diagnostic only.

Evidence is in runs/local-transport-training-20261009/replay-*-diagnostic.json
and their as-run scripts; startup-checkpoint/6 preserves the original actual
state before checkpoint rotation. replay-audit.json truthfully records
all_state_exact=false. The unchanged coordinator resumes from that checkpoint.
