# Broader detail study validation

Both data families pass six real512px GPU updates through recursion1–6 and exact
full-state save/restore. Both unchanged guided counterparts reproduce their
historical six-update model/optimizer/EMA state bitwise. Each new model also
matches the100k parent exactly in32 forward comparisons across1–6 recursion
levels and1/2/5 sampling steps (64 paired comparisons across the two families).

The270-test suite was inherited, not rerun: all218 source/test/config files in
the tested local fitting manifest are byte-identical in this checkout. New setup,
comparison and rendering scripts pass Ruff and compile checks. Preparation
validates actual config deltas, dataset split identity and parent source hashes.
The plan and222-file numerical source are frozen at
`29dde4b1a12ab97d5f1a1323befa07d77af756af`, with14 sealed settings per root.

Both actual trajectories saved and restored update1 before continuing. Their
model/EMA snapshots match the restored checkpoint exactly, optimizer states are
finite and counters equal1. This actual update audit does not directly compare
the optimizer with presave process memory; the separate six-update preflight does
perform that full-state comparison. Both new experiments then resume the saved
optimizer/EMA without resetting the10k schedule.

Separately, the already-trained local width16 raw1200 checkpoint passes CPU JAX
versus CPU ONNX/BGRA checks on three shapes145×129,512×512,513×385, strengths
0/.1/.5 and1/2/5 denoising iterations:27 comparisons, maximum float difference
1.026e-5, maximum byte difference1, five differing bytes in total, and zero
protected-pixel error. The exported checkpoint is the local fitting artifact,
not the still-training broader candidate. It is not adopted. These checks do not
establish Windows DirectML, Krita GUI performance or actual visual quality.

Records: `runs/detail-study-plan-20261008/test-inheritance.json`,
`actual-step1-audit.json`, `local-detail16-runtime/runtime-check.json`, and each
`detail-{strokes,mixed}-20261008-screen` root's preflight/source records.
The fixed review requires all ten native sheets per family, saved in
`visual-review-plan.json`; cropping never depends on candidate output values.
