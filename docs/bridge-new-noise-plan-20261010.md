# BB: New Gaussian draws with fixed weights and solver

Declared after BA's interval-grid36/36 toy pass. Freeze both AZ36,000 raw models
and BA's32-step sin-squared Euler grid. Evaluate1,024 draws per each of the same
12 depth/mask/strength conditions, using previously unused seed2026103089.
Generate64 draws per chunk,16 chunks per group. Derive a key by folding the
same group id as BA into the new seed, then folding chunk0..15. Both models
use identical keys and observed inputs. There is no new training, coefficient,
solver, threshold, checkpoint or output selection.

Apply the unchanged36 criteria per arm at32steps: mean nearest-endpoint RMS
<=.15, fraction within.20>=.8, nearest-A fraction .1–.9. Require finite physical
values and exact protected pixels. Interval-grid is the candidate; MSE-grid is
the matched control. Record all24 full physical/clipped blocks,1,024 samples
per block, and all192 unique chunk keys. Inspect3 fixed native sheets containing
A/B/rough and the first4 outputs per arm. Do not hide a new-draw failure because
the earlier64-draw development set passed.

Tests verify unique/stable chunk keys and disjoint seed use, and that metrics
on concatenated samples agree with properly aggregated chunk metrics. Inherited
solver/mask tests and actual assertions cover numerical behavior. Model files
are copied byte-for-byte from BA and hash-bound before evaluation.

This confirms sampling stability only for the same two memorized teachers.
It is not an independent training seed, new shapes, held-out ordinary images,
the sealed broad-data conditions, or Krita/BGRA/ONNX validation. No adoption or
plugin change. Subsequent geometry and training-seed experiments remain needed.
