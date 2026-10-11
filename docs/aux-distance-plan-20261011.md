# BK: separate image output from training-only distance prediction

BJ eventually fitted all four training pairs, but point/margin field supervision
still damaged unfitted straight/thin/junction/taper cases. Test whether sharing
the physical preclip output between image coverage and distance is restrictive.
This is a new mechanism diagnostic, not advancement of either failed BJ arm.

Use frozen AO settle7200 core weights (via BJ's strictly adapted control export).
Both arms start identical with fresh Adam and EMA equal to core+auxiliary model;
the optimizer reset is explicit and shared. Train1200 additional updates at
constant1e-5 after sqrt(batch4) scaling (base5e-6), no warmup. Control and auxiliary
arms have identical initialization, batches, architecture, hyperparameters and
auxiliary parameter tree; only the auxiliary loss weight differs,0 versus.1.
This does not replicate BJ's initialization or optimizer schedule, so compare
the new matched arms, with frozen AO as a retention reference.

Core inference model remains130605 parameters. A training-only3x3 ConvBlock
with channels9->16->16->1 predicts normalized coverage distance from the same
source/noise/guide/full-resolution encoder/final decoder features available to
the detail head. It has3777 additional parameters (134382 during training),
seed2026101107, zero final projection and ordinary hidden initialization.
Main image loss, source anchor, clipping, and image recursion are unchanged.
The auxiliary prediction never enters an image output or smaller-stage insert.
It is discarded on export. No labels, distance transform or new input at inference.

Auxiliary target is BG coverage-distance/radius8, point interval-L1 only at the
truncation cap, with BH tone-power.5/active per-image normalization. Use the
highest-stage auxiliary loss; gradients traverse shared features and smaller
image stages. Fixed TRAIN4/PROBE4,512px, seed2026101066, NumPy pyramid then noise,
cyclic depths1–6, main AD image objective all unchanged. No new data or coefficient
search. A zero-weight control must match the existing core loss and gradients;
changing auxiliary weights must not change image predictions at any stage.

Keep0/6/400/1200 full wrapper+optimizer+EMA states and16 core-only raw/EMA exports.
Replay first6 updates, all restores, core extraction and source-only protected
pixels. Reproduce all8 AO primary outputs before training and final evaluation.
Final raw1200 only:3 methods x8 cases xdepths1/3/6 =72 rows,18 full prediction
blocks and3 depth6 partial blocks,5 image sheets. Inspect train/probe native and
overview and partial native.22 fixed fitting checks per method (66 total).
For each trained method retain16 F1/deficit checks versus AO; auxiliary also
retains16 versus the matched trained control (48 total). Thresholds unchanged:
F1 loss<=.005, deficit increase<=.01. No checkpoint selection or gate relaxation.

Multitask distance regression is motivated by
[Audebert et al.](https://arxiv.org/abs/1909.01671) and the removable training
regularizer in [Dangi et al.](https://arxiv.org/abs/1901.01238). Those segmentation
results do not establish benefit for antialiased lineart, nor do we reproduce
their architectures. No production/plugin/default changes or adoption. Broad
training, independent/sealed confirmation and BGRA parity would still be needed.
