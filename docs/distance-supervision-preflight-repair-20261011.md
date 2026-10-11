# BH preflight repair before any updates

The original frozen150db4c run stopped before training: its helper returned JAX
arrays to a legacy NumPy pyramid. That pyramid calls the array's mean method,
so downsampling took place in JAX and introduced floating-point differences from
AD. Maximum output difference across the eight controls was8.106232e-6. Reducing
the preflight batch from8 to4 alone did not remove the discrepancy.

Explicit host NumPy conversion before the original preparation made every one
of the eight controls bitwise equal to AD. Evidence is saved in
[preprocessing-repair-audit.json](../runs/distance-supervision-fit-20261011/preprocessing-repair-audit.json).
Zero optimizer updates occurred; the numerical code is left frozen. Continue in
jax-distance-supervision-fit-replay-20261011, explicitly preserving NumPy input
pyramids and original4-image evaluation batches. Add regression tests proving
all prepared arrays equal legacy preparation at depths1/6. The hypothesis,
loss weight, radius, data, init, optimizer, schedule and thresholds are unchanged.
