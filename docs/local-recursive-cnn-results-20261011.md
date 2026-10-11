# CI: full-resolution local CNN fitting pilot

The15809-parameter local CNN learned this fixed synthetic task more efficiently
than the fresh134688-parameter U-Net at1200 matched updates. This does not yet
meet the fitting criteria and is not an adoption candidate. The comparison
changes architecture as a bundle, including activation and input centering;
it does not isolate pooling, and is not a model-size lower bound.

Numerical source: `0b14716eb6c396ec2dbdfdf57dc88ec31e37ff61`.
The original `8bb1cca` run stopped before any SGD on a CPU/GPU initialization
roundoff assertion. Both persisted initial parameter trees and configurations
were reproduced exactly in this worktree. Same fresh Adam/EMA, seed42, four
AD512px pairs, noise seed2026101093, depth1..6, batch4, constant1e-4 LR
(5e-5*sqrt4), CH two-step rollout objective with.1 AD appearance loss.
Final raw1200 only. CHimage8400 is a quality reference with a different training
history, not the matched control.

| Inference steps | CH reference failures/88 | Fresh U-Net failures/88 | Local failures/88 | Local vs U-Net retention failures/64 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 36 | 61 | 43 | 0 |
| 32 | 51 | 71 | 51 | 4 |

Retention covers F1 and opacity deficit only, not all appearance errors. At2
steps local vs CH has6/64 retention failures; at32 it has12/64. Counts are not
an overall quality score. All four training cases still contribute failures.

At2 steps, the local model restores thin903 and all three junction1191 branches
much more clearly than the fresh U-Net. Taper1479 still has a hollow/gray
interior with dark outlines. Curves have small edge and width errors. Parallel
lines have weak unwanted connections (bridge rate2/3 for each draw at2 steps,
1 at32), despite high F1. Gap1383 stays open according to all four connection
probes; its clearance width/position is incorrect and the line becomes too
heavy at32. An initial visual impression that this short gap was closed was
corrected by checking actual pixels and probe values.

Partial outputs preserve protected pixels exactly, but local2 has a dotted
trace just inside the editable mask boundary. Local32 avoids the CH reference's
broad gray fields, but this is not enough for acceptance. Only u=1/.5 were
trained;32 steps evaluate other progress values too. All ten fixed native,
overview and partial sheets were reviewed at original resolution.

Mean update time after the first six compilations was1.1175s for U-Net and
.7199s for local on this AMD8060S (1.55x update throughput). This includes both
rollout passes, gradients, Adam/EMA and synchronization, in sequential paired
updates. It is not an inference benchmark. Local has about1/8.52 as many parameters.

25 tests passed in54.88s. Full states0/6/400/1200, six-update full replay,
all16 raw/EMA exports,50 CH physical/clipped output blocks and192 metric rows
were exact. Evaluations contain576 rows,150 blocks,528 fitting and384 retention
checks; all outputs finite and protected pixels exact. No independent training
seed, sealed confirmation, real rough pairs, actual BGRA/ONNX evaluation or
Krita GUI test. Local width nesting is not implemented in this pilot. Production
weights, runtime defaults and plugin are unchanged.

## Per-case means at recursion depth3

Four Gaussian draws per case. Each entry is F1 / opacity deficit / halo excess.
First four cases are fitted; last four are unfitted probes repeatedly used for
mechanism diagnosis, not sealed confirmation.

### 2 inference steps

| Case | CH reference | Fresh U-Net | Local |
| --- | --- | --- | --- |
| 999 | 0.819747 / 0.073594 / 0.017555 | 0.610028 / 0.200529 / 0.073639 | 0.903589 / 0.060042 / 0.009333 |
| 1095 | 0.966893 / 0.013347 / 0.002465 | 0.873739 / 0.052976 / 0.039022 | 0.968037 / 0.030591 / 0.010174 |
| 1287 | 0.973130 / 0.013090 / 0.001042 | 0.970829 / 0.015782 / 0.004142 | 0.982007 / 0.014101 / 0.001888 |
| 1383 | 0.959629 / 0.083516 / 0.000425 | 0.841269 / 0.069960 / 0.001340 | 0.980723 / 0.033633 / 0.000794 |
| 807 | 0.554694 / 0.513809 / 0.059298 | 0.688500 / 0.298704 / 0.084217 | 0.847910 / 0.086952 / 0.000827 |
| 903 | 0.171444 / 0.711344 / 0.085797 | 0.106757 / 0.727013 / 0.141771 | 0.994260 / 0.010251 / 0.000630 |
| 1191 | 0.202331 / 0.821128 / 0.020155 | 0.415819 / 0.491904 / 0.095763 | 0.860494 / 0.212974 / 0.010227 |
| 1479 | 0.472845 / 0.665749 / 0.009955 | 0.663535 / 0.417015 / 0.050284 | 0.731634 / 0.333538 / 0.012951 |

Images: [train native](../runs/local-recursive-cnn-20261011/evaluation/train-native.png), [probe native](../runs/local-recursive-cnn-20261011/evaluation/probe-native.png), [train overview](../runs/local-recursive-cnn-20261011/evaluation/train-overview.png), [probe overview](../runs/local-recursive-cnn-20261011/evaluation/probe-overview.png), [partial](../runs/local-recursive-cnn-20261011/evaluation/partial-native.png).

### 32 inference steps

| Case | CH reference | Fresh U-Net | Local |
| --- | --- | --- | --- |
| 999 | 0.749020 / 0.067459 / 0.054900 | 0.395324 / 0.389146 / 0.128156 | 0.772383 / 0.127162 / 0.014931 |
| 1095 | 0.948424 / 0.014428 / 0.036174 | 0.651659 / 0.216759 / 0.073124 | 0.971083 / 0.022588 / 0.015303 |
| 1287 | 0.995963 / 0.002621 / 0.023150 | 0.964738 / 0.018635 / 0.004457 | 0.994854 / 0.005122 / 0.001847 |
| 1383 | 0.660943 / 0.007541 / 0.029714 | 0.616195 / 0.071479 / 0.027114 | 0.665162 / 0.099276 / 0.000197 |
| 807 | 0.505214 / 0.591295 / 0.103230 | 0.000000 / 0.978986 / 0.279066 | 0.992219 / 0.019352 / 0.000179 |
| 903 | 0.481322 / 0.272375 / 0.103513 | 0.275677 / 0.653342 / 0.159736 | 0.996305 / 0.008516 / 0.000187 |
| 1191 | 0.176092 / 0.822497 / 0.055628 | 0.215745 / 0.781734 / 0.060738 | 0.774604 / 0.227241 / 0.013143 |
| 1479 | 0.616073 / 0.510862 / 0.025246 | 0.541754 / 0.526018 / 0.081804 | 0.809754 / 0.158532 / 0.026983 |

Images: [train native](../runs/local-recursive-cnn-20261011/evaluation32/train-native.png), [probe native](../runs/local-recursive-cnn-20261011/evaluation32/probe-native.png), [train overview](../runs/local-recursive-cnn-20261011/evaluation32/train-overview.png), [probe overview](../runs/local-recursive-cnn-20261011/evaluation32/probe-overview.png), [partial](../runs/local-recursive-cnn-20261011/evaluation32/partial-native.png).

## Next diagnostic

Continue both actual1200 model/Adam/EMA states to6000 with the same loss,
data stream and constant LR. This tests whether the remaining fixed-pair errors
keep decreasing and whether the smaller architecture retains its advantage
with more training. It cannot by itself teach the model the unseen range of
stroke widths or partial-mask conditions; broader synthetic training remains
necessary before any deployment claim.
