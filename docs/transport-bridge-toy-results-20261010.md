# AW: Conditional minibatch transport results

Complete diagnostic; not an adoption candidate. Changing only training target
assignment to exact conditional minibatch OT reduced high-noise toy error by
another22% versus AV, but22/36 predeclared checks still fail.

| 32-step sampler | Mean nearest-endpoint RMS at strength .5 |
| --- | ---: |
| AT endpoint | .855257305 |
| AV independent velocity | .485600198 |
| AW condition-preserving OT velocity | .378466681 |

At strength .5, none of384 samples are within relative RMS .20. At .1, all
whole-image samples select the two-line mode (nearest-A0/64 for all3 depths).
Their mean errors .15420/.15178/.15283 are close to but above the fixed.15
criterion. Partial .1 at depths2/3 passes the RMS criterion (.14345/.14556),
but mode balance fails. Criteria were not changed. All three native sheets
were inspected: lines improve, but background speckle and mode bias persist.

The transport changes1,296 of3,072 audited labels, preserving the number of
A/B targets in every strength/mask group. The192 minibatches match SciPy's
float64 Hungarian optimum; total displacement cost falls1,723,605.265 to
1,720,849.865. The small relative cost reduction includes large common source
restoration terms. This is minibatch root-scale OT, not population or joint
multiscale OT. Smaller-scale noise remains independent.

30 tests pass in38.16s, including matching the unchanged AV update on the same
prepared tensors and full-state/next-update restore. CPU and GPU data audits
pass. An initial nested JIT changed coarse-input rounding by1.19e-7; separating
the old preparation call from the assignment graph restores exact input
identity. The failed preflight logs are retained. Full states6/2000/6000 and
6 raw/EMA exports are exact. All144 output blocks are finite, with protected
pixels exact, including108 hash-bound cached controls. Final raw6000 and the
first4 displayed draws were fixed in advance.

- [Depth1](../runs/transport-bridge-toy-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/transport-bridge-toy-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/transport-bridge-toy-20261010/evaluation/depth-3-native.png)
- [Metrics](../runs/transport-bridge-toy-20261010/evaluation/metrics.json)
- [Audit](../runs/transport-bridge-toy-20261010/final-audit.json)

Next test: concentrate some training samples near both ends of the path. The
current continuous time is uniform (besides25% at zero). With weak initial
Gaussian noise, choosing a mode occurs near the start; removing residual noise
requires precision near the end. Keep the same zero-time samples and warp the
remaining times with sin²(pi*t/2), leaving labels, assignment and initial noise
unchanged. This tests temporal coverage, not a new loss term or larger model.
