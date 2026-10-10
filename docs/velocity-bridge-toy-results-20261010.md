# AV: Direct velocity bridge results

Complete diagnostic, not an adoption candidate. The same 130,628-parameter CNN
was trained from AT's common random initialization for 6,000 updates. Predicting
velocity directly improved the high-noise ambiguous toy, but failed 27 of the
36 predeclared checks. All 12 groups still exceed the mean relative endpoint
RMS limit of .15. No broad-data or production improvement is established.

| 32-step sampler | Mean endpoint RMS, strength .5 (six groups) | Draws within .20, strength .5 |
| --- | ---: | ---: |
| AT endpoint prediction | .855257305 | 0 / 384 |
| AV direct velocity | .485600198 | 0 / 384 |

The mean RMS fell about 43%, and all three native sheets show straighter,
more coherent lines at .5. Background speckle remains. At .1, whole-image
outputs largely favor the two-line mode (nearest-A counts 1/0/0 out of64 at
depths1/2/3). Partial-input outputs sometimes vary between modes but have
shape errors. The protected half is intentionally left rough.

A separate diagnostic of AT on known noise-free mean trajectories found an
average physical endpoint error of .02972 at remaining time .0001, implying
velocity error297.2 after division. These are known analytic paths, not typical
random high-dimensional states; the result demonstrates possible amplification,
not a complete causal explanation of AT's artifacts or of production grayness.
AV removes explicit division but does not solve the restoration problem.

Validation:24 tests pass in29.28s; a preliminary float32 gradient expectation
was too strict (4.55e-6 accumulation error) and its failed log is retained.
Data-generation function bodies and12 prepared minibatches are identical to AT;
initial parameters/configuration are exact and time-zero outputs agree within
1e-5. States6/2000/6000 and all6 raw/EMA exports restore exactly. All108 output
blocks (72 reused AT controls plus36 new) are finite, and protected pixels are
exact. Three fixed sheets were inspected. Final raw6000 was selected in advance.
No checkpoint, EMA, sample or threshold was selected from the results.

- [Depth1](../runs/velocity-bridge-toy-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/velocity-bridge-toy-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/velocity-bridge-toy-20261010/evaluation/depth-3-native.png)
- [All metrics](../runs/velocity-bridge-toy-20261010/evaluation/metrics.json)
- [Final audit](../runs/velocity-bridge-toy-20261010/final-audit.json)

Next bounded question: does condition-preserving minibatch optimal transport
between initial noisy states and the two clean targets improve the learned
velocity field? AV used independent pairing. This changes the training coupling,
not the inference observations, network size or sampling steps. A toy pass would
still need broad synthetic-data validation and independent confirmation.
