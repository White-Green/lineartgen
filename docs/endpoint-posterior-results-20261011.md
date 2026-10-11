# BY: remaining endpoint error concentrates near the beginning of the path

The analytical two-teacher reference separates early information from late
reconstruction. BX's noisy anchor closely follows the reference after tau .05–.1
but still lags near tau .002–.01. Its initial prediction also has a consistent
A-directed bias. These observations motivate testing early continuous-time
coverage; they do not prove the cause of free-running mode collapse.
[Frozen plan](endpoint-posterior-plan-20261011.md).

| Strength / tau | Fraction with reference posterior outside [.1,.9] | BV noisy RMS vs reference | BX noisy RMS vs reference |
| --- | ---: | ---: | ---: |
| .1 / 0 | 0 | .196426 | .143831 |
| .1 / .002 | .347656 | .587095 | .597783 |
| .1 / .005 | .949219 | .852495 | .804721 |
| .1 / .01 | 1 | .766397 | .509287 |
| .1 / .02 | 1 | .540422 | .123364 |
| .1 / .1 | 1 | .078254 | .017090 |
| .5 / 0 | 0 | .283210 | .203729 |
| .5 / .01 | .414062 | .589911 | .591844 |
| .5 / .02 | .890625 | .751663 | .511161 |
| .5 / .05 | 1 | .393729 | .031285 |
| .5 / .1 | 1 | .061560 | .008448 |

Means combine depth1/3 and whole/partial groups,64 paths/group. This RMS is
against the conditional mean on teacher-interpolated inputs, not the previous
free-running nearest-mode RMS. The posterior uses known prototypes and all
independent scale observations; the CNN only receives states, guides and
compressed recursive inserts. The reference is not a claim of attainable
architecture performance. Float32 generated states are evaluated against the
underlying Gaussian path formula in float64.

A posthoc projection of the saved tau0 predictions onto half(A-B) finds BX
noisy's mean signed bias at strength .1 is +.1046 to+.1800, with per-group
standard deviations .0195–.0390. The theoretical equal-prior mean has score0.
All four groups lean toward A before the sampled paths separate. This is an
association with later collapse, not a causal intervention or an acceptance
criterion. [Posthoc values](../runs/endpoint-posterior-20261011/evaluation/posthoc-initial-projection.json).

All eight native sheets were inspected. At tau0, a gray middle between the two
shared black sides is the correct conditional mean. At strength .1/tau .01,
the reference has selected a teacher while predictions still retain an
uncertain gray middle. BX noisy catches up later; BX source shows uneven and
hollow bars. Near tau .95 every model follows the supplied teacher.
These images are **teacher-conditioned intermediate states**, not final sampler
outputs, and cannot establish real restoration quality:
[whole .1/A](../runs/endpoint-posterior-20261011/evaluation/d3-p0-t0.1-teacher-0-native.png),
[B](../runs/endpoint-posterior-20261011/evaluation/d3-p0-t0.1-teacher-1-native.png),
[whole .5/A](../runs/endpoint-posterior-20261011/evaluation/d3-p0-t0.5-teacher-0-native.png),
[B](../runs/endpoint-posterior-20261011/evaluation/d3-p0-t0.5-teacher-1-native.png),
[partial .1/A](../runs/endpoint-posterior-20261011/evaluation/d3-p1-t0.1-teacher-0-native.png),
[B](../runs/endpoint-posterior-20261011/evaluation/d3-p1-t0.1-teacher-1-native.png),
[partial .5/A](../runs/endpoint-posterior-20261011/evaluation/d3-p1-t0.5-teacher-0-native.png),
[B](../runs/endpoint-posterior-20261011/evaluation/d3-p1-t0.5-teacher-1-native.png).

Audit: three reference-math tests passed in1.11s: equal initial priors, exact
Gaussian density-ratio agreement, additive scale evidence, ignored protected
evidence and near-teacher limit. Zero neural updates;384 model rows across96
input groups; all finite and protected pixels exact. Eight visual hashes and
both parents' final checkpoints are bound in the
[final audit](../runs/endpoint-posterior-20261011/final-audit.json).
Numerical commit `1a31953`. No new teachers, independent seed, sealed conditions,
production model, plugin or runtime-default changes.
