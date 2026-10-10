# AX: Boundary-focused training-time results

Complete diagnostic, not an adoption candidate. Both-mode coverage passes in
all12 groups for the first time in the learned toy series. Shape fidelity is
still insufficient:24/36 fixed criteria fail (all mean-RMS and within-.20
criteria). More balanced outputs do not by themselves solve grayness.

| Method,32 steps | Mean RMS at strength .5 | Weak-noise whole-image A counts, depths1/2/3 |
| --- | ---: | --- |
| AW uniform continuous time | .378466681 | 0 / 0 / 0 |
| AX boundary-focused time | .463701295 | 9 / 25 / 30 |

Each count is out of64 draws. At strength .5, none of384 AX samples is within
RMS .20. AX increases variation but sometimes leaves gray central gaps,
irregular transitions or background speckle. All three fixed sheets were
inspected. The deliberately rough protected halves remain exact.

Only the training-time distribution changed. In192 audited batches (3,072
samples), the exact-zero count stays798. Nonzero times below.01 rise24→138;
times above.99 rise27→192. Initial noise, sources, masks, teachers, conditional
OT assignment and all other numerical function bodies are unchanged. CPU
and GPU audits pass.35 tests pass in45.23s. Full states6/2000/6000 and all6
raw/EMA exports are exact;180 saved output blocks are finite with exact
protected pixels. Final raw6000 and first4 displayed draws were predeclared.

- [Depth1](../runs/boundary-time-bridge-toy-20261010/evaluation/depth-1-native.png)
- [Depth2](../runs/boundary-time-bridge-toy-20261010/evaluation/depth-2-native.png)
- [Depth3](../runs/boundary-time-bridge-toy-20261010/evaluation/depth-3-native.png)
- [Metrics](../runs/boundary-time-bridge-toy-20261010/evaluation/metrics.json)
- [Audit](../runs/boundary-time-bridge-toy-20261010/final-audit.json)

Next bounded test: continue both AW and AX from their actual6,000-step full
states to30,000 updates, with equal additional compute and unchanged1e-5 LR
floor after their original schedule. Compare final raw30,000 for both, without
selecting intermediate checkpoints. This separates remaining fitting duration
from the observed uniform-versus-boundary time tradeoff. No broad-data claim,
production change or independent confirmation has been made.
