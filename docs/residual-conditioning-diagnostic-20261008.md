# Residual scale in the controlled fitting task

The existing residual head uses `prediction=clip(x_t+sqrt(t)*F, -1, 1)`.
For input `x_t=sqrt(1-t)*rough+sqrt(t)*epsilon`, an exact unsaturated target needs

`F*=(clean-sqrt(1-t)*rough)/sqrt(t)-epsilon`.

Denoising a clean image has a small deterministic correction at low t. Restoring
a faint/blurred rough image still requires a large contrast/shape correction as
t approaches zero. The current target is fully clean at every editable t>0,
while t=0 is protected rough input. The required head therefore has a singular
limit when rough differs from clean. This is a parameterization/target-scale
observation, **not evidence that it causes the measured quality failures**.

An algebraic audit on all fixed fitting targets' black-core pixels gives the
following 95th percentile absolute deterministic head values for blur sigma2,
opacity .3. There is no model fitting or evaluation-set use in this audit.

| t | Existing sqrt(t) scale | Hypothetical sqrt(max(t,.05)) scale |
| ---: | ---: | ---: |
| .001 | 57.67 | 8.16 |
| .01 | 18.20 | 8.14 |
| .1 | 5.63 | 5.63 |
| .5 | 2.24 | 2.24 |

The proposed floor here is only an algebraic comparison. A learned model can
rescale its head, so this does not prove a capacity increase or quality benefit.
The loss already has a separate noise-weight floor .05; it does not change the
output head's sqrt(t) multiplier. Any output-floor experiment would need explicit
identity at t=0, paired baseline updates, initial equivalence for t≥.05,
independent confirmation and the unchanged quality safeguards. Neither the
current fitting probe nor production models/defaults were changed by this audit.

The fixed strengths were .001/.003/.01/.03/.1/.3/.5/.9 and hypothetical floors
0/.05/.1/.25. All reconstructed deterministic predictions agree with their
targets within3e-7. Results, plan and input/script hashes are in
`runs/learnability-20261008/residual-conditioning/`. The executed script is kept
as `residual-conditioning-script-as-run.py`; the committed version only wraps one
long string and has the same parsed AST. The ongoing four-arm fitting experiment
will be completed unchanged before interpreting this as a next intervention.
