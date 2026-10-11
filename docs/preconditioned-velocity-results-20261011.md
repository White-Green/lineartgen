# BU: variance preconditioning increased background noise in the two-shape toy

Both matched arms completed6000 updates. Each fails24/36 criteria, versus27 for
the frozen AV control, but their actual shape errors are substantially worse.
The smaller failure count comes from nearest-label frequencies, not accurate
samples. Neither arm produces any draw within the required.20 normalized RMS
at32 steps. No candidate and no broad extension of these weights.
[Frozen plan and derivation](preconditioned-velocity-plan-20261011.md).

| Mean over six depth/mask groups at32 uniform Euler steps | AV control | Preconditioned physical MSE | Preconditioned normalized MSE |
| --- | ---: | ---: | ---: |
| Strength.1 nearest-mode RMS | .248287 | .926214 | 1.018587 |
| Strength.5 nearest-mode RMS | .485600 | .741117 | 1.063457 |
| Strength.1 fraction within.20 | .460938 | 0 | 0 |
| Strength.5 fraction within.20 | 0 | 0 | 0 |

All12 nearest-A frequencies of each new arm fall inside.1–.9, yet the samples
are not close to either teacher. This is why label balance alone is insufficient.
Physical output excursions also grow; final clipping does not fix the visible
noise. All three native sheets were inspected:
[depth1](../runs/preconditioned-velocity-20261011/evaluation/depth-1-native.png),
[depth2](../runs/preconditioned-velocity-20261011/evaluation/depth-2-native.png),
[depth3](../runs/preconditioned-velocity-20261011/evaluation/depth-3-native.png).
Both variants leave a granular background and imperfect bars/gaps; normalized
loss does not make them cleaner. The protected halves are preserved.

The coefficient algebra passed unit-variance and covariance checks under its
declared Gaussian variance model, including both time endpoints. That model is
an approximation to source-conditioned lineart residuals, not an empirical fact
about white paper or thin lines. This run rejects the particular scaling recipe
and6000-update setup; it does not show that all preconditioning is ineffective.
The reference idea is input/output/loss scaling from
[Karras et al., section5](https://papers.nips.cc/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf);
our conditional linear-bridge coefficients are not copied EDM denoiser equations.

Audit:20 tests passed (19 in45.50s and1 recursion-gradient test in1.53s).
The optional raw-head accessor retains default physical residual outputs.
Initial parameters match AV and are identical between arms; initial
preconditioned functions match each other, not the AV velocity function.
Fresh Adam/EMA, first6 updates replay exactly, states0/6/2000/6000 restore fully,
all16 raw/EMA exports match. All6000 input journals have distinct hashes and
balanced8/8 teacher draws; each recursion depth receives2000 updates.
All36 AV control output blocks (physical and clipped) and metrics reproduce
bitwise. All108 evaluated blocks are finite with protected pixels exact.
[Final audit](../runs/preconditioned-velocity-20261011/final-audit.json) binds
the source, configurations, data, parameters, outputs and three reviewed sheets.
Numerical commit `29023e8`.

Before freezing, preparation rejected an unsupported direct-output/progress
configuration. It was replaced with the optional raw-head accessor without
weakening configuration validation. The preserved test failures also caught
NNX keyword resolution and a1.88e-7 float32 reduction discrepancy. All were
resolved before this run's first update. This is still only two memorized
teachers with new Gaussian draws, not new shapes or confirmation. No plugin,
runtime default, production model or deployment change.
