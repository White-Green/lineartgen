# Controlled fitting results

All four arms completed the fixed1,200 updates and evaluations at0/400/1200.
**None sufficiently fits even this restricted task in the tested schedule.** This
does not prove a model-capacity limit: it is one short schedule, and the nominally
wider model retains the same narrow output head. No model is adopted.

Training used19 generated targets × clean/tone/blur =57 fixed pairs, plus57
separate local probe pairs. All inputs are synthetic; no campaign development or
prospective confirmation cases were opened. [Fixed plan](learnability-plan-20261008.md).

Final raw means for the blurred inputs (sigma2, opacity.3), excluding the blank,
over18 targets × two noise strengths. Lower halo is better.

| Model | Parameters | Train F1 | Probe F1 | Train halo | Probe halo | Failed train fit conditions /108 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm, clipped MSE | 126,828 | .4310 | .4786 | .2475 | .2295 | 97 |
| Warm, interval MSE | 126,828 | .4305 | .4796 | .2456 | .2328 | 99 |
| Random, small | 126,828 | .0182 | .0192 | .2175 | .2169 | 108 |
| Random, doubled channels | 503,158 | .0372 | .0427 | .2734 | .2683 | 108 |

The failed-fit count covers clean/tone/blur, not only the blur means in the table.
The original warm model's train blur F1 was .2762, halo .0632 and ink deficit .6859.
The conventional warm model ends with better F1 and deficit .1937, but halo grows
to .2475 and clean F1 falls from .8702 to .7297. Recovering more ink is not enough:
the model also produces substantial incorrect ink. Interval MSE does not remove
this tradeoff in the fitting probe. EMA was retained as a diagnostic, not selected
instead of the declared raw primary.

Native pixel crops were actually inspected at all three milestones: nine fixed
geometry kinds in each split, blur/strength.5, six sheets total. These show
continuous model values quantized to grayscale, before the BGRA alpha cutoff.
Warm models grow broad gray branches and wrong connections; thin curves remain
incomplete, and parallel strokes/gaps remain incorrectly joined. Random models
are still noisy after1,200 updates. This is not a successful overfit, nor evidence
that additional training or a different initialization could never fit the task.

There is a specific limitation in the capacity comparison. The last decoder
always returns two channels. The head is3→3→3→1, with196 parameters and a7×7
direct receptive field, in both the small and doubled-width networks. Doubling
encoder widths does not widen this path. Parameter-movement checks show updates
throughout both networks; the result is not explained by all weights remaining
frozen. Whether this output path limits useful detail remains a hypothesis.

Separately, the [residual-scale audit](residual-conditioning-diagnostic-20261008.md)
shows that restoring rough inputs at tiny nonzero strength requires very large
residual-head values under sqrt(t) scaling. The next fixed local comparison caps
that scaling below.05, under both clipped and interval MSE. It changes neither
the completed four-arm experiment nor the production defaults. Its purpose is
mechanism diagnosis; a local improvement still needs a broader quality study.

Artifacts are in `runs/learnability-20261008`. The completion audit verifies211
pinned source files,7 settings, the source archive,24 raw/EMA model triplets,
7,296 finite predictions, and exact protection on partial masks. Full grouped
states restore exactly at1,200. All24 six-depth preflight updates passed;14
targeted tests passed. The parent's259-test suite was inherited through209
byte-identical source files, not rerun in this checkout. Source pin is
`fda81949bc41028dda3530f614b6f0d823828b90`; later documentation/audit helpers do not
change the pinned training code. The prospective confirmation set stays unopened.
