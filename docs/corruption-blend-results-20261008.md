# Small corruption blends: two strict-final development passes

Select `corruption_blend0625` EMA (alpha=1/16) for independent fine-tuning seed
confirmation. It has zero failures under the unchanged strict final numerical
criteria and the highest predeclared halo ranking among eligible candidates.
It is not adopted: faint-line erasure and topology errors remain in images,
and raw/EMA sensitivity is substantial.

Source `62b464bbea032618623a5b8a88decebae48fe9c8` sealed 199 files and 13 settings
in `runs/corruption-blend-20261008`. The plan hash is
`cebad9c40970bd8105f40e66190a22ad18e6d3e4cb2602aced1186e2b59127c2`.
All three ratios were fixed before their first inference: 1/16, 1/8 and 1/4 of
the mixed-corruption endpoint, with the remainder guided strokes-trained ink
guard. Both endpoints are existing 2k checkpoints from the same original 100k
EMA, seed, optimizer schedule and clean/noise draws. Only their rough corruption
families differ. Each blend is one 126,828-parameter model, not an image ensemble;
no training updates or optimizer history are fabricated for these checkpoints.

This new inference-only experiment explicitly applied **strict final** gates at
the fixed 2k checkpoint age. It does not undo either endpoint's prior rejection
or imply a 10k result. No alpha or acceptance threshold changed after evaluation.

| Mixed weight | EMA real halo reduction | EMA geometry halo reduction | Strict final failures EMA / raw |
|---|---:|---:|---:|
| 6.25% | 14.306% | 14.443% | 0 / 35 |
| 12.5% | 11.680% | 12.221% | 0 / 22 |
| 25% | 6.263% | 7.662% | 5 / 1 |

Halo comparisons use conventional control at the same 2k step. Raw halo gains
are 25.857%/28.846%, 22.943%/26.791%, and 16.902%/22.582%, but retention regresses;
it is not substituted for the predeclared EMA model. Overlapping failure strata
are not independent defect counts. Real cases use synthetically degraded real
clean lineart, not human rough sketches.

For the selected EMA candidate, paired excess-halo opacity deltas are
−.021227 (95% source-cluster interval [−.024192, −.018240], 46 cases / 12 sources)
and −.030047 ([−.034344, −.025428], 128 cases / 32 geometry groups). These intervals
condition on fitted models and do not account for adaptive model selection or
training-seed uncertainty.

Eight fixed native-pixel sheets were actually inspected. Gray reductions are
modest in these crops. Thin strongly blurred lines frequently disappear, nearby
lines merge and deliberate gaps close in some cases; repeated inference can
strengthen wrong connections. There is no new gross collapse relative to the
paired control. Passing relative safeguards does not mean these absolute
shortcomings are solved. Maximum blank active mean opacity is .001241/.001456/
.001950 for EMA and .000419/.000477/.000586 for raw.

Three existing parameter-blending tests passed; unchanged underlying training
and runtime code had passed the preceding 245-test suite. Actual guided GPU
alpha0/1 parameter/forward comparisons were exact in all 36 tests (three cases,
both endpoints/modes, 1/2/5 iterations). All six constructed models restored
exactly after serialization. Eight metric files are complete and finite with
zero protected-region errors. Their hashes and review are retained alongside
the construction and endpoint audits. Evaluation ran approximately 11:53–12:00
UTC on 2026-10-08.

Next: train both endpoint recipes for exactly 2k under seed20261006 from the same
100k EMA and construct the fixed 1/16 blend. Compare with that seed's matched
control. Only if it passes may selected hashes be sealed and the previously
declared fresh synthetic conditions generated. No fresh conditions have yet
been opened, and production weights remain unchanged.
