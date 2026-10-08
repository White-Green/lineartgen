# Broader detail-head result

Both width16 models completed2,000 updates and stop here. Neither advances to10k.
No replacement is adopted and the sealed confirmation draws remain unopened.
This follows the [prior plan](detail-head-broad-plan-20261008.md): numerical
screening and actual visual support were both required to continue.

| Model/mode | Primary halo reduction, real / geometry | Screen failures | Strict final failures |
| --- | --- | --- | --- |
| Strokes EMA |7.35% /6.75% |0 |20 |
| Strokes raw |17.42% /20.06% |3 |51 |
| Mixed EMA |−41.83% /−32.64% |2 |70 |
| Mixed raw |−30.85% /−10.13% |1 |37 |

These gains compare with the original campaign control. Against the matched
guided control, the strokes EMA detail branch instead increases primary halo
by1.28% real /0.85% geometry. Strong geometry blur halo increases3.31%, and its
F1 changes−.00229. Primary F1 gains are small, .00406/.00461. The mixed EMA branch
also increases primary halo against its guided counterpart, by2.04%/.85%.
Paired case tables and clustered intervals are retained in each root's
`analysis-with-baselines-2000.json`; these are reused development cases, not
independent confirmation.

All20 predeclared native sheets were inspected: ten for each family, covering
EMA/raw geometry, real clean art with synthetic corruption, repeated inference
and a partial-mask blank. Thin strongly blurred lines remain mostly absent;
parallel lines merge and intended gaps fill. Corners and real-art outlines retain
gray branches and fringes. Five iterations often darken existing wrong bridges.
The mixed branch produces visible gray marks on the blank panel; the strokes
panel has sparse faint speckles. The detail branch does not show persuasive
visual improvement over its matched guided control. Therefore the strokes arm
stops despite passing the permissive numerical screen. This is not a claim that
its unseen10k endpoint must fail, or that a detail branch cannot help other
training regimes.

Strokes strict EMA failures comprise insufficient primary halo gain, ten ink
deficit limits, seven cut limits and two distant-background limits. Mixed EMA
screen failures are insufficient halo gain and strong-blur bridges +.08333
against a .03 tolerance. Its70 strict failures largely concern background,
F1, bridges and width. Overlapping strata are not independent defects.

The frozen numerical source is29dde4b1a12ab97d5f1a1323befa07d77af756af.
Both restored2k checkpoints have finite model/optimizer/EMA state and exact raw
and EMA snapshot agreement. Initial forward and historical control replay audits
passed;270 inherited tests cover unchanged numerical files. All fixed1,562-case
EMA/raw evaluations, repeat diagnostics, blank panels and supplements completed.
Reviews with image hashes are in each root's
`visual-review/step-2000/review.json`. These images use float inputs formatted on
white; they are not actual Krita BGRA-input or Windows DirectML quality checks.

The earlier local57-pair improvement remains valid but did not establish this
broader benefit. Its strongest gains were at noise.5, with much smaller blur
gains at.1. The [conditional-path diagnostic](conditional-path-diagnostic-20261008.md)
motivates a separate matched comparison of intermediate training states; it is
not evidence that such states caused these failures.
