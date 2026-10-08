# Conditional-path fitting result

Both matched models completed1,200 updates. The bridge candidate fails the fixed
local screen in18 comparisons and is rejected. All ten final native sheets were
actually inspected. No broader training, replacement model, independent seed or
sealed new confirmation draw follows from this result.

| Raw model / iterations | Blur F1, train / probe | Blur halo, train / probe |
| --- | --- | --- |
| Historical detail16 /1 |.53035 / .58043 |.19525 / .18899 |
| Matched progress rough /5 |.18161 / .21869 |.31288 / .27120 |
| Progress bridge /5 |.21152 / .24879 |.33371 / .28501 |

Bridge versus the matched rough five-step arm gains only .02990/.03010 blur F1,
while increasing halo by .02083/.01382 and ink deficit by .01820/.01467. Against
the historical one-step reference it loses .31884/.33164 blur F1, increases halo
.13846/.09603 and loses .55566/.51306 clean F1. At one step, bridge also loses
.09394/.06727 clean F1 versus the matched rough arm. It therefore fails on large
quality regressions, not just a near-threshold training F1 gain.

Native repeat sheets show the central failure: faint gray spurs become dark
branching networks as iterations increase. Thin lines, corners, gaps and tapers
are especially affected. Parallel lines fuse or gain unwanted connections.
Whole blank inputs also acquire invented structures, including dark branches
at noise.1 and large networks at.5. These defects already occur in the historical
detail16 reference at multiple iterations; the bridge did not repair them. EMA
does not rescue the visual result. Protected portions remain unchanged, but
active portions near partial-edit boundaries develop spurious marks.

The final review and image hashes are in
`runs/conditional-path-fitting-20261008/evaluation/step-1200/visual-review/review.json`.
An additional descriptive400-update sheet was inspected before the final result;
it did not change the prescribed1,200 updates or gates. All panels use stored
float inputs, not the BGRA-input API or Windows Krita GUI.

Validation remains separate from quality.279 tests plus two focused summary
tests passed. Both candidate GPU preflights and the historical six-update replay
restored full state exactly; all36 initial forward comparisons match. Actual
updates1/400/1,200 have exact model/optimizer/EMA restore. The0/400/final/reference
evaluations cover13,680 saved predictions, and all historical one-step reference
arrays match bitwise. Protected errors are zero. The frozen228 numerical files
and five settings are unchanged. The separate
[partial-summary correction](fitting-partial-summary-fix-20261008.md) fixed a
`None` comparison after inference, without changing saved predictions or gates.

The oracle derivation and runtime agreement remain valid, but do not establish
learnability under this model, objective and budget. In particular, retained
teacher-dependent tone weights are not plain flow matching, and intermediate
recursion levels are not separately supervised. This rejects this specified
balanced bridge experiment, not every possible bridge objective or architecture.
The independent [tone-weight comparison](tone-balance-diagnostic-20261008.md)
tests a different hypothesis; no new quality claim is made for it yet.
