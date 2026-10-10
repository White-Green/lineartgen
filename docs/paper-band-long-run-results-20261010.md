# AP: AE4000→16000 continuation — insufficient for adoption

The declared final raw16000 model fails56/229 unchanged E/R-based quality checks
(AE4000 failed51/229). It improves some halos and separations, but does not retain
ink consistently enough. No candidate is adopted and no intermediate/EMA is selected.

| Whole-image strength.5 | AE4000 F1 / deficit / halo | AP16000 F1 / deficit / halo |
| --- | --- | --- |
| Corpus blur | .930713 / .074595 / .005860 | .929586 / .084527 / .004587 |
| Corpus strokes | .861751 / .042767 / .061096 | .867361 / .050324 / .055586 |
| Geometry blur | .883581 / .034583 / .025963 | .887237 / .039722 / .021848 |
| Geometry strokes | .688099 / .016403 / .149484 | .696334 / .021637 / .138051 |

Corpus blur halo falls21.7%, and strokes halo9.0%, while ink deficits increase.
Geometry blur width MAE .627790→.604936px and bridge rate .270833→.1875 improve,
but F1 remains below.90. Geometry strokes width MAE2.277171→2.148566px and bridges
.416667→.395833 remain poor. All geometry cuts in these whole-strength.5 groups
are0. Edited black717/727 mean/minimum opacity are now1.0. Blank maximum rises
.039015→.0503351986, narrowly failing the fixed.05 limit. That float artifact is
not visible in the thresholded blank display; do not call it a visually observed mark.

Failures:27 ink-deficit,18 F1-preservation,6 bridging,3 width,1 primary geometry
blur F1,1 blank. Their count is not a scalar quality score. Local checks against
AE4000 have2/10/7 failures at4000/8000/16000; these local controls differ from earlier
studies, so do not rank unrelated experiments by those counts.

All14 final sheets were inspected. Thick corpus outlines are often good, and the
broad-panel thick taper is black (unlike AO's unfitted fixed-pair taper). Fine
hatching and blur curves still have gray/uneven edges; strong-stroke inputs remain
far too wide, and close parallel/endpoint gaps merge. Partial black interiors look
solid. Protected portions in partial-thin sheets intentionally remain rough.

28 tests passed. The evaluation-control naming bug was repaired in a separate
worktree; the six initial updates replayed bitwise-identically in model, optimizer,
EMA, data records and initial predictions.166 prior teacher hashes/data-resume
checks passed. All12000 new sampling records and486 journal entries match expected
teacher/depth counts and constant LR1e-5. States4000/4006/8000/16000, four raw/EMA
exports, local64/broad456 controls and all1566 E/R/AE reference metrics were checked.
Main evaluation contains6264 model/condition rows; it uses44 previously seen corpus
validation identities and synthetic rough inputs, no final-source/sealed cases.

- [Final audit](../runs/paper-band-long-run-20261010/final-audit.json)
- [Corpus examples](../runs/paper-band-long-run-20261010/development/native-review/real.png)
- [Blur examples](../runs/paper-band-long-run-20261010/development/native-review/blur-05.png)
- [Stroke examples](../runs/paper-band-long-run-20261010/development/native-review/strokes-05.png)
- [All failures](../runs/paper-band-long-run-20261010/development/summary.json)

This rules out adopting this particular duration extension. It does not show that
more training, other schedules or more capacity can never help. The model remains
130605 parameters; full-width broad-data capacity has not yet been compared with
this exact loss/data protocol. Production/plugin/defaults remain unchanged.
