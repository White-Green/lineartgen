# AB: paper-clearance loss on broader data — completed

**The mechanism generalizes partially, but36/229 fixed checks fail.** Relative to
matched Y, clearance reduces halo and excessive width and lowers some bridge
rates while keeping geometry blur cuts at0. It does not reliably restore intended
separation, and is not an adoption candidate.

Only gap_clearance_weight0→.1 (radius3) changes from Y. Same130,605-parameter
source-feature model, E initialization, fresh optimizer/EMA, narrower ordinary
geometry widths and strong corruption mixture. Raw4000 was fixed in advance;
AA/Z fitting weights and intermediate/EMA selection were not used.

Whole-image strength.5:

| Metric | R | Y | AB |
| --- | ---: | ---: | ---: |
| Corpus blur F1 | .936629 | .931999 | .933204 |
| Corpus blur ink deficit | .070756 | .049849 | .061188 |
| Corpus blur halo | .008512 | .009048 | .007377 |
| Corpus strokes F1 | .843711 | .846510 | .852321 |
| Corpus strokes ink deficit | .031481 | .027447 | .036191 |
| Corpus strokes halo | .095144 | .086276 | .077641 |
| Geometry blur F1 | .824615 | .871831 | .878952 |
| Geometry blur ink deficit | .108294 | .025878 | .030629 |
| Geometry blur halo | .025011 | .032969 | .029062 |
| Geometry blur width MAE,px | .589909 | .713503 | .661987 |
| Geometry blur cuts | .083333 | .000000 | .000000 |
| Geometry blur bridges | .125000 | .333333 | .270833 |
| Geometry strokes F1 | .682142 | .669175 | .677945 |
| Geometry strokes halo | .161461 | .173564 | .163002 |
| Geometry strokes width MAE,px | 2.339986 | 2.593773 | 2.457403 |
| Geometry strokes cuts | .010417 | .000000 | .000000 |
| Geometry strokes bridges | .395833 | .458333 | .416667 |

Failures:16 grouped F1,8 bridges,6 ink deficit,4 width,1 relative halo and1 primary
geometry F1. Black interiors retain mean/minimum alpha1. Blank maximum .0497301
passes .05 narrowly, unlike Y's .126449. This is not evidence of a robust margin
on unseen blank boundaries.

Local clearance-only failures14 at1k→9 at4k; final total22 also includes Y's13.
Raw4000 local blur.5 F1=.845488 TRAIN/.844065 PROBE, deficit=.046318/.065310,
halo=.022734/.027832. All8 local and6 broader final sheets were viewed and
hash-bound. Fine curves/corners are darker than R; AB slightly reduces Y's fringe.
Repeated-stroke widths, dense tire separations and intended parallel/gap examples
remain wrong. Black fields survive and displayed blank crops look white.

23 tests passed. All166 Y teacher batches replay before training; all4000 actual
teacher/depth records match Y exactly. All6 prepared depth inputs are bitwise
equal, actual6/1k/4k full states and four raw/EMA exports restore exactly.192 local
primary,1368 local broad and6264 main outputs are checked; all E/R/Y main reference
metrics reproduce exactly. Numerical commit`c75c7c6439c06ce8fef8a825fb0ce5fab1fe9de9`;
evidence in`runs/gap-clearance-generalization-20261010/final-audit.json`.

The separate [32-output actual BGRA diagnostic](bgra-clearance-quality-results-20261010.md)
confirms modest Y→AB halo improvements and persistent gaps/width failures. Input
scale1.1 still removes faint-line information; scale1.0 retains it but does not
fix separation. That diagnostic also viewed three final native sheets.

Synthetic rough only,44 existing validation corpus identities; no independent,
sealed, production or plugin changes. The next controlled hypothesis is data
balance: retain this loss and model, increase procedural geometry20%→50% while
keeping nominal coverage10% and decreasing native teachers70%→40%. E learned
fine geometry under100% procedural training; the present mixed training still
loses some of that quality. Earlier P's geometry study used a different loss,
width range and corruption distribution and does not resolve this interaction.
