# AC: increased geometry exposure — completed

**41/229 fixed checks fail; not adopted.** Raising procedural teachers20%→50%
with the same paper-clearance loss reduces some repeated-stroke halo/width, but
does not solve separation or improve blurred geometry substantially. Ink-deficit
failures increase compared with AB. More general geometry exposure alone is not
a sufficient solution under this fixed4k protocol.

Only geometry probability.2→.5 and conditional coverage probability.125→.2 change.
Nominal coverage remains10%; realized teachers are7,979 geometry,6,394 native and
1,627 coverage out of16,000. Same E initialization/fresh optimizer/EMA,model,
all losses,strong corruption mixture,width distribution and4k schedule as AB.

Whole-image strength.5:

| Metric | AB | AC |
| --- | ---: | ---: |
| Corpus blur F1 | .933204 | .933454 |
| Corpus blur ink deficit | .061188 | .068863 |
| Corpus blur halo | .007377 | .007001 |
| Corpus strokes F1 | .852321 | .856223 |
| Corpus strokes ink deficit | .036191 | .043320 |
| Corpus strokes halo | .077641 | .072262 |
| Geometry blur F1 | .878952 | .879378 |
| Geometry blur ink deficit | .030629 | .034696 |
| Geometry blur halo | .029062 | .030360 |
| Geometry blur width MAE,px | .661987 | .665224 |
| Geometry blur cuts | .000000 | .000000 |
| Geometry blur bridges | .270833 | .270833 |
| Geometry strokes F1 | .677945 | .683250 |
| Geometry strokes ink deficit | .012602 | .017923 |
| Geometry strokes halo | .163002 | .155785 |
| Geometry strokes width MAE,px | 2.457403 | 2.330561 |
| Geometry strokes cuts | .000000 | .000000 |
| Geometry strokes bridges | .416667 | .437500 |

Failures comprise16 grouped F1,13 ink deficit,7 bridges,2 relative halo,2 width
and1 primary geometry F1. The checks remain against E/R, with AB as the matched
descriptive control. Black interiors retain mean/minimum alpha1. Blank maximum
.0349112 passes.05. No intermediate/EMA selection was made.

Local balance-only failures10 at1k→12 at4k (final21 includes AB's9). Local final
blur.5 F1=.836956 TRAIN/.836665 PROBE,deficit=.050869/.069422,
halo=.024005/.029016. All8 local and6 broader sheets were viewed and hash-bound.
Balance looks close to clearance: thin blurred curves are darker than R but still
too wide/gray in some conditions; close pairs and intentional gaps merge. Dense
tire hatching and repeated-stroke widths remain wrong. Main contours,thick taper,
black interiors and displayed blank regions survive.

23 tests passed. Before training, all166 AB teacher hashes were replayed and166
new hashes bound. Among664 audited items,133 old geometry,250 native and40 coverage
items remain identical;241 change. All6 depths preserve masks,protected targets,
target/source pyramids and Gaussian draws (residual roundoff<=4.77e-7). All4000
actual teacher-kind/depth records match the expected schedule; depth records match
AB and all166 actual teacher hashes match the audit. Actual6/1k/4k full states,
four raw/EMA exports and E/R/AB evaluation references reproduce exactly.
192 local primary,1368 local broad and6264 main outputs were evaluated.
Numerical commit`b8c67caa3df95848be1b8e0083435eaecd1cdfcc`; evidence in
`runs/clearance-geometry-balance-20261010/final-audit.json`.

Synthetic rough only and the same44 validation corpus identities. No independent,
sealed,production or plugin changes. The optional line-width preference question
has not changed the target or thresholds; evaluation continues to preserve the
original teacher width.
