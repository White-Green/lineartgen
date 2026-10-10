# AE: broad near-line paper-band loss — completed

**51/229 checks fail; not adopted.** The additional paper-band term lowers halo
and some repeated-stroke width errors, but increases ink-deficit failures. The
near-line paper mechanism generalizes partially; this fixed balance of losses
does not preserve the ink sufficiently. Deliberate gaps and close pairs still merge.

Only paper_band_weight0→.1 differs from AB. Same E initialization,fresh optimizer/
EMA,130,605 parameters,70% native/20% geometry/10% coverage,strong corruption.5,
width max6,sampling and4k schedule. Fixed final raw4k,no intermediate/EMA selection.

Whole-image strength.5:

| Metric | AB | AE |
| --- | ---: | ---: |
| real blur F1 | 0.933204 | 0.930713 |
| real blur ink deficit | 0.061188 | 0.074595 |
| real blur halo | 0.007377 | 0.005860 |
| real strokes F1 | 0.852321 | 0.861751 |
| real strokes ink deficit | 0.036191 | 0.042767 |
| real strokes halo | 0.077641 | 0.061096 |
| geometry blur F1 | 0.878952 | 0.883581 |
| geometry blur ink deficit | 0.030629 | 0.034583 |
| geometry blur halo | 0.029062 | 0.025963 |
| geometry blur width MAE,px | 0.661987 | 0.627790 |
| geometry blur cuts | 0.000000 | 0.000000 |
| geometry blur bridges | 0.270833 | 0.270833 |
| geometry strokes F1 | 0.677945 | 0.688099 |
| geometry strokes ink deficit | 0.012602 | 0.016403 |
| geometry strokes halo | 0.163002 | 0.149484 |
| geometry strokes width MAE,px | 2.457403 | 2.277171 |
| geometry strokes cuts | 0.000000 | 0.000000 |
| geometry strokes bridges | 0.416667 | 0.416667 |

Failures comprise21 grouped F1,19 ink deficit,8 bridges,2 width,and1 primary
geometry F1. No halo guard fails. Blank maximum.0390153 passes.05. Black interior
means .994975/.999995 pass.95,but case717 minimum.908916 shows a small uneven
lighter patch absent from AB; the average alone does not imply perfect black.

Local candidate failures7 at1k and7 at4k; final16 includes AB's9. Local blur.5
TRAIN/PROBE F1=.851804/.852455,deficit=.054094/.073929,halo=.019309/.023827.
All8 local and6 main final sheets were actually viewed and hash-bound. Fine
blurred curves remain too gray or wide; parallel1287/gap1383 and dense tire
features still fuse. Thick taper and main contours stay dark. Displayed blank
regions remain white. This uses prepared float inputs,not the actual BGRA API.

33 tests passed. All166 AB clean teacher hashes,crop resume and prepared arrays
at6 depths reproduce exactly; all4000 actual teacher/depth records also match AB.
Actual6/1k/4k full states and4 raw/EMA exports match. E/R/AB reference metrics
reproduce exactly;192 local primary,1368 local broad,and6264 main outputs.
Numerical commit `fe536b952c431217e507fe753e2fbb8e7fad08f6`;
`runs/paper-band-generalization-20261010/final-audit.json` binds the evidence.

Synthetic rough only,44 previously used validation corpus identities. No
independent/sealed/final-source evaluation,production replacement or plugin change.
The next single-factor diagnostic will retain paper penalties and double the
existing preclip opacity-deficit coefficient.3→.6. Its direct gradient is confined
to underpredicted teacher ink,which is disjoint from the paper-band mask; shared
parameters may nevertheless widen lines. The same229 checks will test that risk.
