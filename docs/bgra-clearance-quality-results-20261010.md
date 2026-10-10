# AB raw4000 at the BGRA boundary — completed

The fixed AB endpoint modestly reduces Y's fringe and width in the32 additional
actual CPU ONNX outputs, but it does not fix faint-input clipping or merged
parallel lines/gaps. This is a diagnostic, not adoption.

The [plan](bgra-clearance-quality-plan-20261010.md) reuses the exact16 physical
scribble byte arrays from the [E/R/Y diagnostic](bgra-input-quality-results-20261010.md),
with scales1.1/1.0 and strength.5. The existing96 E/R/Y outputs remain frozen
references. No new training, intermediate selection or runtime-default change.

Selected output-BGRA group metrics, scale1.0:

| Metric | Y | AB |
| --- | ---: | ---: |
| Corpus blur F1 | .901050 | .907903 |
| Corpus blur ink deficit | .065189 | .077519 |
| Corpus blur halo | .025684 | .020243 |
| Corpus strokes F1 | .768889 | .777982 |
| Corpus strokes ink deficit | .066064 | .079052 |
| Corpus strokes halo | .128033 | .116147 |
| Geometry blur F1 | .861946 | .877940 |
| Geometry blur ink deficit | .072685 | .080865 |
| Geometry blur halo | .042004 | .040518 |
| Geometry blur width MAE,px | .602910 | .562746 |
| Geometry blur cuts | .000000 | .000000 |
| Geometry blur bridges | 1.000000 | 1.000000 |

All cases/scales and raw/post-BGRA metrics are retained. AB at scale1.1 has geometry
F1 .311508, ink deficit .575602 and cuts .645833: the severe faint-line regression
persists. Its scale1.1 corpus blur/strokes halo is .005040/.088673, but lower halo
there accompanies much greater missing ink. These strong-only, previously seen
examples do not establish population rates or natural-sketch performance.

All three native sheets were viewed. Main contours and the thick taper survive;
AB is slightly less thick/fringed than Y. Dense tire details, the parallel pair
and the intentional small gap still merge. Scale1.1 nearly erases fine curves and
junction branches in both models. Scale1.0 improves their visibility but leaves
the structural errors.

Input bytes and prepared-input loss statistics reproduce the earlier audit
exactly. All32 JAX/ONNX comparisons pass rtol2e-4/atol2e-5; maximum float difference
8.256e-6 and BGRA difference1 byte. Source/checkpoint/input/exports/predictions and
visual reviews are bound in`runs/bgra-clearance-quality-20261010/final-audit.json`.
No independent/sealed evaluation, Windows/Krita GUI test or production replacement.
