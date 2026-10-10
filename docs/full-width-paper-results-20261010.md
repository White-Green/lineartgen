# AR full-width comparison: capacity helps separation but leaves gray/thick strokes

Completed all 4,000 updates and prescribed evaluations. The 517,865-parameter
candidate fails 41 of the unchanged 229 E/R development checks; it is not adopted.
The matched AE control has 130,605 parameters and 51 failures. Failure counts are
not a scalar quality ranking. Both use identical loss, data, update count, and
initial function (within floating-point tolerance), with fresh AdamW/EMA.

| Whole-image strength .5 | AE small F1 / deficit / halo | AR full width F1 / deficit / halo |
| --- | --- | --- |
| Corpus blur | .930713 / .074595 / .005860 | .927596 / .076873 / .005401 |
| Corpus strokes | .861751 / .042767 / .061096 | .866300 / .046677 / .056835 |
| Geometry blur | .883581 / .034583 / .025963 | .887039 / .031694 / .023971 |
| Geometry strokes | .688099 / .016403 / .149484 | .695372 / .015356 / .138480 |

Geometry blur bridge rate improves .270833→.145833, but width MAE slightly rises
.627790→.638456 px and F1 still misses .90. Strokes bridge rate .416667→.395833,
width MAE 2.277171→2.176991 px. All four cut rates in this table are zero.
Blank maximum improves .039015→.019745. Both edited black interiors are exactly
alpha 1 throughout, replacing AE's faint variation. The remaining failures are
18 deficit-retention, 17 F1-retention, 3 width, 2 bridge, and 1 primary blur F1.
The local comparison against AE has 3 final failures; those local counts use
a different reference from AE's older local summary.

All 14 final sheets were actually inspected. Large contours are similar to AE,
while small hatch marks, curved thin strokes and intersections remain gray or
uneven. Strong strokes remain too wide, with merged parallel lines and closed or
narrowed intended gaps. Some local thin curves darken with the wider model, but
the problem persists. Filled interiors are black and blank displays are white.
Partial-image rough protected areas are intentional. These are prepared float
inputs rendered through the usual alpha cutoff, not actual BGRA-input quality or
Windows/Krita GUI verification.

The tests passed 39 checks in 49.73 s. Initial 48 whole/partial cases over six
depths agree within 1.1324883e-6 (fixed tolerance 1e-5); protected pixels are exact.
All 4,000 sampling records, 166 logged teacher hashes and all six prepared input
depths match AE. Full model/AdamW/EMA restore at 6/1k/4k and all four raw/EMA
exports pass. Reused local control outputs and all 1,566 E/R/AE main metrics match
their originals exactly. Final raw4k was fixed before learning; neither EMA nor
intermediate checkpoints were selected. Numerical commit
`91bc4e306b75d952d534184741693883db82f176`.

- [Frozen plan](full-width-paper-plan-20261010.md)
- [Final audit](../runs/full-width-paper-20261010/final-audit.json)
- [Corpus outputs](../runs/full-width-paper-20261010/development/native-review/real.png)
- [Blur geometry](../runs/full-width-paper-20261010/development/native-review/blur-05.png)
- [Strokes geometry](../runs/full-width-paper-20261010/development/native-review/strokes-05.png)
- [Unfitted local outputs](../runs/full-width-paper-20261010/broad/step-4000/probe-blur-0.5-native.png)

No independent training seed, sealed confirmation, final-source identities,
production export or plugin change was used. Capacity contributes to separation
under this protocol, but this result neither establishes the minimum sufficient
size nor solves gray output.
