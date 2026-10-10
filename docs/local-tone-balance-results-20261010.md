# W: local ink/paper weighting, 2026-10-10

**Completed; not adopted.** The fixed raw4k model fails49/229 established development checks. Local weighting slightly improves corpus blur F1, but increases corpus halos and does not recover pale fine geometry. All14 final sheets were inspected. [Predeclared plan](local-tone-balance-plan-20261010.md).

Only `loss.local_tone_radius=8` differs from R: multiply the existing square-root tone weights by a17×17 local ink/paper minority factor, then normalize per image. Same130,605-parameter model, E initialization/fresh optimizer/EMA, data and4k schedule. Radius0 remains exactly compatible with frozen R. No W radius sweep, V enlargement or U augmentation was combined with it.

Whole-image strength.5:

| Metric | R | W local8 |
| --- | ---: | ---: |
| Corpus blur F1 | .936629 | .941338 |
| Corpus blur ink deficit | .070756 | .063400 |
| Corpus blur halo | .008512 | .010479 |
| Corpus strokes F1 | .843711 | .838075 |
| Corpus strokes halo | .095144 | .103726 |
| Geometry blur F1 | .824615 | .817421 |
| Geometry blur ink deficit | .108294 | .113564 |
| Geometry blur halo | .025011 | .027505 |
| Geometry blur width MAE, px | .589909 | .587831 |
| Geometry blur cuts | .083333 | .078125 |
| Geometry blur bridges | .125000 | .125000 |
| Geometry strokes F1 | .682142 | .685365 |
| Geometry strokes halo | .161461 | .159688 |
| Geometry strokes width MAE, px | 2.339986 | 2.282321 |
| Geometry strokes cuts | .010417 | .010417 |
| Geometry strokes bridges | .395833 | .437500 |

Failures comprise15 grouped F1,9 deficit,12 relative halo,4 bridges,3 cuts,3 width,2 primary geometry and1 absolute corpus strokes halo. Black interiors717/727 remain mean/minimum alpha1. Blank maximum.036755 passes.05.

Local raw4k blur.5 F1 changes .799440→.782988 TRAIN and.785048→.774841 PROBE; deficit .206846→.212183 and.251725→.254108; halo .023949→.028582 and.018760→.022776. Local failures10 at1k→13 at4k. Native sheets retain gray thin curves/corners/tapers, rounded crossings, overthick repeated strokes and merged parallel/gap examples. Fine tire hatching loses separation; simple major contours, black fills and blanks are preserved.

The [weighting mechanism audit](local-tone-mechanism-audit-20261010.md), performed before final outputs, identifies a limitation: square neighborhoods strongly emphasize a paper slit in a mostly black area, but barely emphasize a gap in a thin line. After normalization, its absolute pixel gradient decreases18.9% in a fixed thin-line-gap fixture and13.0% in a parallel-gap fixture. This is a mechanism example, not proof of the cause of W's learned failures. The follow-up `budget-preserving-diagnostic` algebra preserves binary ink/paper weight sums and restores those fixture gradients to.00020612/.00020802 (original.00020208/.00019672). That alternative has **no training implementation or learned result**; for antialias targets its guarantee concerns mixture-component budgets, not effective alpha-weighted final weights, and it was whole-image only.

27 tests passed: default compatibility, complement symmetry, finite normalized fields, local minority emphasis, correct interval-L1 repair directions/zero teacher subgradient, actual update, complete state and next-update equality, screen and sampling-resume regression checks. An initial unit run exposed float32 mean-reduction drift in constant fields and reversed sign assertions in the test; axis-wise final normalization and corrected assertions passed all27 before training. Frozen numerical commit `019e5c8fb8f4480337d36acbd793b11fb56affc5`.

All166 R clean hashes and all6 prepared arrays match; actual startup6/1k/4k full states restore exactly. Four raw/EMA exports match, all4000 sampling records and16000 teacher counts agree, and reused E/R reference metrics reproduce exactly. Evaluation128 primary+912 local broad+4698 broader outputs is finite with exact protected pixels. Fourteen hash-bound reviews and `final-audit.json` are in `runs/local-tone-balance-20261010`.

Synthetic rough only,44 existing validation corpus identities; no independent/sealed confirmation or production/plugin changes. The next data hypothesis is the synthetic teacher width prior: E used maximum6px in its procedural generator, while R uses12px. This is not the only difference between E and R. A separate matched comparison can isolate that bound without changing loss, model, native teachers or wide coverage primitives.
