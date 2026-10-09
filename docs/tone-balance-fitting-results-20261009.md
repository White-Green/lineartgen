# Tone-weight fitting result, 2026-10-09

Both fixed 1,200-update arms are rejected by the predeclared **raw, one-step** local screen. No production model changed. Weakening tone balancing clearly reduces gray halos, but also reduces target-line opacity. This is evidence for a loss tradeoff, not proof that class weighting is the only cause of the artifacts.

The complete record is `runs/tone-balance-fitting-20261008/`. Numerical source is `2aa86743d19f477609a56e9d949dd9684168c5b4`; 233 frozen files and all settings remain unchanged. Both 130,605-parameter arms start at the same original 100k EMA, with zero added outputs, and use the fixed 57 train/57 probe synthetic pairs, seed, batches, noise, recursion, and 1,200-update schedule in the plan. Only the tone-weight exponent differs from the historical balanced width-16 reference.

| Arm | Low-strength blur halo reduction, train / probe | Failed strata / 24 | Reasons |
| --- | --- | --- | --- |
| Square-root weights | 47.486% / 49.433% | 24 | Ink deficit increases by more than .005 in all 24 |
| Unit weights | 82.035% / 83.455% | 24 | Same 24 ink failures, plus F1 loss in 9 overlapping strata |

Square-root weights have no F1 or halo safeguard failures. Their blur results nevertheless show the opacity problem:

| Split / noise | F1 | Halo excess | Ink deficit | Ink-deficit increase vs reference |
| --- | --- | --- | --- | --- |
| Train / .1 | .673605 | .074956 | .191104 | .045677 |
| Probe / .1 | .699833 | .067577 | .206878 | .058513 |
| Train / .5 | .433992 | .094853 | .411642 | .157091 |
| Probe / .5 | .512037 | .091815 | .360004 | .132839 |

All **12 final native-resolution sheets were actually inspected**, including clean/tone, both blur strengths, train/probe, EMA, one/two/five iterations, partial masks, and blank inputs. `evaluation/step-1200/visual-review/review.json` binds individual observations and image hashes. The original renderer manifest remains unchanged; the separate review records completed inspection.

Square-root weights visibly remove many false branches, while plain weights often erase weak strokes. Neither restores thin-line opacity reliably. Repetition darkens some square-root outputs but also broadens lines, fills intended gaps, and creates corner/junction spurs. At noise .5 the square-root blank output still develops false curves after five iterations. The cleaner appearance and improvements at other iteration counts do not replace the declared one-step criterion. These are learned float-input outputs, not a Krita BGRA/Windows quality test.

Validation: the inherited 284-test full suite passed; the separate aggregation correction passed its two focused tests. Both candidate six-update GPU preflights and the historical six-update full-state replay passed. All 36 initial GPU comparisons and actual step-1/400/1200 full-state restore checks were exact. Final candidate evaluation contains 5,472 predictions; reference evaluation contains 2,736, with historical one-step outputs bitwise unchanged. The separate None-aware summary fix is hash-audited; it changes no prediction and retains F1/halo checks for partial rows without active teacher ink. No threshold or arm was changed after viewing results.

The already frozen unweighted rough-path versus conditional-bridge comparison can now run. A possible later study is tempered weighting with an explicit one-sided opacity-retention term; it is not part of this completed experiment. Broader quality, independent new training seed, actual BGRA output checks, and sealed fresh synthetic confirmation remain prerequisites for adoption.
