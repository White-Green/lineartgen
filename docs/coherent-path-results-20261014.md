# DP: coherent local profile choices did not resolve gray bands

Numerical source: `43a1287`; run `runs/coherent-path-20261014`. Both arms completed 6,000 updates from DN `both` raw18000 at unchanged 15,809 parameters. No candidate passes the existing visual gate; the reserved fresh-confirmation data remains unevaluated. Nothing was deployed.

The global arm selects one clean translation across all profiles; the path arm selects locally along each stroke with a transition penalty. Both use the same envelope, raw dense profile loss, physical stage targets, data, augmentation, optimization and initialization. The contrast tests choice coupling. Relative to DN, several common objective components also change, so it does not isolate any one of those components.

## Actual outputs

All 28 declared native comparison pages were viewed at original resolution: 12 standard, 10 stress, 6 multi-shape512. The general pattern remains very close to DN:

- Close curves (`new_probe36/38/39`, scenes2/5) retain gray secondary bands, weak duplicate edges and uneven dark strokes.
- Crossings (`new_probe30`, scene7) still have unwanted central or adjacent cuts. Intentional gaps in scenes1/7 were not counted as failures.
- Wide59/61 keep gray sidebands and end remnants; wide62 remains shortened, distorted or fragmented. Wide58/62 lose active tips under partial masks. Some wide diagonals remain good, but the new losses do not consistently repair the hard cases.
- Black fills have opaque interiors but damaged bright corners and an inset frame. Simple and faint individual strokes remain mostly good.
- Global and local path choice have no convincing visual separation sufficient for adoption.

The updates were real: parameter RMS/max changes from initialization are .00914/.08816 (global) and .01015/.08573 (path). On `new_probe38`, output RMS/max changes are .01478/.39413 and .01662/.50959. This is not an accidental evaluation of identical checkpoints. First/last600 mean training losses were .33468/.30466 and .25521/.22654; different objective values across arms are not quality rankings.

## Direct pixel optimization diagnostic

While training started, a separate fixed diagnostic optimized unconstrained pixel images for 1,000 Adam updates using the same objectives, for new_probe30/38/62/56. Both objectives produced opaque connected crossing/paired curves/thin stroke and an intact black rectangle from normalized rough inputs. Global free-pixel fitting was smoother; local path fitting had visible fine jaggedness along crossings and paired curves. The comparison image was actually inspected.

This shows these losses admit useful opaque solutions on four examples. It does not show that the CNN learns them, that the objective excludes all artifacts, or that local choice guarantees smoothness. In particular, continuing to loosen local geometric matching is not supported as a sufficient remedy.

## Verification

14 CPU tests passed, including exhaustive DP agreement, tie-gradient selection, real-raster gray/missing/cut penalties across scales, zero protected gradients, and real generator update/restore/export. Ruff passed. First-six-step full-state replay differences were 1.79e-7 global and 4.47e-8 path under the predeclared 5e-6/1e-4 envelope. All saved optimizer/EMA states and 16 raw/EMA exports restore exactly. All 6,000 paired training input hashes match.

4,608 standard +4,608 stress +192 multi-shape outputs were checked for finite values and exact protected pixels. All 4,704 paired Gaussian comparisons are exact, as expected for the source-only model. All 3,072 reused DN standard/stress initial predictions are exact. Float/uint8 output RMS mean/max: global .001243/.004076; path .001254/.004260. Last600 timed updates average .10306/.10608 seconds per arm, excluding data preparation and hashing.

Main driver and all four stages exited0. `completion.json`, `diagnostics/integrity.json`, manifests, image hashes and individual observations bind the evidence. Completion means the comparison finished, not satisfactory product quality.

## Next question

The free-raster solution and the trained CNN differ markedly on the same representative shapes. More small changes to pointwise/profile matching are not justified as the sole strategy. The next bounded investigation will test a representation that explicitly draws opaque strokes from continuous geometry, beginning with oracle rendering and geometry-gradient tests. It must preserve valid coarse-scale coverage, crossings, intentional gaps, taper, fills and exact protected pixels before a learned predictor is attempted. This is a hypothesis about a useful inductive bias, not a conclusion that the current capacity is insufficient or that vector prediction will solve the task.
