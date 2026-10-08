# Residual-scale floor: controlled fitting result

Both fixed .05-floor arms completed 1,200 updates and failed the predeclared
advance rule. This short diagnostic does not justify a broader floor study or a
replacement model. The unchanged 100k production checkpoint remains selected.

| Loss counterpart | Blur F1 gain, train / probe | Blur halo change, train / probe |
| --- | --- | --- |
| Conventional clipped MSE | +.00249 / +.00057 | −.01095 / −.00427 |
| Interval MSE | +.00148 / +.00008 | −.00161 / −.00526 |

The required raw blur F1 gain was at least .03 in both splits. Both arms pass the
clean F1 and per-family halo nonregression limits, but neither meets that gain.
All 36 nonblank blur conditions in each split still fail the separate descriptive
fit flag. These comparisons use the already observed local probe, not the sealed
campaign confirmation panel. The .05 floor will not be swept or extended here.

Five predeclared native comparison sheets were inspected: train/probe blur .5 raw,
probe tone .1 raw, probe clean .1 raw, and probe blur .5 EMA. The floor outputs are
visually very close to their controls. Blurred thin lines remain incomplete;
broad gray branches, false connections, merged parallel strokes and filled gaps
remain. Tone inputs recover most main strokes but retain stray fragments and gap
bridging. Clean inputs remain much better. EMA blur still loses thin lines. This
review used continuous grayscale before the runtime alpha cutoff, not Krita GUI.

The conditioning derivation remains algebraically correct, but fixing its low-t
scale alone did not produce the expected fitting gain in this protocol. The
failure does not establish that this parameterization can never help longer or
different training.

Validation: 265 tests passed (including nonzero floor ONNX/BGRA fixtures), 12
candidate GPU depth updates restored full state exactly, and 12 unchanged baseline
updates reproduced parent full states bitwise. A separate low-strength partial
mask probe demonstrated changed candidate gradients with exact protected pixels.
The actual trajectories were restored exactly after update1 and at update1200.
All 912 initial evaluation predictions matched their counterpart exactly for the
fixed evaluation inputs. The final audit verifies 215 source files, 5 settings,
the source archive, 12 finite snapshot triplets, and 3,648 finite metric records
with zero protected-pixel error. The first sandboxed full test attempt stalled at
a checkpoint test and was terminated; an unrestricted unchanged rerun passed.

Artifacts: `runs/residual-floor-fitting-20261008/` in the
`jax-residual-floor-20261008` worktree; numerical source
`5c9ed5588dfb3d7c5ef6d322ceaed586a55b087f`. See `matched-results.json`,
`completion-audit.json`, and `evaluation/step-1200/visual-review/review.json`.

Next hypothesis: the final decoder has only two channels and the original output
head is 3→3→3→1 (196 parameters), even in the wider encoder control. A zero-output
initialized full-resolution detail branch can test this capacity restriction
while preserving the inherited function initially. That is a new experiment,
not a demonstrated cause of the observed artifacts.
