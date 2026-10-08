# Corruption mixture: rejected at 2k

Both EMA candidates fail the unchanged screen: faint-line retention improves,
but excess halo and unwanted bridges increase. Neither trained candidate is
continued to 10k or adopted. Second-seed and fresh-condition confirmation remain
unopened for them.

Root `runs/corruption-mixture-20261008-screen` used source
`56cc1b25520c6d703b82381958055798a4616f2c`, with 198 files and 15 settings sealed.
The only training intervention relative to the corresponding source-guided model
was `sketch.mode= mixed`: at total augmentation probability .5, uniformly select
tone, blur or displaced strokes for each enabled example. All ranges, losses,
initial 100k EMA, guide architecture, clean draws, Gaussian keys/masks, recursion,
stage weights and optimizer schedule were unchanged. Protected targets change
with the rough input, as intended. No coefficient fitting was performed.

245 tests passed (three existing deprecation warnings); both new arms completed
12 six-depth GPU preflight updates with exact full-state restoration. The two
legacy guided arms reproduced their complete historical six-update model,
optimizer and EMA states bitwise. Training-only diagnostic draws over 192
examples allocated 107 clean, 23 tone, 31 blur and 31 strokes, and each mixed
example matched its selected legacy transform within 3e-7. Initial 1,562 metrics
were exactly equal to the guided baseline. Training ran approximately
10:12–11:35 UTC on 2026-10-08; evaluations completed by 11:41.

| EMA method | Real strong-strokes halo increase | Geometry strong-strokes halo increase | 2k screen | Final diagnostic failures |
|---|---:|---:|---|---:|
| mixed_source_control | 38.997% | 31.525% | halo + five bridge groups | 62 |
| mixed_source_ink_guard | 30.854% | 22.715% | halo + five bridge groups | 51 |

Comparisons are against conventional control at the same 2k milestone. Only
comparisons to the matching guided-strokes counterpart isolate the mixture.
Real inputs are clean real lineart corrupted synthetically. Failure counts use
overlapping strata, not independent defects. Raw also fails the screen; its
halo increases are 35.936%/15.107% and 24.266%/4.157%, with final diagnostic
failure counts 32/25.

Real strong-tone ink deficits improve by 9.483/9.610 percentage points;
geometry strong-tone cuts improve by 11.979/12.187 pp. Geometry strong-blur
missing fractions improve by 10.294/10.723 pp, but bridges increase by
14.583/16.667 pp. Strong-strokes width MAE worsens by .940/.781 px. These are
tradeoffs, not evidence that the mixed models are better overall.

Eight fixed native-pixel sheets were inspected, with hashes and observations in
`review.json`. Mixed models produce thicker, darker lines and recover some faint
fragments, but leave more gray marks, fill gaps and merge parallels. Repeated
inference often reinforces these connections. Maximum active mean opacity in
blank tests is .015579/.009444 (EMA), .008035/.003981 (raw). All defined scalar
metrics in eight files are finite; protected errors are zero; exported raw/EMA
weights match restored finite 2k states exactly. `completion-audit.json` retains
this evidence.

The next hypothesis uses the opposing retention/halo behavior of the two guided
ink-guard endpoints. A separate, prospective inference-only study can test a
small fixed grid of parameter interpolations at these existing 2k checkpoints.
It must apply the unchanged strict final criteria immediately, rather than call
a permissive 2k screen pass sufficient for confirmation. No extra training or
optimizer history is attributed to such constructed checkpoints. Candidate
ratios must be fixed before their first inference and validated again with the
same ratio after separately fine-tuning both endpoints under a second seed.
