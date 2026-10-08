# Lineart quality campaign status

Updated 2026-10-08, 22:14 UTC. **No replacement model is adopted yet.** Production
weights remain unchanged. Experiments use synthetic rough inputs only, including
corruptions of clean real lineart. The goal is less gray fringe without losing
thin lines, continuity, width consistency, or intended separation.

| Experiment | Finding | Decision |
| --- | --- | --- |
| Ink/neighbor loss guards, 10k fine-tuning | Halo improves, faint-line retention/cuts regress | Rejected |
| Straight-through/interval output losses and paper guards | Some halo or blur gains, but retention/background regressions | Rejected |
| Original rough-source guide, 10k | Best guarded arm reduces primary halo 29.1% real / 25.4% geometry; 29 strict EMA failures | Rejected |
| Mixed tone/blur/strokes training, raw guide, 2k | Better faint-line retention, but halo and wrong bridges increase | Rejected |
| 6.25% parameter blend of guided strokes/mixed endpoints, 2k | Primary seed: 14.3% / 14.4% halo reduction and zero strict failures | Independent confirmation required |
| Same fixed blend, independent seed20261006 | 12.7% / 14.1% halo reduction, but 3 cut failures in 2 overlapping conditions | Rejected; no EMA-to-raw switch |
| Ink-centered and square-root ink guides × strokes/mixed | Four endpoints completed 2k; all ten fixed EMA models fail strict gates | Rejected |
| Controlled learnability diagnostic | Four arms completed1,200 updates; improved ink retention also creates gray branches; no sufficient fit | Completed, no adoption |
| Residual-scale floor in the same fitting task | Both arms completed1,200; blur F1 gains below .003; five native sheets retain gray branches | Rejected; [result](residual-floor-fitting-results-20261008.md) |
| Full-resolution detail branch, width8/16 | Width16 gains about.10 blur F1 in both local splits and reduces halo; defects remain | Width16 advances to broader study; no adoption |
| Width16 broader strokes/mixed data | Both completed2k; strokes screen passes but matched-guide halo and native review do not support advance; mixed fails | Rejected; [result](detail-head-broad-results-20261008.md) |
| Sampler-compatible conditional-path diagnostic |2,052 oracle runtime cases match the analytic path; existing training differs at intermediate states | Hypothesis only, not learned quality evidence |
| Explicit progress and bridge fitting comparison | Independent implementation passes279 tests; matched two-arm local plan prepared and frozen | GPU six-depth/full-restore, historical replay and36 initial forwards pass; both arms completed400 with exact full-state restore; fixed comparison continues |
| Tone-weight exponent0/.5 versus existing1 | Analytic uncertainty diagnostic motivates a loss-only comparison;284 tests pass | Prepared/frozen; queued after the conditional-path evaluation, GPU preflight pending |

The source guide always uses the actual rough input before Gaussian noise,
downsampled at each resolution. The earlier guide-encoding trial used encodings
that are zero on white and kept the126,828-parameter model size. Its losses,
recursion1–6, stage weights,
sqrt(batch) learning rate, initialization, and the 10k learning-rate clock remain
fixed. [Completed result](ink-guide-results-20261008.md) records the grid and validation.
That trial passed259 tests; all four arms restored full GPU update states exactly,
and both unchanged raw-guide controls reproduce historical updates bitwise.

The failed independent seed informed this next hypothesis. Therefore any selected
new candidate requires a **new seed20261005** and a matched control, then the
already planned fresh synthetic draws. Those draws remain unopened. The source
images in that future panel were evaluated in September; only crops and synthetic
draws will be fresh, not the source-image identity.

Native output inspection has repeatedly found missing thin strongly blurred
lines and merged gaps/parallel strokes, even when average halo scores improve.
The saved reviews record these defects rather than treating a score gain as full
quality acceptance. [Last independent result](corruption-blend-second-seed-results-20261008.md).

Two runtime findings are separate from model quality. Residual ONNX sampling now
preserves zero-noise pixels exactly; 31 tests and 27 actual-checkpoint BGRA/JAX
comparisons passed in an isolated checkout. Also, existing input scale1.1 erases
scribble alpha1–11/255 before noise and guidance. That scale has not been changed.
[Input diagnostic](runtime-input-contrast-note-20261008.md) explains why an explicit
synthetic BGRA quality check is required before adoption. CPU ONNX parity alone
does not establish Windows DirectML or Krita GUI quality.

The completed local detail improvement is concentrated at noise.5: blur F1 gains
are .187/.202 in train/probe, versus .012/.001 at noise.1. The broader comparison
therefore must not treat the aggregate local gain as proof of faint-input cleanup.
The detail branch also costs time: provisional CPU512px one-step latency is
190→252ms, measured during GPU training ([timing record](detail-runtime-latency-20261008.md)).
The new [conditional-path probe](conditional-path-fitting-plan-20261008.md) keeps
matched progress sampling in both arms; the learned comparison is running and no improvement is established yet.
