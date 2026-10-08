# Lineart quality campaign status

Updated 2026-10-08, 15:28 UTC. **No replacement model is adopted yet.** Production
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
| Ink-centered and square-root ink guides × strokes/mixed | Four endpoints training to 2k; ten fixed endpoint/blend models planned | In progress |

The current source guide always uses the actual rough input before Gaussian
noise, downsampled at each resolution. The new encodings are zero on white and
keep the 126,828-parameter model size. Losses, recursion1–6, stage weights,
sqrt(batch) learning rate, initialization, and the 10k learning-rate clock remain
fixed. [Current plan](ink-guide-plan-20261008.md) records the grid and validation.
All 259 tests pass, all four new arms restore full GPU update states exactly,
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
