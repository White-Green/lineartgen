# Native teachers plus explicit ink-area coverage, 2026-10-10

Freeze this plan and code before training. The preceding native-target run sharpened contours but grayed legitimate black interiors (preselected partial cases717/727 mean alpha .651/.624 instead of1). It also removed many blank crops and most extremely wide teacher strokes. Test whether restoring coverage of paper, large ink regions and broad strokes can fix these regressions without restoring blurry super-resolution teachers. This is a data-distribution intervention; it does not prove that missing examples were the only cause.

Run after the uniform-loss study completes, **only if its fixed broader gates do not all pass**. Otherwise prioritize its independent confirmation. Do not choose a best-looking checkpoint or hyperparameter from intermediate outputs. All new teachers are procedural, no validation geometry is copied, and all rough inputs remain synthetic. No production assets, Krita UI or sealed inputs are changed.

Keep original geometry replacement probability .2 and its exact item choices, RNG streams and rendered targets. Among remaining corpus items, independently replace with probability **.125**. Expected mixture:70% native corpus /20% original thin-to-medium geometry /10% new coverage targets. Each coverage replacement chooses uniformly among five kinds: pure paper, pure black, filled rotated ellipse, convex3–6-vertex polygon, and a round-capped wide straight stroke. Shapes have continuous sampled positions, dimensions and angles; wide stroke width is log-uniform16–128px. Render at the existing4x supersampling and downsample with Lanczos once. Preserve legitimate antialiasing. No SwinIR or external generator is involved in new targets.

New `geometry.coverage_probability` defaults to0, and bounds default to16/128. Disabled behavior must reproduce prior input hashes. Coverage never replaces an already chosen original geometry item. It has independent optimizer-step/item keys. Record kinds and generation keys. Pure fields have no fabricated curve profiles. Existing geometry profiles and paths remain unchanged.

Four final methods form a development factorial comparison:

- `pixel`: reuse L native-target control, noise weight floor.05, no coverage.
- `flat`: reuse M uniform-loss control, noise weight floor1, no coverage.
- `coverage`: train coverage enabled with floor.05.
- `flat_coverage`: train coverage enabled with floor1.

All start from **E source-signal raw4k**, not from L/M final weights. Both new arms get fresh identical optimizer/EMA initialization and identical teacher/rough/mask/strength/Gaussian arrays. Preserve seed2026100961, source split891/64/64, batch4/accumulation1, 4,000 updates, baseLR5e-5×sqrt(4)=1e-4, warmup100, cosine to1e-5, EMA.999, original-strength one-pass inference, 130,605 parameters, recursive depths1–6 and stage multiplier0 with recursive gradients. All original loss coefficients stay fixed. Floor1 changes relative pixel/auxiliary loss scale as documented in M. Rotate update order. Use raw4,000 as primary; export/audit EMA without selecting it after inspection.

Before learning, test field/shape generation, exact replay, black interior presence, supersampling, preserved original geometry and no source mutation. Reproduce L/M logged clean hashes with coverage disabled; validate unchanged items when enabled and exact resumed batches. Check full prepared input equality between the two new arms for all depths. Run real optimizer/update/full-state restore tests. Freeze source/settings and copied control hashes. Retain actual startup6, preserve full checkpoints6/1k/4k, and verify restored raw/EMA parameters exactly. No claim of bitwise independent GPU retraining.

At1k/4k evaluate local64+456 per method (256+1,824 outputs per endpoint), reproducing both reused controls exactly. At4k evaluate all1,562 existing development cases+4blank for E initial and all four methods (**7,830 outputs**). Additional Gaussian draws are identical for this architecture, not independent samples.

The goal of this follow-up is **ink retention while preserving clean appearance**, not another arbitrary10% halo reduction from an already sharp but pale model. Both new methods have predeclared screens, reported separately; no endpoint selection:

1. For every corpus clean/tone/blur/strokes whole-strength.5 family, mean ink deficit must fall at least30% versus its matched no-coverage control **or already be <=.05**.
2. In each of preselected partial cases717/727, edited teacher-alpha>.98 pixels at distance>8px from nonblack must have predicted mean alpha>=.95. This is a development diagnostic of observed failure, not unseen confirmation.
3. Corpus whole-strength.5 mean halo <=.02 for blur and <=.10 for strokes. Across every available real/geometry clean/tone/blur/strokes strength.1/.5 whole/partial group, halo <=E-initial halo×1.1+.0005.
4. Against both the matched no-coverage control and E initial, keep existing guards per group: F1 drop<=.005, ink-deficit increase<=.01; whole-geometry width MAE increase<=.10px and cut/bridge increase<=.01. Blank max opacity<=.05, finite outputs and exact protected pixels.

Coverage versus pixel estimates the coverage effect under inverse weighting; flat_coverage versus flat estimates it under uniform weighting. Their cross comparison describes interaction but is not a separately tuned candidate. Initial E remains an absolute preservation reference. Report all checks and failures, including divergent effects between methods. Passing a development screen does not automatically authorize production adoption.

Review all eight final local sheets and six broader native sheets: corpus blur/strokes, geometry blur.5/strokes.1/strokes.5, informative partial717/727 and recorded worst blank pixels. Specifically inspect whether black interiors stay black, thin lines remain continuous, nearby lines stay separate, widths are consistent and white paper is clean. A promising method still requires independent training-seed confirmation, sealed new-condition testing and actual BGRA/ONNX parity before adoption.

The inherited local summary compares all displayed methods against the common `pixel` reference; it remains a diagnostic. The broader matched-control screens above determine development status, and the reused flat control is never counted as a newly trained treatment. Preserve the full per-method local rows so those differences remain visible.
