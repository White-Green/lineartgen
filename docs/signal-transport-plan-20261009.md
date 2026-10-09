# Local opacity distribution with original-source features

Test whether the earlier local opacity transport can sharpen stable original-
rough features without erasing thin lines. The earlier noisy-feature experiment
reduced halo but failed badly on opacity and cuts. The completed source-feature
study stabilizes position and width; its remaining gray edges and bridges make
this interaction worth testing. The current structural comparison and broader
diagnostic continue to their fixed endpoints independently. Do not select their
intermediate checkpoint or alter this coefficient after seeing their results.

Control is the completed source-signal raw4000 model and its0/1000 snapshots.
Treatment restarts from that study's identical zero-output-head initial weights,
NOT its trained endpoint. Add only3.0 times the existing16x16 sorted local
physical-opacity smooth-absolute transport loss. Reuse its exact implementation,
including ink/paper rank balancing and .25 blank-tile weighting, protected teacher
substitution and physical preclip gradients. No region/connectivity additions,
residual-floor, sampler, model-size, augmentation or coefficient changes.

Both models retain130,605 parameters, original-source CNN features and output
anchor, sqrt(t) correction, clipped feedback, preclip opacity-deficit.3, sqrt-tone
pixel weighting, identity.1/edge.1/contrast.01. Seed2026100951,4 fresh512px geometry
examples/update,9 kinds plus5% blanks,80% blur/opacity corruption, sigma0–2,
opacity.25–1, batch4 and base1e-4*sqrt(4), warmup100, fixed4000 cosine to1e-5,
EMA.999. Depths1–6, stage multiplier0 and downsample-before-Gaussian unchanged.

Copy the control snapshots and require exact64 primary and456 broad predictions/
metrics per endpoint. Independently replay its six updates. Preflight and actual
treatment both evaluate step0 before six updates, fixing the startup ordering
that differed in the earlier transport study. Require exact six-update full state,
exact restore and raw/EMA checkpoint exports. Stop on failures; do not silently
relax replay tolerances. Freeze source, settings, plan and input bindings first.

Final raw4000 is primary;1000 diagnostic only. The same local gates apply:
aggregate blur halo>=10% reduction on BOTH splits; each family/strength/mask
opacity-deficit increase<=.005 and F1 loss<=.005; primary per-case width-error
increase<=.10px, cuts/bridges increase<=.01. Inspect all8 final native sheets,
blank and protected boundaries, and report absolute failures too. Gaussian draw
repetitions are identical for these source-only models, not independent samples.

Even a relative pass does not establish deployment quality. Source-only features
also remove dependence on intermediate sampled states; repeated inference must
be diagnosed and addressed separately, not silently declared compatible. Broader
synthetic rough, real-clean-derived corruption, independent training seed and
unopened fresh-draw confirmation remain required. This run waits for the currently
queued broader GPU diagnostic, and never replaces production or changes Krita.
