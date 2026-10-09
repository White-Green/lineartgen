# Remove artificial Gaussian from feature extraction: controlled learnability study

Source-anchor output removes the mandatory noisy skip but still allows the
feature extractor to inject Gaussian-dependent corrections. Test whether using
the original rough image as the main CNN image signal resolves that burden.
This is an input-use ablation, not a new loss and not evidence of sufficient
model size before it is measured. The concurrent anchor4000 study is unchanged.

Control: completed source-anchor arm from source-anchor-20261009, feature_input
noisy. Treatment signal: same source anchor, feature_input source. Its encoder,
main head and detail head read the original downsampled rough guide in place
of the artificially noised image. Noise-strength map, shared recursive smaller
prediction, guide projections, parameters and tensor shapes are unchanged.
At t=0 the output remains the incoming protected image. The candidate receives
no clean teacher at inference. Any noise inherent in the user's rough image
remains in the guide and still needs restoration.

For this candidate, changing the Gaussian seed no longer changes the output;
this is deliberate and tested, not claimed to preserve stochastic semantics.
The public BGRA/strength/mask API is unchanged. Training still builds the same
pyramid and adds the same Gaussian AFTER downsampling, so exact paired data
and masks are retained. Only the network's use of that artificial noise differs.
Keep recursive depths1–6, direct stage multiplier0 and sqrt(batch) LR.

Both start from the anchor study's identical saved zero-output-head initial
weights, same optimizer/EMA initialization, procedural seed2026100951,
16,000 new geometry examples,80% blur/opacity augmentation, batch4, peak2e-4,
warmup100, cosine4000 to1e-5, EMA.999. Existing sqrt-tone loss and physical
preclip opacity-deficit.3 remain; no connection/region/transport loss added.
Compare raw1000 and primary raw4000, no endpoint selection or extra training.

Reuse the control's0/1000/4000 snapshots exactly, with source renamed pixel for
the existing evaluator. Reproduce six control updates exactly; require all64
primary and456 broad control predictions/metrics to match the saved parent.
This avoids rerunning the same4000 control updates. Treatment preflight and
actual startup both evaluate step0 before updates; exact initial prediction
identity, six-update replay and full-state restoration are required. Preserve
raw/EMA and full checkpoints at1000/4000. Freeze source/settings/inputs first.

Same screen as the parent: >=10% aggregate blur-halo improvement on both splits,
no family/strength/mask stratum mean ink-deficit increase>.005 or F1 decrease
>.005, no primary case width error increase>.10px or cut/bridge increase>.01.
Native review of all8 fixed final sheets covers nine kinds, two strengths and
splits, white backgrounds and protected boundaries. Relative wins against a
weak control are insufficient: also inspect absolute thin-line/width/opacity
quality, then compare promising outputs with the established non-reset pixel
reference before broad confirmation. Real-clean-derived synthetic rough,
repeat strokes, multi-iteration BGRA quality, independent seed and sealed fresh
confirmation remain downstream. Production model and Krita plugin stay unchanged.
