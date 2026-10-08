# Ink-centered rough-source guidance

The raw-guide 6.25% blend failed independent seed20261006 confirmation on cuts,
despite reproducible halo reduction. It is not adopted. Fresh synthetic conditions
remain unopened. This new experiment changes the guide representation, not that
study's alpha, gates, or decision.

The two existing bias-free guide convolutions see white as +1. Their response to
faint ink is therefore coupled to their response on paper. Test `ink=(1-guide)/2`
and `sqrt_ink=sqrt(clip(ink,0,1))`: both have zero white response, and the latter
amplifies low-opacity structure. This is a hypothesis, not an established gain.
Encoding happens after each source-pyramid downsampling. The guide remains the
actual augmented rough input before Gaussian noise; no clean target is supplied.

Train each encoding under the already fixed strokes and mixed corruption recipes:
four endpoints at 2,000 updates, seed20261007, common100k EMA initialization,
unchanged 10k cosine clock, zero guide projections, 126,828 parameters. Keep ink
and region loss coefficients, clean samples, masks, noise, recursion1–6, stage
weights, sqrt(batch) learning rate, optimizer and EMA settings unchanged.

Before training, fix a within-encoding alpha grid **0, 1/16, 1/8, 1/4, 1** between
strokes and mixed endpoints. The endpoints plus six interpolations give ten models.
Evaluate all against the matched conventional control at strict final thresholds
at the fixed2k age. EMA is primary; raw, blank, 1/2/5 repeats, and native pixels
are diagnostics. Select at most one visually acceptable strict-passing model by
sum of primary halo gains. No permissive screen or automatic10k continuation.

A passing candidate must reproduce with **new fine-tuning seed20261005**, fixed
encoding and alpha, and a newly trained matched control. Seed20261006 informed
this hypothesis and cannot count as independent confirmation for it. Only then
seal checkpoints and open the already declared prospective synthetic draws.

Validation before training: complete tests, actual six-depth GPU updates/restores
for every new arm, full six-update raw-guide replay against existing strokes and
mixed controls, and exact zero-guide forward checks across recursion1–6 and
1/2/5 sampling. ONNX exports must preserve the BGRA API with nonzero guide weights.

Pretraining validation completed on 2026-10-08: 259 tests passed (3 existing
warnings, 354.05 seconds). All four arms completed six actual GPU updates, with
exact complete-state restore. Both raw-guide counterparts reproduced their
historical model/optimizer/EMA states exactly after six updates. Zero-initialized
guide forwards match the 100k parent bitwise in 64 comparisons per corruption
family (two encodings, four cases, depths1–6, and 1/2/5 steps at depth3). All eight
guide-branch parameter tensors are finite and nonzero after those six updates.
The plan remains unchanged. An initial directory-creation error in setup was
fixed before any preflight or training; its partial setup and log were retained.
