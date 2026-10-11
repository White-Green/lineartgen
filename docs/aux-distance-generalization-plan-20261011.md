# BL: auxiliary distance learning on diverse procedural clean targets

BK separated distance from image output and retained its fixed fit, but failed
straight-line F1 retention and did not remove gray thick-line interiors. This
new broad diagnostic tests generalization under diverse training exposure. It is
not adoption/advancement of BK's weights and does not relax BK's failed criteria.

Both control and auxiliary start at the same E4000 core, before fixed AD/AO/BK
fitting, with identical BK auxiliary heads (3777 parameters, seed2026101107),
fresh Adam and EMA. Model130605 inference/134382 training parameters. Only loss
weight0 versus.1 differs between arms. AD main image loss unchanged; auxiliary
normalized coverage-distance/radius8 is separate and removed on export. Use
the full clean teacher for the field and its tone weights, with active mask,
while the main image target retains original rough at protected pixels.
E's old export lacks the later one-pass metadata flag. The local initial export
sets it to one-pass, with identical weights and all other model settings. One
sampling step must reproduce the original E outputs bitwise before training.

Train4000 updates, batch4, accumulate1, seed2026101111, baseLR5e-5 with sqrt4
scaling (peak1e-4), warm100/cosine to1e-5. Cycle highest-resolution512px depths1–6;
top-stage losses backpropagate through smaller clipped image predictions.
Generate all16000 teachers procedurally: geometry probability.9, coverage1
conditional on not selecting geometry, so no native corpus training samples.
Existing generator samples1–4 groups from9 line kinds, widths loguniform.6–6,
thin.6–1.8, blank probability.05,4x antialias rasterization. Coverage includes
paper/black/ellipse/polygon/wide strokes16–128. Keep image contrast1.1.

Use existing mixed sketch augmentation probability.8, equal tone/blur/strokes,
strong probability.5: ordinary opacity.25–1, sigma0–2, displacement2; strong
opacity.2–.45, sigma1.5–2.5, displacement3. Radius2 dilation blend1,3 displaced
stroke composites. Existing strength/mask sampling and noise after resizing.
Both arms receive the exact same prepared batch. Keyed generation has no access
to evaluation images or geometry probes; no real rough/clean paired data.

Keep full wrapper/Adam/EMA states0/6/1000/4000,16 core raw/EMA exports. Audit
all4000 depth/teacher-count/target-hash records; independent CPU regeneration at
steps0–11 and every25th step. Regenerate and replay first6 updates completely.
Check initial model equality, optimizer/EMA initialization, all restores and
exports. Finalraw4000 only;1000 is retained for restoration, not selection.

Evaluate existing development1566 conditions/model at depth3 and one-pass, using
the locked AE reference and panels: initialE, frozenR4000(pixel), matchedcontrol,
auxiliary (6264 rows). These are reused development conditions, not confirmation.
44 corpus identities appear only as clean sources for synthetic evaluation
corruptions, plus synthetic geometries/coverage/blank. Training uses none of them.
Replay8 stored E/R primary predictions before training and all frozen-reference
metrics plus saved panel predictions at evaluation. Review all6 native sheets
(blur, strokes.1/.5, corpus, partial, worst blank), retaining prepared-float
protection, finiteness, and actual black interior checks. Display conversion is
not a full BGRA API or Windows GUI test.

Keep historical229 criteria against E/R for each trained arm (458total), including
geometry blur.5 F1>=.90/cuts<=.02; corpus halo<=.02 blur and<=.10 strokes;
F1 drop<=.005, deficit increase<=.01, halo<=reference*1.1+.0005 across matched
groups; geometry width increase<=.10 and cuts/bridges<=.01; black interior
alpha>=.95 and blank max<=.05. Also report the direct control/auxiliary differences.
No new thresholds, no selecting a visually attractive checkpoint, no sealed or
independent training-seed confirmation yet. No production/plugin/default change.
