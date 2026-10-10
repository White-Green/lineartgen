# P: line-retention data distribution, 2026-10-10

Freeze before either new arm trains. This plan follows completed O results: inverse-strength L1 reduced corpus blur halo by44.45% versus its MSE control, retained filled black interiors, but geometry blur F1=.85088/cut=.07083 and overthick strokes remain. Neither O arm passed. Keep O inverse-strength L1 fixed and independently test two data changes. These are development diagnostics, not independent confirmation of earlier results.

- `pixel`: reuse O `l1` raw/EMA snapshots0/1k/4k as the matched control. Expected training mix70%native corpus,20%original procedural geometry,10%field coverage;80%mixed rough augmentation enabled (equal tone/blur/strokes).
- `geometry50`: original geometry probability.2→.5; coverage conditional probability.125→.2, keeping expected overall coverage10%. Expected40%native corpus/50%original geometry/10%coverage. All rough augmentation settings unchanged.
- `blur80`: same clean teacher distribution as O, rough augmentation mode `mixed`→`blur`. Enabled probability remains.8, hence80%blur/20%clean. Geometry and coverage settings unchanged.

The geometry comparison changes clean teacher frequencies, not the original geometry generator. The blur comparison changes rough inputs, not clean teachers. Coverage proportions are expectations; retain actual counts. Each arm separately starts from common **E raw4k** with fresh optimizer/EMA, not O's trained endpoint. No combined geometry+blur arm is included in P.

Common settings:130,605 parameters, source-only features and source residual anchor, one-pass original-strength inference, native corpus teacher crops, seed2026100961, batch4/accumulation1,4,000 updates, baseLR5e-5×sqrt(4)=1e-4,warmup100,cosine minimum1e-5,EMA.999. Loss is absolute clipped pixel error with inverse-strength floor.05, sqrt-tone balance, identity.1,edge.1,contrast.01,physical preclip opacity.3. Preserve recurrence1–6, stage loss multiplier0 with recursive gradients, and downsample-before-Gaussian. Rotate arm update order. Fixed primary endpoint is raw4k; EMA and intermediate checkpoints are retained for audit, not selected for passing scores.

Before training, test actual updates with different arm teachers and full-state restore. Audit that only the listed config fields change. Replay every O logged clean hash; blur80 clean targets must be byte-identical. The original20% geometry examples must remain identical in geometry50. Compare batch restart6–11, first12 target hashes per arm, and shared noise masks across all6 recursion depths; edited targets must equal each arm's clean teacher and protected targets its rough source. Derive expected4k teacher and depth counts independently from indexed RNG. Commit/freeze code, config, plan, input and reference bindings. Retain actual startup6 updates and full states6/1k/4k, with raw/EMA exports at1k/4k.

At1k/4k evaluate local64+456 conditions per method (192+1,368 outputs), with reused O control outputs and metrics required to reproduce exactly. At4k evaluate1,562 existing development conditions+4blanks for E initial, O control and both new arms: **6,264 outputs**. Corpus evaluation sources remain44validation identities, no training/final-test identities. `real` means the existing corpus with synthetic rough augmentation, not natural sketch/clean pairs. Additional Gaussian draws are identical with source-only features, not independent observations.

Screen each new arm separately:

1. Whole geometry blur strength.5: F1>=.90 and connection cut rate<=.02.
2. Whole corpus strength.5 halo: blur<=.02,strokes<=.10.
3. Every available real/geometry clean/tone/blur/strokes group at **strength.1/.5 only**, whole/partial: halo<=reference×1.1+.0005 versus both E initial and O control; F1 decline<=.005 and ink-deficit increase<=.01 versus both references.
4. Whole geometry groups: width MAE increase<=.10px, cut/bridge increases<=.01 versus both references.
5. Preselected717/727 edited black interiors (targetalpha>.98,distance>8px): mean predictedalpha>=.95. Global blank maximumalpha<=.05; all outputs finite and protected pixels exactly preserved.

No new10% halo reduction is required; the hypothesis is recovering thin-line continuity while retaining O's blur gains. Strength0/.3/.7 observations remain available but are not added to this predefined group screen. A regression test explicitly excludes an artificial bad.3 result. Keep every failure and report both arms. The inherited local diagnostic summary uses O pixel control; the broader screen above determines whether a candidate merits confirmation.

Review all8 final local and6 broad native-pixel sheets, including faint contours, crossings, parallel lines/gaps, filled areas and blank edges. Record observations and image hashes. Before deployment any passing candidate still needs a fresh matched training-seed comparison, sealed new-condition evaluation and actual BGRA/ONNX checks. No production checkpoint, Krita plugin, CI artifact or sealed evaluation changes in P.
