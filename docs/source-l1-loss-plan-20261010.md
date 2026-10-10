# O: absolute pixel error with covered native teachers, 2026-10-10

Prepare and commit before inspecting final N coverage outputs. Run **only if neither N candidate passes its frozen broader development screen**. If either passes, prioritize independent confirmation and leave this comparison unrun. No production model, plugin, sealed inputs, or active N numerical files change.

Most recent comparisons change weighting or add structural penalties around squared pixel error. Test a simpler remaining alternative: absolute pixel error. [Zhao et al., 2017](https://research.nvidia.com/sites/default/files/publications/NN_ImgProc.pdf) compare L1/L2 and other losses with the same restoration networks, reporting that the objective can affect artifacts and convergence. [Restormer](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf) also uses L1 for restoration. Neither establishes an advantage for our small lineart network.

Our hypothesis is that absolute error keeps useful gradients for small residual gray regions instead of reducing their gradients proportionally to error. It also changes relative error priorities and the scale versus auxiliary terms; this is not a scale-matched gradient experiment. It is not itself a topology or black/white prior. Correct fractional antialias pixels still have zero loss. Ambiguous thin lines could be deleted rather than averaged, so opacity and connectivity checks remain mandatory. Saturated output gradients are unchanged; the existing physical preclip opacity term remains active.

Add optional `loss.pixel_loss="clipped_l1"`; default remains `clipped_mse`, and `interval_mse` is untouched. Error is `abs(clipped_prediction - target)` with explicitly zero subgradient at an exact match. Both editable denoise and protected identity use this pixel error. Keep the existing denominator `max(strength, noise_weight_floor)` and all tone, edge, contrast and preclip-opacity coefficients. No auxiliary term is added or retuned.

Four methods form paired comparisons:

- `pixel`: reuse N `coverage` raw/EMA 0/1k/4k snapshots, inverse-strength floor.05.
- `flat`: reuse N `flat_coverage` snapshots, floor1.
- `l1`: same N coverage config changing only pixel loss to absolute error, floor.05.
- `flat_l1`: same N flat_coverage config changing only pixel loss, floor1.

Both new arms start from the common **E raw4k** initialization with fresh optimizer/EMA, not from N endpoints. Preserve 130,605 parameters, source-only features/source residual anchor, original-strength one-pass inference, seed2026100961, native corpus teachers, original geometry.2 and conditional field-coverage.125, same mixed rough augmentation, batch4/accumulation1, 4,000 updates, baseLR5e-5×sqrt(4)=1e-4, warmup100, cosine to1e-5 and EMA.999. Rotate update order. Raw4k is the fixed primary endpoint, with no intermediate/EMA selection. Preserve recurrence1–6, stage multiplier0 with recursive gradients and downsample-before-Gaussian.

Before learning, verify the exact value/gradient including zero error and correct antialias targets, mask handling, both strength weightings, existing loss compatibility, real optimizer updates and full-state restore. Replay all N logged clean hashes, resume batches6–11, and compare prepared arrays across all6 depths for all4 configurations. Only `loss.pixel_loss` may differ from the matched config. Bind parent data-audit counts and first12 clean hashes. Retain startup6 actual updates, full states6/1k/4k, raw/EMA exports and fixed code/config/inputs.

At1k/4k evaluate local64+456 conditions per method and require the two reused controls to reproduce their N outputs exactly. At4k evaluate1,562 existing development conditions+4blanks for E initial and4methods, **7,830 outputs**. N counts and recursion depth choices must agree with this study. `real` denotes the existing corpus, not verified natural rough/clean pairs. Additional Gaussian draws are identical for source-only features, not independent samples.

Screen each new method separately against its matched MSE control and E initial:

1. Corpus blur and strokes, whole strength.5: each mean halo at least10% below its matched control, and respectively <=.02/.10.
2. Every available real/geometry clean/tone/blur/strokes strength.1/.5 whole/partial group: halo <=E-initial×1.1+.0005; F1 decline<=.005 and ink-deficit increase<=.01 versus both matched control and E initial.
3. Whole geometry groups: width MAE increase<=.10px and cut/bridge increase<=.01 versus both references.
4. Edited black interiors of preselected717/727: targetalpha>.98, distance>8px, mean predictedalpha>=.95. Blank maximumalpha<=.05, finite outputs and protected pixels exactly preserved.

Keep all failures. Passing one arm does not require the other arm to pass; both arms and all results are reported. The inherited local summary uses the common pixel reference and is only diagnostic; the broader screens use matched controls. Review all8 final local and6 broad native-pixel sheets, particularly faint contours, width consistency, parallel lines/gaps, filled black interiors and paper edges. A promising candidate still requires a fresh matched training-seed comparison, sealed new conditions and actual BGRA/ONNX validation. No result is presumed successful from the literature or from test execution.
