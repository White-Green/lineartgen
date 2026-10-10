# AE: broad near-line paper-band loss comparison

AD reduced trained-case fitting failures3→1, passed all trained retention guards,
and cleared corner/parallel halo. Its unfitted junction and straight-line deficit
regressed. Test broad generalization with one changed loss coefficient; never
initialize from fitted development weights. This study is a development diagnostic.

One `band` arm adds paper_band_weight=.1,radius3 to AB's `clearance` configuration.
Keep gap_clearance_weight=.1/radius3 and every other loss/model/data/optimizer value
unchanged. Use AB's20% geometry,10% coverage,70% native mixture,not AC's50% geometry.
Keep width max6,strong corruption probability.5,source-only features/source anchor,
hard-clipped one_pass model with130,605 parameters. Fresh E initialization,
optimizer and EMA; no AD/AA fitted parameters. Whole/partial synthetic rough only.

Batch4,accumulation1,baseLR5e-5×sqrt4=1e-4,warm100,cosine to1e-5 at4k; seed2026100961.
Depths1–6,stage-loss multiplier0 with recursive gradients,downsample-before-Gaussian,
prepared-float protected pixels and all sampling schedules remain fixed. Reuse R
`pixel` and AB `clearance` controls at0/1k/4k; they must reproduce exactly. Main
comparison is E/R/AB/AE. Final raw4k is fixed; no intermediate or EMA selection.

Advance only after AD's final review and trained retention guards pass, all four
trained halo values<=.003,and trained cuts/bridges are zero. Before training,
verify configuration differs only in paper_band_weight,166 AB teacher hashes,
crop resume,first12 teachers and all6 prepared-array depths. Actual4k sampling
records,teacher hashes,depths must match AB. Retain actual6/1k/4k states,four raw/EMA
exports and exact controls. Test loss/defaults/full-state/update and sampling resume.

Evaluate local64 primary and456 broad conditions per model and1,566 main conditions
per model. Main corpus uses the same44 validation identities,not natural sketches.
Keep AB's229 checks unchanged against E/R: geometry blur.5 F1>=.90,cuts<=.02;
corpus blur/strokes.5 halo<=.02/.10; grouped strengths.1/.5 whole/partial halo<=each
reference×1.1+.0005,F1 decrease<=.005,deficit increase<=.01; whole geometry width
MAE increase<=.10,cuts/bridges increase<=.01; black interiors mean>=.95,blank maximum
<=.05,finite/protected exact. AB is the matched descriptive comparison.

Inspect all8 local and6 main final native sheets,including junction/thick taper,
close pairs,deliberate gaps,dense repeated strokes and partial black/blank fields.
Bind images to final weights and record failures. The float-input display pipeline
is not an actual BGRA input-quality or Windows/Krita GUI test. Any eligible model
still requires independent training and fresh fixed synthetic confirmation plus
actual BGRA/ONNX input-scale checks before adoption. No sealed/final-source images,
production replacement or plugin changes in this study.
