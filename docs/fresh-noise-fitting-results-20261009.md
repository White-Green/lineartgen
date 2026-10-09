# Fresh-noise single-case fitting results

Resampling Gaussian noise every update substantially improves restoration under another noise draw, but **does not solve the fitting problem**. All32 predeclared eight-draw conditions still fail the absolute screen at800. Thin-line continuity improves while lines remain too wide; intended gaps are filled with gray ink. This diagnostic cannot supply a deployable candidate or establish a sufficient minimum model size.

The [plan](fresh-noise-fitting-plan-20261009.md) holds the four independent130,605-parameter models, already seen training images, common preclip-opacity.3 initialization, loss,800-update budget, batch1, recursion1–6, stage weighting0 and sqrt(batch) LR fixed. Only the Gaussian training draw changes each update, after downsampling. Numerical pin: `9a630197408e7f7e1009df0889fb2e7144ab6907` (245 files,7 settings). The comparison uses eight independent noise draws on each of the same four training images; these are not new drawings or the sealed fresh confirmation.

Mean over eight draws, raw one-step depth3 at800:

| Case | Fixed-noise F1 | Fresh-noise F1 | Fixed ink deficit | Fresh ink deficit | Fixed halo | Fresh halo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Straight | .668697 | .823870 | .323653 | .020472 | .031055 | .041680 |
| Thin | .415627 | .627157 | .559392 | .032008 | .044738 | .100460 |
| Corner | .873673 | .942927 | .137047 | .011818 | .015184 | .013639 |
| Gap | .747692 | .851200 | .209690 | .032733 | .031554 | .035819 |

The fixed screen remains F1α50 >=.95, ink deficit <=.05 and halo excess <=.02. Fresh-noise passage is0/32 at0/200/800. On the original cached inputs it also passes0/4 primary depth3 cases and0/24 all-depth conditions at800; F1 is.807598/.643137/.941486/.852692 for straight/thin/corner/gap. The old-transfer draw is also retained as a separate diagnostic. No checkpoint, noise draw, iteration count or averaging of outputs is substituted for the declared result.

All five required native sheets were actually inspected. Fresh-noise learning removes much of the original branching residue and restores a more continuous thin line under new noise, but that line is visibly too thick and has a soft gray fringe. Depth1 remains noisier than deeper recursion. The deliberate gap remains connected across the displayed conditions. Sparse faint background marks remain. Central160px crops do not describe the entire512px image, so visually attractive central segments are not treated as whole-image successes.

The additional path audit reconstructs all four original target rasters bitwise from their stored generation keys. Across the eight draws, thin-line cut rate decreases from.833333 for fixed-noise learning to0 for fresh-noise learning, but normal-profile width error increases from.3631px to1.6234px. Its initial-model width error was2.6143px. The respective fresh width errors are.7910px straight,.3393px corner and.6168px gap. These measurements explain why improved ink retention alone is insufficient.

The intended gap is13.7384px between its generated endpoints. For the original cached input, the middle half of that gap has zero target opacity but mean predicted opacity.3466 with fixed-noise800 and.4575 with fresh-noise800. Across the eight new draws, all eight fresh predictions connect the two strokes at10% and25% opacity; six do at37.5%, three at50%, and none at62.5%. Mean gap-middle opacity is.3366. The existing three-threshold bridge rate improves from.875 for fixed-noise to.375 for fresh-noise, but this still fails the required separation. Gray bridges can survive an apparently favorable50%-threshold F1.

Noise sensitivity also falls: mean pixelwise opacity standard deviation within teacher ink decreases from.3485/.3232/.1745/.2775 in the fixed specialists to.0885/.1215/.0438/.0972 in fresh specialists, respectively. This is descriptive variation over eight correlated draws of each seen image, not ensemble inference or evidence about unseen drawings. The output becomes more stable while retaining a systematic width/gap error.

Validation: prior cached six-update full-state replay exact; all four initial predictions exact; fresh six-depth preflight finite with exact restore; actual1/200/800 model/optimizer/EMA restores exact; eight old-transfer comparisons exact at each evaluation milestone. There are72 cached-condition predictions and324 transfer predictions across0/200/800. The104 unchanged core/test files retain the previous298-pass manifest; new orchestration passes Ruff/compilation and actual-path checks. Results, per-condition scores, source/weight bindings and native reviews are under `runs/fresh-noise-fitting-20261009/`; the128-row descriptive geometry audit is under `runs/single-case-path-audit-20261009/`.

This shows that the impressive fixed-input fit was partly specific to the sampled noise. It does not show that resampling is a new fix for ordinary training: the ordinary training path already draws fresh noise. It narrows the remaining issue to learning clean width and separation under varying inputs in this objective/initialization/budget. Further loss or architecture changes should be distinguished from insufficient optimization; the current result alone cannot decide among them. Production weights and Krita/runtime behavior remain unchanged, and the sealed fresh confirmation is unopened.
