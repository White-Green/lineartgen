# Corruption support and exposure audit

This read-only diagnostic counts the actual indexed augmentation draws for R's completed4k training and the existing development corruption parameters. It changes no model, condition or threshold. T remains fixed. [Predeclared count plan](../runs/corruption-support-20261010/plan.json), [results and latent draws](../runs/corruption-support-20261010/results.json), [script](../runs/corruption-support-20261010/audit-script.py).

R/S/T use opacity.25–1, blur sigma0–2, maximum displacement2. The development strong regime uses opacity.2–.45, sigma1.5–2.5, maximum displacement3. Among44 distinct strong blur corruptions,5 have opacity below.25 and18 have sigma above2;22 violate at least one of those bounds. Strokes shares these draws and additionally has a larger configured displacement limit. That last statement is about the sampling parameter; it does not mean every warped pixel exceeds the training displacement bound. Mild corruption draws lie inside the old opacity/sigma support.

The local blur panel has opacity.3/sigma2, within training support. Therefore out-of-support corruption cannot explain all observed failures. Multiple strengths and masks reuse the same corruptions, and are not additional independent source draws.

Actual4k training contains16,000 items, of which3,197 are procedural geometry. The table counts enabled augmentation draws before editing masks; these are not loss-weighted pixel counts. “Severe region” here is the fixed opacity<=.45 and, for blur/strokes, sigma>=1.5.

| Family | All augmented items | Severe region | Geometry augmented items | Geometry severe region |
|---|---:|---:|---:|---:|
| Tone |4,244|1,180|860|260|
| Blur |4,327|300|884|53|
| Strokes |4,187|265|793|53|

Thus the4k comparison has only53 geometry images with simultaneously low opacity and strong blur, and53 with strong strokes. These include all generated geometric types, multiple paths per image and some blank targets; the audit does not claim53 thin-line examples or53 independent updates with a nonzero editing loss. The shape generator, masks, strengths and tone weighting further affect useful exposure.

This suggests testing the allocation of training examples before attributing every remaining faint-line failure to insufficient model capacity or a missing loss. One possible next comparison is a fixed mixture of the original augmentation and a stronger regime, while retaining the same teacher stream and mixed family probabilities. It must be predeclared and compared with a matched control. The count audit alone is not evidence that such training will improve quality, and does not justify changing T midway or relaxing its safeguards.
