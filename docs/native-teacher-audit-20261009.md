# Native teacher audit, 2026-10-09

The same first 256 training-source draws used by the earlier [scaled teacher audit](../../jax-source-mixture-training-20261009/docs/clean-target-audit-20261009.md) were loaded with `native_training_targets=true`, before online geometry replacement and rough augmentation. Source IDs and virtual-depth choices match; the crop field of view changes because actual target resolution changes.

| Quantity, 256 crops | Scaled teachers | Native teachers |
|---|---:|---:|
| Blank crops (maximum alpha <=0.02) | 39 | 0 |
| Mean fraction in gray bands with soft-distance >2 px | 0.0022283 | 0.0001514 |
| Maximum distance inside intermediate-opacity pixels | 26.08 px | 5.39 px |

Intermediate opacity is alpha in (0.02, 0.98). This finite, training-only sample is not a corpus prevalence estimate. The distance transform measures gray regions, not a guaranteed half-width at crop boundaries. The intervention changes widths, context and blank frequency as well as blur; it does not establish blur as the isolated cause of output halos.

All six worst-soft native crops, selected by virtual-depth group, were viewed at original size. Contours are visibly sharper than the previous high-scale worst examples. Some native teachers still contain very faint lines and antialiasing; they should not be treated as perfect binary ground truth. These are worst-case selections, not typical-output illustrations.

[Native audit metrics](../runs/native-target-training-20261009/teacher-audit/metrics.json), [paired audit summary](../runs/native-target-training-20261009/teacher-audit/comparison.json), [reviewed teacher crops](../runs/native-target-training-20261009/teacher-audit/worst-soft-native.png), [visual audit](../runs/native-target-training-20261009/teacher-audit/visual-manifest.json). Training plan and source were frozen separately before this supplementary audit was interpreted. No training settings or selection gates changed.
