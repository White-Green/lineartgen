# CA: bounded endpoints and time modulation did not transfer to 512px geometry

Both recipes completed6000 geometry updates from their respective toy6000 weights. Plain fails81/88 fitting criteria; Fourier FiLM fails61/88, against52/88 for the frozen BC point-velocity reference. Neither is adoptable. [Frozen plan](geometry-endpoint-plan-20261011.md).

At primary recursion depth3, averages across all four fixed Gaussian draws:

| Case | BC F1 / deficit | Plain F1 / deficit | Modulated F1 / deficit |
| --- | ---: | ---: | ---: |
| 999 | 0.830584 / 0.194004 | 0.503854 / 0.254816 | 0.557761 / 0.429790 |
| 1095 | 0.930086 / 0.066400 | 0.831114 / 0.097410 | 0.845933 / 0.097953 |
| 1287 | 0.937094 / 0.054826 | 0.736611 / 0.039849 | 0.911515 / 0.045176 |
| 1383 | 0.991897 / 0.085070 | 0.503888 / 0.069281 | 0.967479 / 0.098964 |
| 807 | 0.448114 / 0.648264 | 0.797191 / 0.092070 | 0.414754 / 0.671943 |
| 903 | 0.556131 / 0.285215 | 0.793496 / 0.046055 | 0.526084 / 0.309216 |
| 1191 | 0.165073 / 0.824911 | 0.227840 / 0.804216 | 0.051999 / 0.949274 |
| 1479 | 0.703419 / 0.411060 | 0.830577 / 0.045984 | 0.571311 / 0.560222 |

Cases999/1095/1287/1383 are fitted curve/corner/parallel/gap examples;807/903/1191/1479 are unfitted straight/thin/junction/taper probes. F1 higher and opacity deficit lower are better. The fixed fitting thresholds are unchanged; counts are diagnostics, not a quality score.

Modulation preserves the parallel pair (bridge rate0), unlike plain (1), and reduces the ragged thick artifacts of plain. However, fitted curve F1 is.557761 versus BC.830584, curve cuts are1 versus.166667, and unseen junction deficit is.949274 versus.824911. The tapered stroke becomes too thin. The narrow gap remains incorrectly connected in all modulated draws. There is no broad improvement over BC.

F1/deficit retention failures are24/64 plain versus BC,55/64 modulated versus BC and41/64 modulated versus plain. Both have clean distant backgrounds but that alone does not preserve a line. Modulated training loss being lower did not predict acceptable free-running restoration.

All five sheets were inspected: [train native](../runs/geometry-endpoint-20261011/evaluation/train-native.png), [train overview](../runs/geometry-endpoint-20261011/evaluation/train-overview.png), [probe native](../runs/geometry-endpoint-20261011/evaluation/probe-native.png), [probe overview](../runs/geometry-endpoint-20261011/evaluation/probe-overview.png), [partial native](../runs/geometry-endpoint-20261011/evaluation/partial-native.png). Whole-image comparisons show all four fixed draws at depth3. The partial sheet contains one fixed noise draw for each of four train cases at depth6. Gray protected halves remain the rough source, as required.

Audit:14 tests passed in39.39s. All first6 updates and full states0/6/2000/6000 replay/restore exactly. All16 raw/EMA exports match state;6000 input records are complete and unique. All25 BC physical/clipped blocks replay bitwise. All75 evaluated blocks (288 whole-image metric rows plus partial diagnostics) are finite and protected pixels are exact. [Final audit](../runs/geometry-endpoint-20261011/final-audit.json). Numerical commit `4541e10`.

This compares complete recipes: plain and modulated have different toy-trained weights and130628/134688 parameters; BC has different pretraining, velocity loss and solver. It does not isolate FiLM alone or prove insufficient capacity. No sealed conditions, independent training seed, production model, runtime default, Krita plugin or deployment changed. CB conditional transport was fixed before these results and remains a separate toy diagnostic.
