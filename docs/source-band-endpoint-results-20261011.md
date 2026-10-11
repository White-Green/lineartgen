# CC: source-band weighting gives small fitted gains but loses unseen thin lines

Both arms continued the actual CA modulated6000 state to7200 at LR1e-5. Source-only band weighting fails53/88 fitting criteria versus58/88 for the matched continuation and61/88 initially. It is not adoptable; the unseen thin-line deficit worsens markedly. [Frozen plan](source-band-endpoint-plan-20261011.md).

Primary depth3; all four fixed noise draws averaged:

| Case | Initial F1 / deficit | Ordinary7200 F1 / deficit | Band7200 F1 / deficit |
| --- | ---: | ---: | ---: |
| 999 | 0.557761 / 0.429790 | 0.704261 / 0.287519 | 0.713174 / 0.286462 |
| 1095 | 0.845933 / 0.097953 | 0.965822 / 0.025907 | 0.971391 / 0.023061 |
| 1287 | 0.911515 / 0.045176 | 0.933635 / 0.028647 | 0.943422 / 0.032146 |
| 1383 | 0.967479 / 0.098964 | 0.922738 / 0.084245 | 0.927734 / 0.075406 |
| 807 | 0.414754 / 0.671943 | 0.513813 / 0.625171 | 0.515224 / 0.622372 |
| 903 | 0.526084 / 0.309216 | 0.205919 / 0.730277 | 0.114174 / 0.812319 |
| 1191 | 0.051999 / 0.949274 | 0.052800 / 0.956736 | 0.059473 / 0.950008 |
| 1479 | 0.571311 / 0.560222 | 0.513564 / 0.622596 | 0.522941 / 0.612600 |

The band uses9x9 max-pooled observed source ink and averages its normalized risk equally with the ordinary active-pixel risk. It uses no clean-teacher weights. With unconstrained separate pixel predictions it preserves the same conditional risk minimizers; finite shared-network optimization still changes.

Against the matched ordinary continuation, fitted curve F1 improves only.704261 to.713174; corner F1 improves.965822 to.971391 and corner cut rate.5 to.333333. Parallel lines stay separated, but the narrow gap remains connected. Unseen thin-line F1 drops.205919 to.114174 and opacity deficit rises.730277 to.812319. Its deficit was.309216 before either continuation. Unseen junction deficit remains about.95. Tapered strokes are too thin and develop breaks.

Retention failures:24/64 ordinary versus initial,21/64 band versus initial and9/64 band versus ordinary. Improvements over CA6000 cannot all be attributed to weighting: ordinary continuation already supplies most of the curve/corner gain.

All five sheets were inspected: [train native](../runs/source-band-endpoint-20261011/evaluation/train-native.png), [train overview](../runs/source-band-endpoint-20261011/evaluation/train-overview.png), [probe native](../runs/source-band-endpoint-20261011/evaluation/probe-native.png), [probe overview](../runs/source-band-endpoint-20261011/evaluation/probe-overview.png), [partial native](../runs/source-band-endpoint-20261011/evaluation/partial-native.png). Whole-image sheets include all four draws; partial has one fixed draw per four train cases. Gray protected halves stay unchanged.

Audit:12 tests passed in40.62s. Full parent model/Adam/EMA state copied exactly; first6 updates across all6 recursion depths replay exactly. All states6000/6006/6600/7200 fully restore;16 raw/EMA exports match state.1200 input records complete and unique. All25 CA physical/clipped blocks and96 metric rows replay bitwise. All75 evaluated blocks are finite with protected pixels exact. [Final audit](../runs/source-band-endpoint-20261011/final-audit.json). Numerical commit `10845e8`.

No broad data or new Gaussian seed was tested. No independent training seed, sealed confirmation, production model, plugin, runtime default, export or deployment changed. These results do not justify broad training with this weighting recipe yet.
