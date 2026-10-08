# Interval loss: rejected at the 2k screen

Both EMA candidates fail the unchanged 2k quality screen. The proper pre-clip
objective reduces some faint-profile missingness but does not achieve the
required halo reduction, and increases distant background ink. Neither model
continues to 10k. No production model, second fine-tuning seed, or fresh
confirmation set was used for these rejected candidates.

The [plan](interval-loss-study-20261008.md) was executed from source
`47d454e4ed1f62d7ff0adad8bd76d86603e44c32`: 189 source files and 13 settings/audit
files frozen. The common 100k EMA, seed20261007, synthetic corruption, recursion,
stage weights, sqrt(batch) LR and 10k schedule match the paired completed hard
baselines. Initial inference matches all 1,562 baseline records exactly.

Before training, all 238 tests passed, including ONNX validation. Both candidates
completed actual GPU updates at depths1–6 with exact checkpoint restoration.
Disabling the new loss reproduced all model, optimizer and EMA arrays of the
two previous hard baselines after six GPU updates. The twelve fixed training
batch diagnostics made no weight updates or coefficient changes. Parameter
gradient norm ratios (new/old) had medians1.786/1.704 and maxima19.511/18.976;
the existing global gradient clipping remained unchanged. The largest ratio
occurred at depth6: norms54.145/54.069 versus2.775/2.849.

| EMA arm | Real strong-strokes halo gain | Geometry strong-strokes halo gain | 2k screen |
|---|---:|---:|---|
| interval_control | -11.34% | -6.93% | insufficient halo gain |
| interval_ink_guard | -2.14% | +2.08% | insufficient gain; mild-blur bridges +4.17pp |

Positive halo gain means improvement relative to the conventional hard control.
Real means real clean lineart corrupted synthetically, not human rough sketches.
The required screen gain is >=5% in either primary domain, with the existing
retention/topology tolerances. Raw control also fails halo; raw ink_guard gains
1.84%/6.26% but fails the same bridge check. It is not substituted for EMA.

On strong geometry blur, EMA profile missing fraction decreases by4.31pp and
4.12pp, ink deficit by.50pp and.24pp. However, distant excess opacity increases
by.000947/.000856, and position MAE by.111/.108px. Both are beyond the stricter
final tolerances. These 2k final diagnostics contain90/79 overlapping failures
(raw85/94); they are not independent defect counts or actual 10k measurements.
Both EMA arms have56 final background-excess failures across overlapping strata.

Eight native-pixel sheets were inspected: all24 EMA geometry crops, eight raw
blur crops, real cases479/480/640 and raw480, four one-vs-five-step crops, and a
partial blank five-step image. Strong blur still erases faint lines. Parallel
lines remain merged and deliberate gaps fill in. Real crops retain gray
fringes and false marks. Blank images expose added speckles. Repeat inference
can sharpen existing wrong connections. Better missing-profile statistics alone
do not establish satisfactory lineart.

All eight metric files are complete and their defined scalar metrics finite.
Full states and inference exports were audited. The root
`runs/interval-loss-20261008-screen` contains the paired analysis, gates,
checkpoint audit, native image manifests, rejection review and completion audit.
Training ran00:24–01:36 UTC and numerical evaluation finished around01:40.
The initial diagnostic watcher hit a startup race before the controller wrote
its status file; restarting only the watcher fixed it. Training was not
interrupted. The original log and recovery record are retained.

The clipping dead-gradient issue is real, but removing it is insufficient for
the desired quality in this experiment. The next hypothesis is to expose the
already-available original rough input as a separate condition alongside its
noisy version. This changes the model's available information, so it must be
tested separately from loss changes. It does not provide the clean teacher.
