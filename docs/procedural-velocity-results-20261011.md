# BS: diverse procedural velocity training did not produce a candidate

Both130628-parameter arms completed4000 updates on the identical16000 fresh
procedural teachers. Point MSE failed185/229 broad guards; interval velocity
failed138/229. These counts summarize failures, not a quality ranking. Interval
reduces halo but often removes line coverage or breaks it into dark dots.
Neither is adopted. [Frozen plan](procedural-velocity-plan-20261011.md).

| Full-image strength.5 | E one-pass | R one-pass | Point velocity | Interval velocity |
| --- | ---: | ---: | ---: | ---: |
| Geometry blur F1 | .905006 | .824615 | .578275 | .645904 |
| Geometry blur ink deficit | .083295 | .108294 | .454843 | .418679 |
| Geometry blur halo | .028519 | .025011 | .038758 | .010709 |
| Geometry blur width MAE px | .463556 | .589909 | .899763 | 1.185299 |
| Geometry blur cut rate | .007292 | .083333 | .675000 | .684896 |
| Geometry blur bridge rate | .291667 | .125000 | .020833 | .020833 |
| Geometry strokes F1 | .711129 | .682142 | .641647 | .703612 |
| Geometry strokes ink deficit | .023284 | .031959 | .295950 | .252340 |
| Geometry strokes halo | .153336 | .161461 | .116610 | .061619 |
| Corpus blur F1 | .944971 | .936629 | .818752 | .814848 |
| Corpus blur ink deficit | .048584 | .070756 | .246099 | .284961 |
| Corpus blur halo | .011933 | .008512 | .031076 | .002085 |
| Corpus strokes F1 | .829934 | .843711 | .826061 | .826498 |
| Corpus strokes ink deficit | .015854 | .031481 | .155053 | .175358 |
| Corpus strokes halo | .123276 | .095144 | .107473 | .052839 |

The corpus uses synthetic corruption of clean drawings, not real paired rough
sketches. Geometry aggregates include64 cases per displayed group; corpus24.
Reduced bridging is not useful when valid connections disappear. Point active
black-interior mean alpha is.941613/.945574 for717/727 (below.95); interval1/1.
Blank maximum alpha is.189403 point and.535328 interval, both above.05.

All six native sheets were actually inspected:

- [Blur.5](../runs/procedural-velocity-20261011/development/native-review/blur-05.png): curves, corners, straight lines and junctions largely disappear; interval leaves isolated dots. Taper and parallel lines remain granular.
- [Strokes.1](../runs/procedural-velocity-20261011/development/native-review/strokes-01.png): more continuity, but gray and narrowed strokes remain.
- [Strokes.5](../runs/procedural-velocity-20261011/development/native-review/strokes-05.png): gray scratches versus disconnected dark clumps; neither is clean linework.
- [Corpus](../runs/procedural-velocity-20261011/development/native-review/real.png): fine details disappear and outlines become granular or fragmented.
- [Partial](../runs/procedural-velocity-20261011/development/native-review/partial.png): point black fill is speckled; interval preserves its black core better.
- [Worst blank](../runs/procedural-velocity-20261011/development/native-review/blank-worst.png): sparse interval dots remain even though the sheet looks mostly white.

Audit:30 tests passed in59.19s; numerical commit
`2814a138fe4425eedfe39c8f07880cbe5a46eff5`. Initial16 E/R outputs and8 AZ/BC
outputs reproduce bitwise; six full source batches match BL. All4000 teacher
records match BL exactly (14428 geometry;321 wide stroke,312 black,299 ellipse,
329 polygon,311 paper), including171 independently regenerated target hashes.
Both arms have equal initial parameters, fresh Adam/EMA, identical prepared
inputs, and exact first-six-update replay. States0/6/1000/4000 restore exactly;
all16 raw/EMA exports match the states. All6264 evaluation rows are finite and
protected float pixels exact;3132 E/R reference rows and selected images replay
bitwise. Full evidence is bound in
[final-audit.json](../runs/procedural-velocity-20261011/final-audit.json).

The controlled contrast here is point versus interval velocity. Comparison to
BC changes data and schedule; comparison to E/R also changes initialization,
architecture path and sampler. This does not prove velocity models cannot work
or establish a capacity limit. It does rule out treating the existing4000-update
procedural recipe as a solution to gray halos. Preselected final EMA evaluation
BT was registered before these final outputs were inspected; it is pending.
No independent training-seed or sealed confirmation, BGRA API quality check or
Krita GUI test. The production model, plugin and runtime defaults are unchanged.
