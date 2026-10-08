# Fixed 6.25% corruption blend: independent fine-tuning seed

The preselected EMA candidate **fails independent seed20261006 confirmation**.
It reproduces halo reduction but exceeds the unchanged connection-cut limits.
The prospective synthetic confirmation set remains unopened; no model was adopted.

Both endpoints trained for exactly 2,000 updates on the original 10,000-update
schedule from the same original 100k EMA checkpoint. Alpha remained 1/16.
Training completed at 13:52 UTC and evaluation at 13:59 UTC on 2026-10-08.

| Weights | Real strong-strokes halo reduction | Geometry reduction | Strict failures |
| --- | ---: | ---: | ---: |
| EMA, preselected primary | 12.734% | 14.143% | 3 |
| Raw sensitivity | 17.885% | 17.535% | 0 |

EMA failures are all connection cuts: +1.25 percentage points for geometry strong
strokes at noise .1, and +1.71875 points for geometry mild blur overall and whole
images. The latter two overlap; these are two conditions, not three independent
defects. The frozen limit is +1 point. All other strict retention, width, location,
background, bridge, and F1 checks pass. Raw is not substituted after the result:
the primary seed's raw blend had 35 failures and EMA was fixed beforehand.

EMA paired halo opacity changes have cluster 95% intervals excluding zero:
real −0.018729 [−0.021650, −0.015837], 46 cases/12 source clusters;
geometry −0.030092 [−0.034563, −0.025314], 128 cases/32 kind/variant clusters.
These intervals are conditional on fitted models and not adjusted for selection.
Blank-paper maximum mean opacity is .0004893 EMA and .0002032 raw, versus
.0104384 and .0040020 for the matched controls.

Eight native-pixel sheets were actually inspected. Strongly blurred thin lines
still frequently disappear in both candidate and control. Parallel strokes and
intended gaps can merge. Real contours show modest fringe improvement but retain
substantial missing/wrong details; five iterations can strengthen wrong joins.
The images do not justify overriding the failed numerical gate.

Both complete training states and exports were audited. The historical control
reproduces all 1,562 metric records exactly in EMA and raw. Endpoint forward checks
are bitwise exact in 36 comparisons; all eight metric reports contain finite
defined values and exact protected pixels. Construction serialization is exact.
Records, image hashes, and decisions are under
`runs/corruption-blend-seed20261006-evaluation/`.

Preparation initially refused an untracked 51-byte ONNX runtime session-cache
file created during an earlier CPU export. The cache and failed log were retained
inside the run, and freezing was resumed with the already constructed models.
No candidate inference or parameter changes occurred during this recovery.

Next development hypothesis: the current bias-free guide convolutions receive
white as +1, so their response on paper is coupled to the response to faint ink.
Test an ink-centered guide, zero on white, and a square-root ink transform that
amplifies faint signals. Keep parameters, original-source guidance, loss weights,
noise generation, recursion, and learning-rate schedule otherwise unchanged.
This is a new experiment; the failed independent-seed result remains a rejection.
