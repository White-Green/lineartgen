# Detail head controlled fitting result

The width16 branch passes the fixed local advance rule and receives visual
support for a broader experiment. Width8 fails the required blur F1 gain. Neither
model is adopted or sufficiently fits the task: all36 nonblank blur conditions
per split still fail the descriptive absolute fit flag.

Both arms completed exactly1,200 updates from the common100k checkpoint with
zero-initialized added outputs. All912 initial predictions matched the control
bitwise. The raw final comparison is against the completed, otherwise matched
`warm_hard` fitting trajectory.

| Branch | Blur F1 gain train / probe | Blur halo change train / probe | Local advance |
| --- | --- | --- | --- |
| Width8,128,141 parameters | +.01425 / +.01161 | −.00445 / +.00188 | No |
| Width16,130,605 parameters | +.09933 / +.10183 | −.05225 / −.04055 | Yes |

Width16 final blur F1 is.53035/.58043 and halo.19525/.18899. Its clean F1 improves
by.16990/.12752 and tone F1 by.17218/.14464. Every family/split halo mean improves.
Train blur ink deficit slightly worsens by.00628; probe improves by.00194. This
side effect remains recorded even though deficit was not a separate local gate.
The improvement is incomplete and does not satisfy the broader campaign gates.

Five fixed final native sheets were inspected. Width16 sharpens several main
lines/corners and reduces broad gray branches in blur inputs; some thin diagonal
evidence improves. Missing thin curves, false joins, merged parallel strokes and
filled gaps persist. Clean input retains sharp main lines and shows a smaller
crossing break; tone input still has fragments and a filled gap. EMA blur retains
substantial defects and does not replace the declared raw primary. Two additional
native sheets were inspected at400; their earlier loss of thin lines was recorded
without changing the schedule or selecting the milestone.

A descriptive strength split shows that most of the gain is at noise.5. Width16
blur F1 gains are+.1867/+.2024 (train/probe) at.5, but only+.0120/+.0013 at.1.
Clean F1 gains at.5 are+.3329/+.2534, versus+.0069/+.0016 at.1. This supports a
substantial improvement in noisy-input handling; it does not demonstrate strong
blur removal at low noise. The predeclared aggregate gate is unchanged. See the
local run's`stratified-diagnostic.json` for all strengths/families/milestones.

The new branch tests a full-resolution feature route and explicit noise/guide
inputs as well as hidden width. This result supports that intervention; it does
not isolate the old two-channel decoder as the sole cause of the artifacts.
Only the width16 architecture advances. The tiny-corpus checkpoint will not be
silently substituted for common100k initialization in the broader comparison.

Validation: all270 tests passed,12 candidate GPU updates at depths1–6 restored
full states exactly, and six unchanged control updates reproduced historical
model/optimizer/EMA tensors bitwise. Every added branch layer updates. The actual
trajectory restored exactly at1,400 and1,200. The completion audit verifies218
pinned source files,5 settings, the source archive,12 finite model triplets and
3,648 finite metric records with exact protected pixels. Numerical source is
`223b28ee4f1349cc76a84e812f8c3075d7f87769`. Later diagnostics/docs do not modify it.

Artifacts: `runs/detail-head-fitting-20261008/`, especially`matched-results.json`,
`completion-audit.json` and`evaluation/step-1200/visual-review/review.json`.
A separate [known-blur input diagnostic](input-recoverability-diagnostic-20261008.md)
confirms retained information in float inputs and illustrates quantization
sensitivity. It does not change the model's local or broader acceptance criteria.
