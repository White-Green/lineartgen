# AT: learned path can sharpen the ambiguous mean but fails mode fidelity

Both 130,628-parameter CNNs completed6,000 updates. The fixed bridge32 result
fails30/36 toy checks. All outputs are finite and protected pixels exact, but
the mean relative distance threshold .15 is missed in every one of12 groups.
No broad training, adoption or plugin change follows from this toy.

| Depth | Whole .1 RMS / nearest A of64 | Whole .5 RMS / nearest A of64 | Partial .1 RMS / nearest A of64 | Partial .5 RMS / nearest A of64 |
| --- | --- | --- | --- | --- |
|1|.21366 /64|.87538 /59|.33664 /1|.81008 /50|
|2|.32063 /1|.88889 /60|.16958 /1|.85997 /37|
|3|.40554 /1|.89794 /56|.20020 /5|.79927 /44|

RMS is distance to the nearer clean endpoint divided by the norm of half the
endpoint difference. The exact conditional mean has1 and an endpoint has0.
Nearest A is not a claim that the sample is a valid A: at .5 none of the384
bridge32 outputs across the six groups reaches relative RMS .20.
Direct1 averages stay near the gray mean, with RMS
.96389–1.00150. They have lower teacher-averaged MSE than bridge32 in all12 groups,
illustrating the difference between paired error and selecting a crisp mode.

All three final native sheets were inspected. At .1, many bridge samples look
more like one black/white endpoint, but the preferred mode changes with recursion
depth and almost all draws in a group choose the same one. At .5, irregular gray
centers and residual paper noise remain. Partial protected rough areas are
intentional. Saved predictions include final unclipped extrema; the final clip
does not fix these learned artifacts. The oracle's earlier success did not
establish that this CNN/objective/budget would learn its field accurately.

16 tests passed in19.94s. Before freezing, tests caught missing independent-noise
configuration and two fixture issues: exact equality at roundoff and an update
test combining zero warmup LR / sampled time0 with an expected progress update.
The final fixture explicitly exercises nonzero time and LR. All failed logs are
retained. Initial arm tensors/outputs match exactly, arm states are separate,
state and next-update restore pass. Actual GPU states6/2000/6000 and12 raw/EMA
exports are exact. All72 final blocks (2 methods ×12 groups ×3 step counts) and
their hashes are stored. Numerical commit `9880d11`.

- [Frozen plan](learned-bridge-toy-plan-20261010.md)
- [Final audit](../runs/learned-bridge-toy-20261010/final-audit.json)
- [Depth1 comparison](../runs/learned-bridge-toy-20261010/evaluation/depth-1-native.png)
- [Depth2 comparison](../runs/learned-bridge-toy-20261010/evaluation/depth-2-native.png)
- [Depth3 comparison](../runs/learned-bridge-toy-20261010/evaluation/depth-3-native.png)

The test has only two memorized shapes, new Gaussian draws, and64px depths1–3.
It does not validate512px depths1–6, ordinary sketches, BGRA/ONNX, independent
training seeds or latency. Thirty-two recursive evaluations cost32×depth UNet
calls, versus depth calls for direct1. The next diagnostic holds these final
weights fixed and increases solver steps to distinguish finite-step error from
learned-field limitations. AT's fixed32-step failure remains unchanged.
