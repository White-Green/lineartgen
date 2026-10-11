# BV: bounded endpoints clear background grain but still mix or collapse modes

Both130628-parameter arms completed6000 updates on the original two-shape toy.
The noisy anchor fails22/36 criteria; the source anchor18/36. Neither is a
candidate. Source anchoring improves weak-noise shape error but worsens the
strong-noise average and collapses several weak-noise conditions to one shape.
[Frozen plan](anchored-endpoint-plan-20261011.md).

| Mean across six depth/mask groups at32 uniform steps | AV velocity control | Noisy endpoint anchor | Source endpoint anchor |
| --- | ---: | ---: | ---: |
| Strength.1 nearest-mode RMS | .248287 | .166042 | .065419 |
| Strength.5 nearest-mode RMS | .485600 | .393049 | .560199 |
| Strength.1 fraction within.20 | .460938 | .778646 | .921875 |
| Strength.5 fraction within.20 | 0 | .424479 | .408854 |

Source weak-noise partial cases select A64/64 at every depth; depth2 whole-image
selects A63/64. That violates the mode-frequency criterion even where shape error
is zero. At strength.5 the source method's group RMS spans.444207–.727397,
well above.15. Outputs can combine part of the one-bar shape with part of the
two-bar shape, creating holes or interrupted gaps rather than a valid sample.

All three sheets were inspected:
[depth1](../runs/anchored-endpoint-20261011/evaluation/depth-1-native.png),
[depth2](../runs/anchored-endpoint-20261011/evaluation/depth-2-native.png),
[depth3](../runs/anchored-endpoint-20261011/evaluation/depth-3-native.png).
Both new methods produce cleaner white backgrounds than the grainy AV control.
This is a change of endpoint parameterization, interval weighting and bounded
sampler compared with AV; background gains cannot be attributed to source anchor
alone. Noisy versus source is the matched anchor contrast. Crisp black/white
pixels do not guarantee coherent geometry or the intended conditional distribution.

Audit:23 tests passed in41.54s. Initial parameters match AV; the noisy endpoint
at all three depths replays AT's physical endpoints bitwise. The depth1 anchor
shift, AA/white/black interval gradients, time cap, teacher-free sampler,
protection and gradient through the smaller stage were tested. Fresh Adam/EMA,
first6 updates replay exactly, all full states0/6/2000/6000 restore exactly,
and16 raw/EMA exports match their states. The6000 input/hash/depth/label records
match BU exactly. All36 AV physical/clipped output blocks and metrics reproduce
bitwise. All108 evaluated blocks are finite with protected pixels exact; primary
32-step physical endpoints are within[-1,1].
[Final audit](../runs/anchored-endpoint-20261011/final-audit.json) binds the evidence.
Numerical commit `b224283`.

Only two memorized synthetic teachers with reused development Gaussian draws
were tested. No new shapes or independent confirmation and no deployed model,
Krita plugin or runtime-default change. These weights require the custom
endpoint sampler; generic runtime loading is not an implementation of this recipe.
The next inexpensive diagnostic is fixed-weight integration near the start/end
of the path, with explicit treatment of the uniform sampler's remaining-time cap.
