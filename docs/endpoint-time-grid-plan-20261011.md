# BW: fixed endpoint weights with finer beginning/end integration

BV cleared background grain but still produced mixed or collapsed modes. Before
training again, compare its exact final raw6000 noisy/source weights at uniform32,
sin²32 and sin²128 Euler steps. No neural updates or checkpoint selection.
The new grids follow BA's existing node formula; use the same Gaussian draws,
teacher/source/mask arrays and12 groups as BV. This is repeated development.

BV's velocity denominator max(u,1/32) was exact at its uniform1/8/32 nodes but
would artificially slow the last stages of a concentrated grid. This diagnostic
explicitly uses true remaining time u for sampling, guarded only by float32 eps
outside the sampled domain. All prescribed u values exceed that guard. The
training loss and its1/32 cap remain unchanged. Reproduce all24 BV uniform32
physical/clipped prediction blocks and metrics bitwise, proving that the new
denominator does not change the old primary calculation.

Keep the same learned physical endpoint, its[-1,1] bound, recursive clipped
endpoint inserts, source/noisy anchor definitions and exact protected source
projection. No terminal division by zero: every network call precedes tau=1.
The final Euler step uses the remaining interval and lands at its predicted
bounded endpoint, up to floating-point rounding. Do not claim this is a
matched-inference-cost comparison:128 steps costs four times as many calls.

Save72 blocks (two anchors×three schemes×12 groups),64 draws each, physical and
clipped;216 checks (36/method), unchanged nearest-mode RMS<=.15, at least80%
within.20 and nearest-A fraction.1–.9. Review all six native sheets (anchor×depth),
the fixed first-four draws for all three schemes, plus finite/protected checks.
Unit tests cover true-denominator behavior below1/32, exact old uniform replay,
and known-endpoint convergence under the new grids. No criterion changes,
independent training seed, new teacher shapes or sealed confirmation. No plugin,
production model, runtime default or deployment changes.
