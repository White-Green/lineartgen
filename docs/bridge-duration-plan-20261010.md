# AY: Equal continuation of AW and AX to30,000 updates

Declared after AX. AW gives smaller shape errors but collapses weak-noise modes;
AX passes mode balance in all12 groups but has larger shape error. Continue both
actual6,000-step states to30,000, adding24,000 updates per arm. This is a fitting
comparison on the same two ambiguous templates, not broad-data validation.

Restore model, Adam moments/counts and EMA. Config train.steps becomes30,000
and learning_rate_decay_steps is explicitly6,000, preserving the exact original
schedule and its1e-5 floor for every additional update. Verify schedules at
0/100/5999/6000/12000/30000 and compare the full initial states to independent
parent restores. No optimizer reset, new teacher, noise, mask, strength, objective,
architecture, recursion or sampler. Each arm calls its own frozen parent's batch
function at indices6000..29999; equal seeds and conditional OT persist, differing
only in the previously declared time distribution. Random draws remain after
resizing. The64px depth1–3 limit does not change production512px depth1–6.

Retain full states6000/6006/12000/30000 and16 raw/EMA exports. Replay the first6
updates independently from each parent and require exact metrics and full-state
identity. Unit tests cover restoring both branches, retained schedule/counts,
next-update identity, and ordinary checkpoint resume. Full snapshots are audited
against the corresponding raw/EMA state. Final raw30,000 is fixed for both;
intermediate snapshots are not candidate-selection opportunities.

Reuse all180 AT/AV/AW/AX output blocks with hashes. Evaluate each new arm on
12groups×64draws×1/8/32steps, keeping all252 blocks. Apply the same36 toy checks
to each arm's32-step sampler (72 total): mean relative nearest-endpoint RMS<=.15,
fraction within.20>=.8, nearest-A fraction .1–.9 per group. Report every failure
by arm; do not choose a different checkpoint or lower standards. Finite physical
outputs and protected exactness mandatory. Inspect3 fixed native sheets showing
A/B/rough, each6,000-step baseline, and first4 draws of each30,000-step arm.

Even a pass is only learned toy feasibility. It would require a separate broader
synthetic-data experiment, independent training seed and untouched conditions,
then actualBGRA/ONNX quality checks. No plugin update or deployable artifact.
