# AW: Condition-preserving minibatch transport

Declared after AV: direct velocity reduced high-noise shape error but still
failed27/36 toy checks. Compare the same CNN, loss, sampling, optimizer,
initialization and6,000 updates, changing only the pairing of initial noisy
states with the two clean endpoints during training.

Reuse AV's exact16-item minibatches (seed2026101088), including guides, masks,
strengths, Gaussian draws after resizing, times and original label multiset.
Within each strength/mask group preserve the count of each target. Assign A to
states with largest dot(z-source,A-B) over active root pixels. This sorting is
the exact minimum-cost assignment for two repeated targets and squared root
endpoint distance. Ties use index order. Assignment uses no model output or
path time, and supplies the same selected teacher to all recursive stages.
Groups are small (on average4 items); this is minibatch conditional OT, not
population OT or a globally consistent coupling of all pyramid noises. Coarse
noise is unchanged and independently generated as in AV. Do not claim otherwise.

Test against SciPy Hungarian assignment for each group using float64 costs,
verify non-increasing cost versus independent pairing, and preserve exact input
noise, source, masks, times, condition-specific labels and protected targets.
Audit192 indexed minibatches (64 spread over the6,000 updates ×3 depths).
The sampler is exactly AV's teacher-free direct-velocity sampler; no teacher
or target assignment runs at inference. Recursion1–3 is the64px toy constraint;
production512px recursion1–6 remains unchanged. Preserve stage-loss weighting
and recursive gradients, sqrt(batch) LR, and all AV training settings.

Use the same random initial parameters, not AV fitted weights. Reuse all108
hash-bound AT/AV output blocks. Train only the transport arm; retain full states
6/2000/6000 and6 raw/EMA exports. Final raw6000 fixed. Evaluate12 groups ×64draws
×steps1/8/32. Apply the unchanged36 criteria to transport32: mean nearest-mode
RMS<=.15, fraction within.20>=.8, mode-A fraction .1–.9 in every group. Finite
physical outputs and exact protected pixels required. Keep all144 output blocks
and inspect3 fixed native sheets. No sample, checkpoint or EMA selection.

This is still a two-template mechanism study. Do not adopt toy weights or
change the plugin. Even a pass needs broader synthetic shapes, an independent
training seed and held-out conditions, then actual BGRA/ONNX quality checks.
