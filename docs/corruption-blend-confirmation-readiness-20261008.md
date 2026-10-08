# Fixed-blend confirmation preparation

The selected development candidate is `corruption_blend0625` EMA: 15/16 of
the strokes-trained source/ink model and 1/16 of the mixed-corruption model,
both at exactly 2,000 updates on the original 10,000-update learning-rate clock.
The secondary fine-tuning seed is 20261006. The ratio is not reselected.

`scripts/confirm_corruption_blend.py` seals four exact EMA checkpoints only after
both seeds pass the unchanged strict development gates and recorded visual review.
It then evaluates the already planned 2,784 prospective synthetic cases at 1, 2,
and 5 denoising iterations. Each seed uses its own matched control. Zero-strength
controls are checked for identity and excluded from active quality summaries.
Iteration counts are assessed separately; only one iteration is the primary gate.
Case results are resumable and bound to model, code, settings, and input hashes.
No fresh cases have been generated as part of this preparation.

The existing generator fixture and two new aggregation fixtures passed: 3 tests
in 1.90 seconds, `/tmp/corruption-confirmation-fixture-tests-20261008.log`.
They do not read the prospective data. The full inherited 245-test validation is
recorded separately; it was not rerun for these orchestration additions.

The actual primary candidate exported successfully to dynamic ONNX. On 27
synthetic BGRA API comparisons (three shapes, strengths 0/.1/.5, iterations 1/2/5),
CPU ONNX Runtime and CPU JAX differed by at most 0.000017643 in continuous output
and one alpha level after conversion. Only five output bytes differed in total.
The odd image dimensions were 145×129 and 513×385; the third shape was 512×512.
The inputs include transparent thick/thin strokes and fully/partly protected
lineart. This is runtime parity evidence, not Windows DirectML or Krita GUI testing.

The first attempt's exact protected-float assertion exposed sampler roundoff:
the NumPy iteration lacks JAX's residual-output zero-mask guard. The maximum
measured error was 5.96e-8, with no material visual impact established. Its failed
attempt and source were preserved under
`runs/corruption-blend-runtime-20261008-roundoff-first-attempt`; the complete
measurement is in `runs/corruption-blend-runtime-20261008/runtime-check.json`.
The runtime guard will be fixed in an isolated checkout so the active training's
pinned source remains unchanged.

The candidate is not adopted. Second-seed quality, fresh conditions, and native
pixel inspection still determine whether it is eligible.
