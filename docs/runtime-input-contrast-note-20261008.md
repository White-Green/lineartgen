# Low-opacity information at the BGRA boundary

A separate handcrafted BGRA diagnostic found that the current `image_scale=1.1`
preprocessing clips black scribble alpha levels 1–11/255 to exactly white. Alpha
12/255 becomes normalized ink opacity .001765; alpha25/255 becomes .057843.
With scale1.0, all nonzero alpha levels survive. The analytic cutoff before
quantization is `(1 - 1/1.1)/2 = .04545` input opacity.

The source guide currently comes from that same prepared image. Thus this loss
of faint information occurs before both Gaussian noise and guide encoding; changing
the guide representation cannot recover pixels already clipped to white.
This is measured preprocessing behavior, not evidence that scale1.0 improves a
learned model. Training normalization and input distributions must be considered
before changing the deployed scale.

The current development evaluations consume stored normalized augmented arrays
directly. Augmentation is applied after training-image normalization. Their
native-pixel output sheets exercise output BGRA conversion, but not input BGRA
conversion. The actual-checkpoint ONNX parity test exercises both input and output
conversion and confirms backend agreement; it does not resolve this quality issue.

Evidence: `runs/ink-guide-plan-20261008/runtime-input-contrast-audit.json`, generated
without model fitting or prospective confirmation cases. No training settings,
current model selection rules, runtime defaults, or production assets were changed.
Before adopting a candidate, add a quality check using explicit synthetic BGRA
inputs, especially faint/thin/blurred strokes, so array-space improvements are not
mistaken for validated end-to-end Krita behavior.
