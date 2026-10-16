# FM: narrow residual contours

Minimum-width opening on fixed FH zero contours removes the residual traces in the five known crops without visibly reopening the inspected branches/crossing. This motivates a training prior, not an inference-filter adoption or a broad quality claim.

All10 native pages were inspected. A0.5px disk opening removes most of oldtrain36's middle trace and probe37's tiny remnant. Scene2's lateral duplicate requires1px to remove most of it and1.5px for near-complete removal. Scene1's connected branch and scene7's crossing bridge remain at all three widths; larger openings round/shorten tapered endpoints and remove scene1's small lower hook. Float/uint8 behavior agrees.

Across512 training records,480 contain paths with111408 width samples and57922.2584 units of segment length. Only the48 taper records have normalized widths below1.5px. Below0.5/1/1.5px are1028/2112/3380 samples and818.972/1671.489/2645.953 units of length. Thus a blanket minimum-width penalty conflicts with intended taper geometry. Use a teacher-derived exemption rather than assuming all strokes should exceed1px.

Boundary-gradient medians across these crops/encodings range0.7195–0.9828. They do not identify a gradient-norm defect as the sole cause; the unwanted contours have actual small geometric width. Removed ink area is descriptive and includes legitimate caps, not an error metric.

Analytic band, constant-sign, opening subset/idempotence and plane-area checks passed; plane alpha error0.005<=1/32. Ruff passed. CPU evaluation completed in7.882008s with exit0,2026-10-16 07:19:48–07:19:55UTC.50 crop outputs are bound to the fixed FI depth6 fields, with8px halo removed from measurements. Frozen implementation b2dd40f; completed evidence is `runs/field-width-20261016/completion.json`. No training, checkpoint change, production adoption or fresh16 evaluation.

Next: teacher-compensated soft opening consistency on the finest field. Preserve the existing aligned geometry objective and all stage losses, and compare a matched continuation to the unchanged control. Check that taper/endpoint targets incur no extra penalty and gradients suppress narrow extraneous contours before training.
