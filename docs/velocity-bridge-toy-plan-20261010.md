# AV: direct velocity on the same ambiguous toy

Declared after AU's256-step comparison. The endpoint model divides residual
prediction error by remaining time. If the error does not decay appropriately,
this conversion amplifies it near the endpoint; finer integration cannot enforce
the required decay. Audit that residual on AT's known paths before AV training.
This mechanism is a hypothesis about AT's artifacts, not a proven explanation
of gray output in the original restoration model.

Train the same130,628-parameter CNN to predict velocity directly. Interpret its
existing physical residual `raw-image` as v. At each recursion stage use
`clip(image + remaining_time*v)` as the predicted clean insert for the next
larger stage. Return the un-clipped velocity for supervision and integration.
Protected velocities are zero and endpoints/states equal the rough source.
Train MSE(v, teacher-z), averaged per image over active pixels. Inference Euler
updates x += v/N, without division by remaining time. This jointly changes the
output interpretation, consistent recursive inserts and objective; it is one
velocity parameterization comparison, not a loss-only ablation.

Everything else matches AT: common seed42 initial parameters, fresh AdamW/EMA,
6,000 updates, batch16, baseLR5e-5 times sqrt(16)=2e-4, warmup100/cosine to1e-5,
seed2026101088, same two64px templates, labels, masks, strengths, times and
Gaussian draws. Reuse the exact batch generation and verify equality across all
three depths and multiple indexed updates. No AT fitted weights initialize AV.
Initial time0 endpoints must match AT's initial model within1e-5, acknowledging
roundoff from subtraction/addition. Depths1–3 and top-stage loss with recursive
gradients are toy constraints, not changes to production512px depth1–6.

Retain full states6/2000/6000 and six raw/EMA exports. Final raw6000 is fixed.
Evaluate the same12 groups,64 draws/group, at1/8/32 steps, with AT's exact cached
direct and endpoint-trained controls (hash-bound; no retraining). Apply the same
36 toy checks to velocity32: every group mean normalized nearest-endpoint RMS
<=.15, >=80% of draws within.20, and10–90% assigned to each mode. Finite outputs
and protected exactness mandatory. Keep every failure and all108 output blocks,
including72 reused controls. Review three fixed sheets: A/B/rough/direct1/AT32,
velocity1, and first four velocity32 draws. No sample/EMA/checkpoint selection.

Tests must verify target velocity, time0 agreement, endpoint recursion and
gradients at smaller stages, teacher-free integration, finite bounded-field
integration through time1, masks, actual updates/full-state/next-update restore,
and exact unchanged batch sampling. No new coefficient is tuned from outputs.
An optional endpoint-error audit uses only AT weights and known trajectories;
it cannot validate AV in advance. A toy pass still needs a separate broad-data
experiment; no toy weights or custom sampler are deployable. No adoption,
ordinary-sketch claim, sealed confirmation, independent training seed or plugin
change is made here.
