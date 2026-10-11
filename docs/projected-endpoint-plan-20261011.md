# BD: bounded predicted endpoints during velocity integration

Predeclared after BC failed on512px antialiased geometry. Its interval objective
allows correctly saturated endpoints to exceed[-1,1], while the sampler feeds
unbounded images into the following evaluation. Recursive smaller-stage inserts
are already clipped. Test this mismatch before changing the loss or retraining.
This is a hypothesis about sampling stability, not a claim of exact transport.

Use BC finalraw6000 MSE and interval weights unchanged. No updates. The original
32-step sin² Euler grid and Gaussian initialization after each resize are exact.
At every stage and integration step, let u=1-tau, y=x+u*v. Replace v by
v+(clip(y,-1,1)-y)/u, then use x_next=x+delta_tau*v_bounded. All u are strictly
positive on the fixed grid; no division at the endpoint. This projects predicted
endpoints, not the initial Gaussian or each intermediate input. Protected pixels
are restored from the original guide at every step. No teacher enters sampling.

Reproduce all52 BC physical/clipped blocks bitwise with the original sampler,
including initial-weight references and partial masks. Cache its200 rows, add192
projected rows (2 models x8 cases x3 depths x4 draws). Keep102 prediction blocks:
52 originals,48 new full-mask and2 new partial. Primary depth3 keeps the same
88 fit checks per arm:352 across the four original/projected arms. Compare each
projected arm against its own original with64 F1/ink-deficit retention checks,
128 total. Other depths remain diagnostic, not an override. Do not choose a
checkpoint, draw, recursion depth or threshold based on results.

All512px sources, target antialias coverage, strength.5, EVAL_SEED2026102093,
TRAIN999/1095/1287/1383, PROBE807/903/1191/1479, and crop coordinates are exactlyBC.
Record32 root trajectory summaries for both train/probe depth3 draw0 under both
samplers and both models (eight traces). Inspect seven fixed sheets: interval
train/probe native and overview, interval partial native, MSE train/probe native.
Four primary draws appear in every full-mask sheet. Require finite values and
exact prepared-float protected pixels. Partial masks remain invariant diagnostics.

If projection fails the fixed fitting criteria it is not a candidate. Do not
consume independent training seed, sealed broad conditions or final identities.
If it succeeds, confirm the learned two-mode toy before broader training. No
production/plugin/runtime/default changes or claim of rawBGRA preservation.
