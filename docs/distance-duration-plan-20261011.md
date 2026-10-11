# BJ: matched duration and settling for coverage-distance objectives

BH point-distance and BI saturated-distance bounds have complementary failures
after1200 fixed-pair updates. Test whether more fitting removes these before
changing either loss. Continue both actual1200 model/Adam/EMA states,4800 updates
at constant1e-4 to6000, then1200 updates at constant1e-5 to7200. Base rates5e-5
and5e-6 retain sqrt(batch4) scaling. No Adam/EMA reset at either transition.
This matches AD->AM->AO's7200 fixed-pair exposure and LR schedule. Frozen AO
settle7200 is the main control; do not retrain or select intermediate checkpoints.

Keep model130605, source-feature/source-anchor one-pass architecture, fixed4
train and4 unfitted diagnostic probe pairs, seed2026101066, host NumPy pyramids,
noise after resizing, cyclic depths1–6, highest-stage image loss and gradients
through all smaller stages. The original AD loss remains unchanged. Added field
weight.1/radius8 and the respective BH/BI residuals remain unchanged. No new data,
model capacity, OT assignment, sampling rule or loss coefficient changes.

Keep full states1200/1206/6000/6006/7200 and20 raw/EMA exports. Replay the first6
continuation steps against original parent routines; replay the first6 low-rate
steps from saved6000. Verify all numerical states, optimizer counts, phase-switch
state preservation, exports and initial outputs. Before training and again at
evaluation, replay all8 AO primary outputs bitwise in original groups of4. Tests
cover nonzero optimizer/EMA transfer, phase rates, actual next updates and restore.
AO's frozen config contains five historical training-only fields absent from this
branch. An explicit adapter checks their expected values, removes only those
fields from a local copy, and verifies identical model config and weight bytes.
The original files are hash-bound and never changed; exact output replay is still
mandatory. These fields have no role in inference.

Evaluate finalraw7200 only:3 models x8 cases x depths1/3/6 =72 rows,18 full
prediction blocks plus3 depth6 partial-mask blocks. Inspect5 sheets:train/probe
native and overview, partial native. Prepared-float protection must be exact;
this is not yet a raw BGRA runtime test. Depth3 has22 fit checks/model (66total).
Each new arm also has16 retention checks versus AO across all8 cases: F1 decline
at most.005, deficit increase at most.01 (32total). Other depths are diagnostic.
No threshold changes or intermediate checkpoint selection. Do not treat repeated
noise for this deterministic source model as independent trials.

A fixed fitting pass is a prerequisite, not production eligibility. Broad
synthetic training, independent/sealed confirmation and actual runtime parity
remain required. No production/plugin/default changes or adoption in this study.
