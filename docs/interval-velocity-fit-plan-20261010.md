# AZ: Interval velocity loss from the balanced30,000-step model

Declared after AY. Boundary-time training at30,000 preserves both modes in
all12 groups, but background speckle and some gray gaps remain. Split its actual
model/Adam/count/EMA state into two identical arms for6,000 additional updates.
One keeps ordinary active-pixel velocity MSE; the other changes only its error:

- target paper y=+1: error=min(v_pred-v_target,0);
- target opaque ink y=-1: error=max(v_pred-v_target,0);
- intermediate coverage: error=v_pred-v_target.

Square the residual, average over active pixels per image, then over the batch,
exactly as before. No coefficient or margin. On training paths x=u*z+(1-u)*y,
endpoint=x+u*v, this residual is the physical endpoint's signed interval error
divided by u for u>0. Implementation never divides by u and remains finite at
u=0. Correctly saturated predictions get zero error, incorrect saturation must
retain gradient, and intermediate coverage retains the ordinary derivative.
Protected pixels are excluded. This changes the statistical target of flow
matching at saturated endpoints; no exact-distribution or optimal-flow claim.

Both use the same frozen AX boundary-time conditional-OT minibatch function at
indices30000..35999, equal order-independent data, seeds, masks and noise after
resizing. Same130,628-parameter CNN,64px depths1–3, top-stage loss with recursive
gradients, sqrt(batch) LR and original schedule at its1e-5 floor. Configs record
36,000 total steps with6,000 decay steps. No optimizer reset or new data.

Tests cover gradient direction/zeroes, interval/endpoint algebra, protected loss,
unchanged MSE update, independent copies, schedule and full-state/next-update
restore. Require full equality to the parent at30,000 and exact replay of the
first6 additional updates. Retain full states30000/30006/32000/36000 and16
raw/EMA exports. Evaluate final raw36,000 only for each arm; no checkpoint/EMA
selection. Reuse252 hash-bound controls, keep324 output blocks in total.

Evaluate the same12groups×64draws×1/8/32steps. Apply unchanged36 checks per arm
at32steps (RMS<=.15, fraction within.20>=.8, mode-A .1–.9), report all72 checks
and require finite physical values/protected exactness. Inspect3 fixed native
sheets with6,000/30,000 boundary baselines and first4 final draws per arm.
Even a passing toy still needs broader synthetic shapes, independent seed and
untouched conditions, then actualBGRA/ONNX quality checks. No plugin changes.
