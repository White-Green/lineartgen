# BA: Boundary-focused inference time grid

Declared after AZ. Interval velocity passes all shape criteria at32 uniform
Euler steps but misses one mode-balance condition (58/64 A, limit<=.9). Do not
relax that condition. Test a grid matching the training-time emphasis:
tau_k=sin²(pi*k/(2*N)), k=0..N; x_next=x+(tau_next-tau)*v(x,remaining=1-tau).
Use the exact same fixed MSE/interval36,000-step weights, inputs and noise.
There are N field evaluations per recursive scale, as before. No new training,
loss, clipping, teacher access or target assignment during inference.

Replay all72 uniform-solver blocks for the two AZ arms at1/8/32 steps and
require bitwise physical/clipped output identity. Add72 boundary-grid blocks
to the324 cached controls (396 total). Apply the unchanged36 criteria per arm
to grid32: mean nearest-endpoint RMS<=.15, fraction within.20>=.8 and A fraction
.1–.9 for every one of12 groups. Interval-grid is the hypothesis of interest;
MSE-grid remains a matched fixed-weight comparison. No choice among checkpoints,
step counts or samples. Keep all failures and finite/protected checks.

Tests cover monotone nodes, endpoints, constant-field integration, protected
pixels, teacher-free predictor arguments, one-step equivalence to uniform
Euler, and invalid step counts. Inspect3 fixed native sheets showing the same
first4 draws of each grid arm and each uniform baseline. This remains a two-shape
mechanism test. Even a pass does not establish broad synthetic-lineart quality,
independent-seed confirmation, or actualBGRA/ONNX compatibility. No adoption.
