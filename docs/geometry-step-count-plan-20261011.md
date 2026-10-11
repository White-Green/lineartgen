# CD: inspect whether the learned geometry endpoint fails before or during iteration

CC's final control/band7200 endpoints still miss curves, thin lines and junctions.
Before further training, evaluate both fixed models at1/8/32/128 uniform Euler
steps. No new weights, epochs, input corruption, teacher guidance or noise draws.
This diagnoses inference behavior; it cannot attribute every error to integration
or establish broader quality. No choice of a checkpoint from intermediate states.

Use BW's endpoint sampler velocity (bounded_D-x)/remaining_time with machine
epsilon only outside the prescribed domain. This equals the current denominator
max(u,1/32) at all32-step nodes (and at1/8 nodes), so both32-step references must
replay bitwise. At128 steps, the last three nodes use actual remaining u below
1/32, rather than the training loss floor. This is explicitly a sampling-time
extension of the learned endpoint field, not an unchanged late capped velocity.
All schemes end at the bounded endpoint on their final Euler step up to float
rounding. Noise is
still added independently after each source resize, amplitude .25*t. Fixed noise
keys are shared across step counts. Protected pixels stay exact at every step.

Freeze source/model/config/data/plan hashes before evaluation. Both134688-parameter
models are final raw7200 from CC, copied and bound to their original files.
No optimizer, EMA selection or further neural updates. CPU tests check the
actual32-step model replay, protected pixels, prescribed counts, final bounded
oracle endpoints and finite late-time denominators.

Evaluate all2 models x4 step counts x3 depths(1/3/6) x4 draws x8 cases:768 rows,
plus partial depth6 batch per method,200 physical/clipped blocks. Replay both
original32-step references:50 blocks and192 metric rows. At primary depth3 keep
88 fitting checks per method,704 total. For each model compare1/8/128 against
its own32-step F1/deficit retention:64 checks each,384 total. Keep all criteria;
no posthoc relaxation and no pooled failure count treated as a quality score.

Render and inspect five fixed sheets per model: train/probe native and overview
with all four draws at each step count, partial native with one fixed draw per
train case. Ten sheets total, all original AD crops. Store every evaluated output.
The1-step result is the model's actual initial denoised endpoint, not a teacher
or a separately trained one-pass model.128 steps cost4x32 network evaluations.

No new shapes, independent training or evaluation seed, sealed confirmation,
production model, plugin, runtime defaults, export or deployment.
