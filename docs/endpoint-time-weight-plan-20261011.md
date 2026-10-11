# CF: reduce late-time amplification in bounded endpoint training

The bounded endpoint head does not shrink its residual with remaining time u,
while the current loss divides endpoint error by max(u,1/32) before squaring.
Thus the scalar weight reaches1024 at late times. Test whether this allocation
of finite network capacity/optimization effort hurts early restoration. This is
a hypothesis; neither a proof of the failure cause nor a published recipe.

Use the actual CA modulated6000 model/Adam/EMA state and exact CC step6000..7199
inputs, depth1..6 rotation, batch4, Gaussian draws,25%tau0/otherwise sin² times.
Two new arms continue1200 updates: `cap` uses max(u,1/4), `endpoint` uses1.
Keep raw interval endpoint error, active-pixel mean, noisy residual anchor,
Fourier FiLM,134688 parameters, and top-stage gradient through clipped lower
inserts. The matched `control` is CC ordinary7200, frozen before this comparison;
its floor stays1/32. No source-band weighting, new image loss or new noise.
Both new arms start at exactly the same parameters and optimizer/EMA states.
The positive time weights preserve unconstrained fixed-time conditional risk
minimizers, but finite model fitting, total gradient scale and Adam dynamics
change; this experiment does not isolate time allocation from update magnitude.
No global loss rescaling or tuned learning rate is introduced. Keep constant
LR5e-6*sqrt4=1e-5, no warmup, as CC. This is a temporary fitting diagnostic, not
a change to the production multi-stage loss contract.

Model definition and inference do not change: uniform32 steps, bounded endpoint
velocity divided by max(u,1/32). The new floors apply only to training risk.
Save and restore full states6000/6006/6600/7200,16 raw/EMA exports; replay all six
initial updates and compare all1200 input journal hashes to CC. Test loss
weights analytically, nonzero finite gradients, protected-pixel exclusion,
unchanged default objective/gradients, and state restore/next update.

Final raw7200 only. Evaluate control/cap/endpoint at512px, the same four TRAIN
and four PROBE pairs, depth1/3/6, four Gaussian draws:288 rows and75 blocks with
partial depth6. Replay all25 CC control physical/clipped blocks and96 metric
rows bitwise. Primary depth3:88 fit checks/method,64 F1/deficit retention checks
for cap/control,endpoint/control,endpoint/cap, unchanged limits. Review all
five fixed native/overview/partial sheets. Four draws appear on whole sheets;
partial has one draw per train case. No checkpoint selection, new shape,
independent seed, sealed confirmation, plugin/default/export/deployment change.
