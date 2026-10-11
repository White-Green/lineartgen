# CH: supervise line appearance on the actual two-step rollout

CG improved the fitted curve but retained gray/missing lines. Continue its actual
rollout7200 model/Adam/EMA to8400 in two arms: unchanged `control`, and `image`
with coefficient.1 times the existing AD image loss on the second raw endpoint.
No teacher image enters the rollout states. Keep both CG interval terms and
backpropagate the added image term through the first predicted midpoint.

Reuse the full AD loss configuration, not a new coefficient search: interval L1,
tone power.5, preclip opacity deficit.3, gap clearance.1, paper band.1, identity.1,
edge.1, active contrast.01, pixel strength floor.05. Compute image_loss separately
per image and average. The scalar coefficient.1 applies at every training sample
on the second endpoint; it is not BQ's25%-frequency start-only auxiliary, and
not a claim that any single component is responsible. It changes relative
priorities and gradient scale. Antialias targets remain fractional; the inherited
contrast term is a weak binary preference, so gray-target fidelity still needs
measurement. No black/white threshold is applied to the target.

Same134688-parameter model, batch4, constant LR5e-6*sqrt4=1e-5, no warmup,
1200 updates with original step-indexed source/target/Gaussians7200..8399 and
depth1..6 rotation. Old random time draws are unused: both passes remain at
u=1/.5. All smaller predictions feed only larger ones. Top-stage fitting loss
is an isolated diagnostic, not a production multi-stage loss change. Exactly
copy full parent states into both arms, replay six updates, restore all states
7200/7206/7800/8400 and16 raw/EMA exports. Test coefficient0 exact objective and
gradient agreement, finite nonzero added gradients including through midpoint,
protected outputs and actual state restore/next update. Bind AD config plus
both parent evaluation references before learning.

Final raw8400 only, frozen initialCGrollout7200; primary2 steps, secondary32,
same eight AD512px pairs, depth1/3/6 and four fixed Gaussian draws. Each count
has288 rows,75 blocks,264 fitting checks and192 retention checks for
control/initial,image/initial,image/control. Reproduce all50 CG physical/clipped
blocks and192 metric rows (25/96 per count) bitwise. Keep all existing thresholds.
Review all10 sheets: train/probe native and overview per count with four draws;
partial native per count with one draw per four training cases. All finite and
protected pixels exact. No checkpoint/seed/metric selection after results.

Only synthetic rough inputs. No new shapes, independent seed, sealed
confirmation, production model, plugin, runtime default, export or deployment.
Fixed-pair improvements do not establish broad quality or multimodal transport.
