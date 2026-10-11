# BE: paired continuation with more observed-source training

BC's512px interval model still loses curves and thin-line geometry; endpoint
projection in BD only partly helped and harmed unfitted thin lines. Keep the
original sampler. Test whether the model needs more learning directly from the
noisy rough source, rather than intermediate paths that already contain clean
teacher information. This is a hypothesis, not an established cause.

Both arms inherit the actual BC interval6000 model, Adam moments/counts and EMA.
Continue6000 additional updates to12000. Set a common constant1e-5 learning rate
(base5e-6 times sqrt(batch4)); no optimizer reset. This common settling change
means comparison against BC alone confounds duration and LR. The paired control
isolates the changed time atom. Preserve all512px data, batch4, noise strength.5,
Gaussian after each resize, seed2026101093 and stage cycle1–6. Highest-stage loss
with gradients through smaller clipped endpoint inserts, unchanged interval
velocity objective,130628 parameters and all architecture settings.

Control imports BC's prepared_batch unchanged:25%tau0, otherwise sin²(pi*U/2).
Focused uses the exact same draws and changes only the atom probability to75%.
Select the already-computed direct batch when draw0<.75; otherwise use the
original bridge. This preserves images bitwise where the path is unchanged and
preserves direct Gaussian/source/teacher/levels everywhere. No target shuffling.
The continuous-time density conditional on being nonzero is unchanged; this
changes training-time weighting with no importance correction. Record actual
zero-time exposures over all24000 examples per arm. The75% probability is fixed,
not a parameter sweep. No model selection using probe images.

Retain full states6000/6006/8000/12000 and16 raw/EMA exports. Assert initial full
state equality against actual BC, schedule1e-5, and first six actual updates
replayed exactly. Use final raw12000 only. Evaluate original32-step sin²-grid
sampling at depths1/3/6 with four fixed Gaussian draws and all8 BC cases.
Keep200 metric rows (192 final plus8 initial depth3/draw0) and52 physical/clipped
blocks including both final partial-mask checks. Primary depth3 retains88 fit
checks per arm and64 focused-vs-control F1/ink-deficit retention checks. All
thresholds/crops/seed identical BC. Five sheets: train/probe native and overviews
with all four draws per arm, plus native partial mask. Require finite outputs and
exact protected prepared-float pixels. Partial masks are not a trained quality
claim. Broad data, independent training seed, sealed conditions, actualBGRA/ONNX
quality remain separate. No production/plugin/default changes.
