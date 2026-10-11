# BH: direct distance supervision in the one-pass restoration model

BG establishes exact coverage reconstruction, not learned quality. Test its
extended target on the established source-feature/source-anchor one-pass model,
where unbounded preclip values are not repeatedly fed back as physical flow
states. This is a separate diagnostic from BC–BF's velocity flow. It does not
solve multiple possible clean modes or prove transport sampling. The network
architecture, parameter count130605 and renderer remain unchanged.

Reuse exactly AD's R4000 initial parameters, fresh Adam/EMA and config: batch4,
baseLR5e-5 scaled by sqrt(4) to constant1e-4, warmup0,1200 updates, seed2026101066,
512px four fixed training pairs, fixed original noise generated after each resize,
cycle depths1–6, highest-stage objective with gradients through smaller stages.
Main loss remains AD's intervalL1/tone-power.5/deficit/gap/paper-band objective.
The frozen AD band1200 is the control, not a retrained or selected checkpoint.

Only add .1 times the direct field loss. Interpret physical preclip p/2 as d.
Build BG radius8 coverage-preserving teacher D on CPU. Compute interval absolute
error between p/16 and D/8: interior targets match exactly; cap+1 permits values
>=1 and cap-1 permits<=-1. Use existing teacher-tone weights(power.5), active
pixels only, normalized per image then average. No extra division by noise level.
This supplies a distance target beyond the antialias band, and preserves live
inward gradients at wrong saturation. It is not predicted-EDT backprop, a true
SDF/Eikonal constraint, a hard lineart prior, or an additional output head. The
weight and radius are fixed without a sweep.

Verify zero loss/gradient on exact target fields and valid capped saturation,
inward gradients and actual descent from a wrong field, zero protected gradients,
and actual CNN update/save/restore. At weight0, base loss and parameter gradients
must match the unchanged AD objective. Before training, replay all8 frozen AD
primary outputs bitwise. Keep states0/6/400/1200 and8 raw/EMA exports; first six
updates replay exactly across all depths. Final raw1200 only.

Evaluate final control/distance at depths1/3/6 on all8 cases, plus initial model
at depth3:56 rows and14 full prediction blocks. Add2 partial-mask blocks at depth6,
16 total. Same primary depth3 thresholds,22 fit checks per arm(44 total), and
16 F1/ink-deficit retention checks across all8 cases. Other depths diagnostic.
Source-only features/base make this model deterministic for full-mask input;
do not present repeated noise draws as independent trials. Original AD8 primary
outputs must replay exactly. Five sheets: train/probe native and overviews plus
partial native. All finite and protected prepared-float pixels exact. Partial
mask is an invariant check, not a trained mask-quality claim. No production,
plugin, defaults, sealed conditions or independent training seed consumed.

Preflight repair: the original frozen BH run stopped before any update because
JAX arrays reached the legacy host pyramid. This worktree converts to NumPy
before any resize and retains original batch4 grouping. Host-pyramid exactness
is tested at depths1/6; no scientific condition or acceptance threshold changes.
See the [original repair record](../../jax-distance-supervision-fit-20261011/docs/distance-supervision-preflight-repair-20261011.md).
