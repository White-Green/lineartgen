# BI: distance-dependent bounds that retain valid white/black saturation

BH improved unseen junction/taper, but introduced a parallel-line bridge and
worsened the trained curve. Its normalized distance targets are point values
except at the truncation cap; even a correctly white near-boundary preclip value
can be pulled back to the clipping threshold. Test that restriction directly.
This is a hypothesis about the observed regression, not a causal conclusion.

Repeat BH's exact1200-update initialization/config/data/host NumPy preparation,
source-only one-pass model130605 parameters, stage cycle1–6, top-stage loss with
recursive gradients, fixed source-after-resize noise and constant1e-4 LR after
sqrt(batch4) scaling. Fresh Adam/EMA, same R4000 initial state. Original main
AD image objective unchanged. Radius8 and coefficient.1 unchanged. Compare to
frozen BH finalraw1200; no model size, new shapes or other loss terms changed.

Only change the field residual e=p/16-D/8: if the clean teacher is saturated
white, use min(e,0); if saturated black, use max(e,0); otherwise keep e. Take
absolute error with zero gradient at exact zero, tone-power.5 weights and active
per-image normalization as BH. Thus every saturated teacher permits a correct
one-sided distance bound, including near-boundary pixels. Antialias coverage
still has an exact scalar target. This is a spatial margin, not exact SDF
regression, Eikonal regularization, topology guarantee or conditional-flow proof.

Retain0/6/400/1200 full states and8 raw/EMA exports, replay first6 actual updates.
Before training and at evaluation reproduce all8 frozen BH primary outputs
bitwise, preserving original4-image grouping and host NumPy pyramids. Test both
near-boundary valid saturation and wrong saturation gradients, exact fractional
targets, protected pixels, actual CNN update and full restore. Final raw1200 only.

Same56 rows,16 prediction blocks and5 sheets as BH: final2 models x8 cases x
3 depths, plus8 initial depth3 references; partial-mask invariant checks at depth6.
Primary22 fitting criteria each(44),16 F1/deficit retention criteria relative BH
across all8 cases. Other depths remain diagnostic. Inspect train/probe native and
overview plus partial native. Do not select a checkpoint, alter thresholds or
call deterministic noise repetitions independent trials. No production/plugin/
default changes. A successful fixed fit would still require broad synthetic
training and independent/sealed/BGRA validation before adoption.
