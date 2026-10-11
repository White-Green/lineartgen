# BP: lower initial Gaussian amplitude, same source observation

BO's wider velocity model improved some fitted strokes but did not pass. Its
posthoc source/noise audit motivates testing the arbitrary initial coefficient.
The source is also available noiselessly as a guide; this does not assume that
information was irreversibly destroyed or that noise is the sole failure cause.

Return to BC130628 model, original AZ interval36000 initialization and point
velocity MSE, fresh Adam/EMA. Only change initial z=s+.25*t*epsilon to
z=s+.05*t*epsilon, in BOTH training and inference at every resolution. Keep the
same standardized Gaussian draws after resizing, guides, strength maps, targets,
masking, seed2026101093,25%tau0 otherwise sin²(pi*U/2), recursive depth1–6,
highest-stage velocity supervision and recursive gradients,32-step sin² Euler.
No input created from a resized prediction; original rough input remains s.
No clipping of the Gaussian. No data/architecture/extra loss or OT change.

6000updates, batch4, baseLR5e-5*sqrt4=1e-4 constant. Same4 TRAIN/4 unfitted PROBE
512px antialiased pairs, strength.5. Initial sigma.125 becomes.025. Frozen BC
MSE6000 at coefficient.25 is the control. The candidate and its initial model
are evaluated at.05. This changes initial distribution AND its required target
velocities; it is not solely an inference adjustment. No claim about preserving
mode coverage on ambiguous inputs follows from these fixed pairs.

Before updates, the new preparation with coefficient.25 must reproduce all
arrays and tau bitwise on steps0–5/depths1–6. At.05, all guides, levels, teachers,
and tau remain bitwise equal; initial perturbations equal0.2 times the old ones
within3e-7 absolute float error. The new sampler at.25 must reproduce8 BC final
and8 initial primary outputs bitwise, then all25 final control blocks on final
evaluation. Tests include independent noise after resizing, path derivative,
sampler agreement at.25, same-node exact oracle with both amplitudes, zero-strength
protection, and actual update/full Adam+EMA restoration.

States0/6/2000/6000,8 raw/EMA exports, first6 exact replay and all state/export
checks. Primary finalraw6000. Depth1/3/6,4 fixed fresh Gaussian draws,8 cases plus
initialdepth3/draw0:200 rows,52 prediction blocks,88 fitting criteria per model,
64 retention checks againstBC. Review all5 native/overview/partial sheets. No
checkpoint/draw/EMA selection, criteria relaxation, production/plugin/default
change or adoption. Broader synthetic data, independent/sealed confirmation and
actual runtime parity remain necessary before deployment.
