# BS: point versus interval velocity on diverse procedural teachers

Fixed-four-pair velocity diagnostics BC and BM–BR did not produce a candidate.
BL's diverse procedural one-pass model also failed its existing broad guards.
This diagnostic tests the existing velocity recipe with diverse synthetic
exposure. It does not advance any failed weights or relax their criteria.

Two arms, ordinary point velocity MSE and interval velocity MSE, start from the
same AZ interval36000 weights stored in BC/initial. Both have130628 parameters,
channels4/10/20/40, detail16, noisy features/base, raw-source guidance and
remaining-time conditioning. Fresh Adam/EMA,4000 updates,batch4,baseLR5e-5
scaled by sqrt4 to1e-4,warm100,cosine decay through4000 to1e-5. Explicit decay
length4000 is equivalent to BL's null decay length falling back to steps4000.
No distance head, region balance, multiscale loss, endpoint loss or noise
reduction. Highest-stage loss differentiates through all smaller stage inserts;
depth cycles1–6,512px highest. This is the BC recipe, not a change to production
or a decision about its per-stage loss design.

Use BL's exact source config and teacher/corruption stream, seed2026101111:
16000 fresh procedural teachers, geometry probability.9, conditional coverage1;
nine geometry kinds, widths.6–6,thin.6–1.8,1–4 groups,5% blank,4x antialiasing,
contrast1.1. Coverage includes paper, black, ellipse, polygon and wide strokes.
Mixed tone/blur/strokes corruption probability.8, strong probability.5; ordinary
opacity.25–1,sigma0–2,displacement2; strong opacity.2–.45,sigma1.5–2.5,
displacement3; dilation radius2/blend1 and3 stroke composites. Keep BL's exact
strength and mask sampling. The old Gaussian input is discarded after source
preparation; no native corpus image is used in training.

Feed those prepared full-resolution source/strength/masked targets into BC's
unchanged path preparation (time/noise seed2026101093). Source and strength are
resized before independent Gaussian generation per scale: z=s+.25*t*epsilon,
x_tau=(1-tau)*z+tau*y, target velocity y-z. Time is zero with probability.25,
otherwise sin²(pi*U/2). All protected pixels preserve the source. There is no
minibatch transport assignment. Both arms receive identical arrays, including
the Gaussian draws and times. Point versus interval is the controlled contrast.
Compared with fixed BC, data and schedule differ; compared with E/R, architecture,
objective, initialization and sampler differ. Do not attribute all broad changes
to one component or claim this establishes a minimum model size.

Before updates, reproduce16 E/R fixed predictions bitwise and8 AZ initial BC
predictions bitwise. Compare six prepared source/strength/target batches against
BL. Record every teacher hash/depth/count; independently regenerate171 batches
and require the full4000-row teacher journal to equal BL. Save full model/Adam/EMA
at0/6/1000/4000 and16 raw/EMA exports; replay first6 updates from scratch; verify
all restores and export parameters. Final raw4000 only, no checkpoint selection.

Evaluate the same1566 reused development conditions for E, R, point and interval
(6264 rows). E/R keep their exact one-pass sampler and NumPy pyramid. Velocity
arms use32 sin²-spaced Euler steps at depth3 and PRNGKey(case seed), same noise
for both arms. Check finite physical outputs and clipped float protection. Replay
all3132 E/R rows and selected images bitwise. These corpus clean images receive
synthetic corruptions; they are not real rough/clean pairs or unseen confirmation.

Use unchanged229 E/R guards per trained arm: geometry blur.5 F1>=.90 and cuts<=.02;
corpus halo<=.02 for blur and<=.10 for strokes; F1 drop<=.005, deficit increase<=.01,
halo<=ref*1.1+.0005 across matched groups; geometry width increase<=.10,
cuts/bridges increase<=.01; edited black interior mean alpha>=.95 and blank
maximum<=.05. Fail counts are not a quality score. Review all six native sheets
(blur.5,strokes.1/.5,corpus,partial,worst blank), retaining input and teacher.
Display conversion alone is not actual BGRA API or Krita GUI verification.

No sealed evaluation or independent training seed yet. No production model,
Krita plugin, runtime defaults or deployment changes. All guards and visual
review are required before considering further independent confirmation.
