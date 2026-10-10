# AN: conditional broad continuation of paper-band training

Prepare while AM runs; do not start updates unless AM final raw6000 passes all
22 original fit criteria and eight AD retention guards after actual image review.
Bind that completed final audit at launch. AM fitted weights are never used here.

One band arm restores AE raw4000 full model, AdamW and EMA; continue the original
indexed data/noise clock at step4000. Change only train.steps4000→16000 and set
learning_rate_decay_steps=4000 to preserve the original LR schedule exactly;
updates4000–15999 stay at the existing1e-5 floor. No reset/warm restart. Same
130605 model, all losses, data split/augmentation/sampling, batch4/sqrt(batch)LR,
recursive depths1–6, highest-stage loss and protected pixels. No short-gap,
opacity.6, margin or mass-normalized loss changes. All rough inputs are synthetic.

Before updates verify166 AE teacher hashes, crop resume across4000/4006 and all
six prepared-array depths. Test complete parent/next-update state and continuation
journal trimming. Freeze source/config/data/plan/test logs; bind parent full state.
Retain actual4000/4006/8000/16000 states and four new raw/EMA exports. Local64 and
broad456 conditions per model at4000/8000/16000; reused AE4000 outputs/metrics must
reproduce exactly. Data audits bind all12000 actual sampling records and journals.

Final raw16000 is fixed. Main1566 conditions for E, R, frozenAE4000 and extendedAE;
keep all229 existing E/R-based gates from AE unchanged. AE is the matched
before/after comparison, not an independent seed. Review eight final local and
six main native sheets, including partial black, blank, thick taper, thin lines,
close pairs and deliberate gaps. Do not select intermediate or EMA outputs.

Only an eligible final model proceeds to independent-seed training, sealed fixed
synthetic confirmation and actual BGRA/ONNX input-scale checks. This study uses
no sealed/final-source identities, changes no production/plugin defaults and is
not Windows/Krita GUI validation. If AM fails, retain this as unrun preparation.
