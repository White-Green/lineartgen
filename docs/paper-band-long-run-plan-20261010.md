# AP: broader AE training-duration study, declared before AO results

AM's fixed-pair curve improves with more updates; its unfitted taper remains gray.
These are different questions. AE was trained only4000 updates, and T's16000
study had different augmentation/losses. Test broader AE duration directly rather
than infer generalization from memorizing four examples. This plan is declared
before any AO evaluation beyond its initial state and does not select its result.

AN remains unrun because its AM pass prerequisite failed. AP explicitly has no
fixed-pair pass prerequisite: it tests the already-existing AE loss, not a newly
selected AO loss/model. Waiting for AO's reviewed completion is only sequencing
of GPU work. No fitted weights, AO learning rates or intermediate selections enter
AP. This changes which diagnostic we run, not any model adoption criterion.

One band arm restores AE4000 complete model, AdamW moments/count and EMA. Extend
to16000, changing only train.steps and learning_rate_decay_steps=4000 to retain
the original LR curve and its1e-5 floor. No reset or warm restart. Keep130605
parameters, all losses/data/split/corruptions, the indexed crop/noise clock,
batch4/sqrt(batch)LR, recursive depths1–6, highest-stage loss with recursive
gradients, downsample-before-Gaussian and protected prepared-float pixels.

Verify166 AE teacher hashes, crop resume4000/4006 and all six prepared-array
depths. Test actual parent/next-update restoration, journal trimming, launch
completion binding and exact reproduction of AE's229 quality checks. Freeze
source/config/data/plan/test logs before updates. Retain4000/4006/8000/16000 full
states and four new raw/EMA exports; compare exact reused AE local64/broad456
outputs at4000/8000/16000. Bind all12000 actual sampling records and journals.

The fixed final raw16000 output gets the same1566 main conditions for E, R,
AE4000 and continuedAE. Keep all229 E/R-based quality gates exactly unchanged;
AE4000 is the matched duration comparison, not an independent seed. Inspect eight
local and six main final sheets. No intermediate/EMA choice. Even a pass is only
a candidate: independent-seed training, sealed synthetic confirmation and actual
BGRA/ONNX input-scale checks precede adoption. No final-source/sealed identities,
production/plugin changes or Windows/Krita GUI claims in this study.
