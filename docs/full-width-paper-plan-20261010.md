# AR: full-width UNet on the AE broad training protocol

Declared before AR training or candidate evaluation, 2026-10-10 UTC.

AP's completed 16,000-update continuation reduced halo but failed 56 of 229
unchanged development checks. Its 14 final images retain gray fine strokes and
merged lines. AO showed that the small network can fit four pairs after settling;
that does not establish enough capacity for the broader distribution. Q enlarged
encoder/detail widths on older loss/data, and AJ tested all widths only on four
fixed pairs. AR tests the remaining broader capacity question.

## One architecture comparison

Use AE's 130,605-parameter raw checkpoints at 0/1,000/4,000 as the small control.
Train a new 517,865-parameter model for exactly 4,000 updates. Only these four
model fields change from AE: channels (4,10,20,40) to (8,20,40,80), detail 16 to
32, final decoder 2 to 4, and head hidden default 3 to explicit 6. Both use E
raw4k as their starting function, fresh AdamW/EMA, and the identical loss/data.
Embed inherited parameters with widening seed 2026101087; inherited channels
initially ignore new channels. Convolution summation order can change, so initial
agreement is absolute tolerance 1e-5, not bitwise equality. Check 48 whole/partial
inputs across all six recursion depths on GPU before any updates. New paths must
receive gradients in the real training objective, tested with full-state restore.

Training seed 2026100961, batch 4, base LR 5e-5 times sqrt(batch)=1e-4, 100-step
warmup, cosine to 1e-5 at 4,000. No continuation of AP or AO weights. Preserve
AE's indexed crops, teacher mixture, corruption, Gaussian draws, recursion 1–6,
scale-loss multiplier 0 with recursive gradients, protected targets, interval L1,
opacity deficit .3, gap clearance .1, and paper-band .1. Audit all 166 historical
teacher hashes, first 12 targets, resume 6–11, and prepared arrays at all six depths.
Retain full states at 6/1,000/4,000 and verify all four raw/EMA milestone exports.
Audit all 4,000 sampling records against AE after training.

## Fixed assessment

Local control is `band` (AE), candidate `wide`: 64 primary and 456 broad conditions
per method at 1,000 and 4,000, plus initial primary agreement. Main development
uses initial E, reference R (`pixel`), AE (`band`), and AR (`wide`), 1,566 each,
6,264 total. Only final raw4k decides; intermediate/EMA weights are diagnostic.
Reproduce all existing 229 E/R adoption checks exactly under the candidate rename.
This keeps earlier thresholds and references rather than selecting an easier
benchmark. The AE comparison separately isolates capacity.

Primary geometry blur .5 F1 >= .90 and cuts <= .02; corpus blur/strokes .5 halo
<= .02/.10. Across clean/tone/blur/strokes .1/.5 whole/partial, halo <= reference
times 1.1 + .0005, F1 drop <= .005, deficit increase <= .01, relative to both E/R.
Whole geometry additionally allows width MAE increase <= .10 and cut/bridge
increase <= .01. Edited black interiors mean >= .95, blank max <= .05. Finite
outputs and prepared-float protected pixels exact. Tests replay AE's real summary
through AR's evaluator to establish identical gates. Review all six final main
sheets and eight local sheets, binding images to predictions and source.

AP completion is context and resource sequencing, not a fitting-pass launch gate.
No old failed launch gate or study is edited. Only synthetic corruptions and
existing development identities are used; no natural rough/clean pairs, sealed
confirmation, or final-source images. A pass would require an independent matched
training seed and unseen confirmation before adoption, then actual BGRA/ONNX checks.
Krita runtime, distributed weights, production defaults, and remote branches stay
outside this experiment. No smallest-model or real-sketch-quality claim follows
from this single comparison.
