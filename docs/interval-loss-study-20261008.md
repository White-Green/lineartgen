# Primary loss before clipping

The hard-clipped losses cannot repair a wrong saturated output locally. Passing
all gradients through clipping restored some faint profiles but introduced
background marks, and the fixed parameter blends also failed the quality gates.
This study tests a true primary loss that supplies repair gradients without
changing clipping derivatives or recursive feedback.

Let z be the physical clean estimate before clipping (input + sqrt(noise) times
the residual head), and y the clean teacher in [-1,1]. The pixel error is:

```
y ==  1: min(z - y, 0)^2
y == -1: max(z - y, 0)^2
otherwise: (z - y)^2
```

This is squared distance to the set of pre-clip values that yield the teacher.
It equals MSE inside the valid output range. Correct white and black margins
have zero loss and gradient. A wrong saturated value has an inward gradient;
gray antialiasing targets remain continuous and exact. Each pixel function is
convex; the network optimization is not. This is a mathematical construction,
not evidence of better image quality. In particular, nearly-white gray targets
penalize large positive margins that conventional clipped MSE treats as small
errors, so gradient size and line/background side effects must be checked.

Two paired arms, `interval_control` and `interval_ink_guard`, replace only the
primary pixel error of the completed hard `control` and `ink_guard` arms. The
former retains contrast .01; the latter retains its existing region and ink
retention weights, with contrast disabled. All auxiliary losses still receive
hard-clipped predictions and their true derivatives. No new coefficient or
quality-driven normalization is introduced. The optional pre-clip values are
used only for supervised stages; every smaller-stage prediction fed into the
next stage remains clipped. Existing APIs and ONNX outputs remain unchanged.

Use the common largest 126,765-parameter 100k EMA, seed20261007, effective batch
16, sqrt(batch)-scaled peak LR4e-5 and the unchanged 10k cosine clock. Preserve
recursion1–6, stage multiplier0, data/sketch/geometry draws and all acceptance
thresholds. Reuse the completed paired hard controls rather than retrain them.
Before training, compare accumulated parameter-gradient norms on twelve fixed
training batches (seed20262006; two batches per recursion depth), without model
updates or coefficient tuning. Run actual GPU six-depth updates and full-state
checkpoint restore for both arms. With the new option disabled, reproduce all
six completed hard baseline updates and their optimizer/EMA states exactly.
Tests cover endpoint stationarity, inward finite-difference gradients, gray
teachers, protected residual pixels, clipped recursion, stage weights, both
trainers, serialized inference and ONNX validation.

Apply the inherited 2k screen and native-pixel visual review before continuing
at most two passing arms from full states to 10k. Final adoption still requires
both primary halo reductions >=10%, all line/background safety gates, visual
review, a second fine-tuning seed, and the previously fixed fresh synthetic
conditions sealed after candidate selection. No second seed or fresh cases
will be consumed for rejected candidates. This worktree keeps all completed
experiment sources and shipped Krita artifacts unchanged.

Settings are frozen in `runs/interval-loss-20261008-screen`. This document is a
plan; it does not claim the proposed loss has passed quality evaluation.
