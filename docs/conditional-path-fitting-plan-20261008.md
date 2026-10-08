# Conditional intermediate-state fitting probe

Declared before implementation tests, candidate updates or candidate inference.
The ongoing broad detail-head study stays in its separate frozen checkout.
This probe follows the algebraic diagnostic; it does not assume that the training
path mismatch causes the learned gray fringes.

Train two independent width16 detail models, `progress_rough` and
`progress_bridge`, both with original raw-source guidance and explicit remaining
progress. Add zero-initialized, bias-free1×1 progress projections to the first
encoder, inherited output head and detail head, after all existing parameters
have been initialized. The added feature is `1-u`, with the same constant progress
per example at every resolution. No existing parameter shape or initial value
changes. Both restart the original100k EMA plus the same zero-output detail head;
neither reuses trained local or broad detail weights.

Use the exact57 training and57 already-observed local probe pairs from the detail
fitting study. Keep its batch4, seed2026100801, indexed batches, independent
Gaussian draws, initial strength/mask draws, depth1–6 cycle, stage weighting0,
sqrt(batch) scaling, baseLR1e-4, warmup50, cosine1200 clock, EMA.999 and
identity.1/edge.1/contrast.01 objective. Exactly1,200 updates per arm, no extension.
No natural rough/clean pairs or sealed confirmation draws are used.

Independently draw remaining progress per example: probability.5 gives `u=1`;
otherwise uniform on[.05,1]. This covers all forward times used by the runtime's
supported1–20 iterations while retaining explicit starting-state supervision.
Draw progress with a separately folded key, leaving the original data, mask and
Gaussian random streams unchanged. Form source, masked-target and initial-mask
pyramids before adding independent noise at each scale. Effective strength is
`t=s*u` in both arms. With `r,c,s,e` denoting those level's rough source, target,
initial mask and Gaussian noise:

```
progress_rough:  x = sqrt(1-s*u) r + sqrt(s*u) e
progress_bridge: x = sqrt(1-s*u) c
                    + sqrt(u)*sqrt(1-s) (r-c) + sqrt(s*u) e
```

The target remains `c` in both. Preserve protected input values exactly and
make the `u=1` branch use the existing initialization expression exactly. The
existing finite sampler update and Krita BGRA API stay unchanged; only an
internal progress input is added for these explicitly configured models. The
rough guide is always the original source. The teacher is never passed as a
guide or model argument during inference. Default legacy models keep their
existing input contract and numerical path.

Before training require the complete tests, exact legacy inputs and initialization,
analytic bridge/oracle tests with partial masks, gradients through new projections,
serialization and JAX/ONNX/BGRA agreement with nonzero learned-like projections.
Run six actual GPU updates at all recursion depths for each arm with exact full
model/optimizer/EMA save/restore. Replay the old width16 first six updates exactly.
Save and restore the first actual update before continuation. Freeze numerical
source, settings, data and parent artifacts.

Evaluate at0/400/1200 on strengths.1/.5, both local splits, raw and EMA, with1/2/5
sampling iterations. At1200 add the existing fixed partial masks. Compare the
paired arms and the completed original `warm_detail16`1200 checkpoint evaluated
under the same sampler. The original arm distinguishes progress-distribution and
conditioning changes from the path change. Raw weights and five iterations are
primary; EMA and intermediate checkpoints are diagnostic, never substitutes.

For local advance, `progress_bridge` at1200 must gain at least.03 mean blur F1
in BOTH splits against (a) `progress_rough` at five iterations and (b) original
`warm_detail16` at one iteration. Its five-iteration mean halo must not increase
by more than.01, and mean ink deficit must not increase by more than.01, in any
clean/tone/blur × split against either reference. Clean F1 may fall by at most.01
against either reference. Its one-iteration clean F1 must also stay within.01
of the paired rough arm, and one-iteration halo within+.01 in every family/split.
Retain strength-separated and partial-mask results and every inherited absolute
fit failure; an average gain alone is not an adoption decision.

Inspect target-centered160px native crops for all nine first-variant geometries:
raw train blur.5, raw probe blur.5, raw probe tone.1, raw probe clean.1 and EMA
probe blur.5, with target/source/original1step/paired-rough5step/bridge5step.
Also inspect raw probe blur.1 and blur.5 sheets comparing each model at1/2/5,
plus the fixed partial-mask and blank outputs. Both numerical and actual visual
support are required. No blends, coefficient/time-distribution sweeps or changing
the primary iteration count after seeing results.

Passing would justify a separately declared broad repeated-inference experiment,
not adoption. Its acceptance must include strict retention/topology criteria,
actual synthetic BGRA quality and latency, independent training seed20261005 and
sealed fresh synthetic draws. If the ongoing broader architecture study succeeds,
this exploratory preparation does not delay its confirmation or replace it.
