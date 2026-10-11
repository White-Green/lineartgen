# CA: do bounded endpoints and time modulation transfer beyond the two bars?

Registered while BZ trains; fix both initial checkpoints now, independent of
BZ's eventual result. Train noisy-anchor bounded endpoint models on AD's fixed
512px AA curves/corners/parallel lines/gaps. Plain starts from BV noisy6000
(130628 parameters), modulated from BX noisy6000 (134688). Both had the same
toy data and6000-update schedule; their trained inherited weights now differ.
This compares complete recipes, not modulation at a matched weight state.

Use BC's exact four train pairs999/1095/1287/1383 and four unfitted probes
807/903/1191/1479; source is blur sigma2, opacity.3, strength.5, contrast1.1.
Fresh Adam/EMA for both;6000 geometry updates each, batch4, base5e-5*sqrt(4)=1e-4,
constant LR1e-4 without warmup. Same BC seed2026101093, step-indexed data/noise,
depth1–6 rotation,25% tau0 and sin²-transformed continuous tau. Gaussian .25*t
is added independently after every resize. Both arms receive identical arrays.

Same BV/BX endpoint objective: top-stage raw interval error /max(u,1/32), squared
and averaged over active pixels per image. White/black permit saturation, AA
uses point error. Gradients traverse clipped smaller endpoint inserts. Both
anchor to noisy state and sample with bounded endpoints at32 uniform Euler
steps; no solver selection. No additional geometry loss, early-time change,
posterior teacher or new corruption. Keep BZ separate.

Freeze initial/config/data hashes. Save and fully restore0/6/2000/6000 including
optimizer/EMA;16 raw/EMA exports. Replay first6 updates exactly. Journal all6000
shared input hashes/depths/time means. Test actual updates, independent arms,
full next-update restore, sampler choice/protection and the inherited512px
downsample/noise protocol. Initial learned weights must match their own parents.

Compare both final raw6000 models with BC's frozen point-velocity6000 result.
BC keeps its original sin²32 sampler; replay its25 physical/clipped blocks
exactly (24 whole plus partial). This old recipe had AZ36000 warm-start weights,
different velocity loss and solver; it is a quality reference, not a causal
control for time modulation. Plain versus modulated has matched geometry data,
loss/sampler/update count but different toy-trained weights and parameter count.

Evaluation: depth1/3/6, four existing Gaussian draws, four train/four probe
cases; partial mask at depth6.288 metric rows,75 blocks. At primary depth3,
retain all22 fit criteria per draw (88/model): TRAIN F1>=.98, deficit<=.02,
halo<=.003, width MAE<=.2, cuts/bridges0 when defined. Compare F1 drop<=.005
and deficit increase<=.01 on all eight cases/four draws: plain vs BC, modulated
vs BC, modulated vs plain (64 guards/comparison,192 total). These are diagnostics,
not a new broad adoption rule. Finite/protected checks mandatory.

Review all five fixed native/overview sheets with all four draws and unchanged
AD crops. No cherry-picked crops or checkpoints. Repeated development probes
are not independent confirmation. This is geometry fitting, not broad corpus
validation; no sealed conditions or independent seed. No production model,
Krita plugin, runtime default or deployment changes.
