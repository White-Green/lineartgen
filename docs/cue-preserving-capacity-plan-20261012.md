# Cue-preserving capacity diagnostic (CR)

Frozen before training. This is a learnability experiment, not a replacement
model or an adoption/confirmation campaign. Synthetic inputs only; Krita,
production weights, sealed holdout and independent confirmation seed unchanged.

## Question and comparison

Can a small self-recursive local CNN learn to restore line width, opacity and
connectivity when the source retains the required information? Compare two
independent fresh models with 16 / 32 channels (15,809 / 59,265 parameters),
three two-convolution residual blocks and the same external recursion. No
shared weights or output exchange. Initial output heads are zero, so outputs
and losses must match at all six recursion depths before training. Equal
updates and data, not equal compute or independently optimized learning rates.

## Data and information audit

96 fixed 256px teachers per split: 8 instances each of straight, thin, Bezier,
corner, junction, crossing, parallel, gap and taper; 4 blank; 4 black rectangles;
8 wide lines; 8 mixed shapes. A 24px white margin surrounds the 208px drawing.
Black/blank are repeated constants, not distinct unseen shapes. All other
train/probe raster hashes must be disjoint. Probe shapes are new random
instances of the same families, not withheld families or real rough drawings.

For teacher opacity a=(1-y)/2, source s=1-2*0.4*A(a), where
A=0.4*I+0.6*G and G is circular convolution with a normalized, sampled
Gaussian (computed via its Fourier multiplier), sigma=1.5px.
Its spectrum lies in [0.4,1], so the condition number is at most 2.5. The
known inverse amplifies source errors by at most 6.25. No dilation, displacement,
quantization or threshold is applied. Audit float32 inverse max error <2e-6
for every stored source. This oracle is an audit/reference only; no inverse or
clean teacher is passed into the model. The source guide is available to the
network, including its retained high-frequency core.

Source and mask are downsampled first. Independent Gaussian noise is then
added at each resolution: x=s+0.25*t*N(0,1). Strength modes: whole image 0.1,
whole image 0.5, diagonal partial 0.5 (protected when x+0.3*y<0.60*size).
Targets are the downsampled clean image in active pixels and the source
elsewhere. Fresh training noise; two separate evaluation noise draws per case.

## Training

One forward restoration through the spatial pyramid, progress=1 at each
scale. No multi-time rollout or flow-matching objective in this diagnostic.
Loss is tone-balanced interval L1 (power 0.5), noise-weight floor 0.05,
identity weight 0.1, all other auxiliary weights zero. Every stage contributes
with normalized weights 0.5**level; gradients traverse coarse predictions.
Recursion cycles 1..6. Batch 4, 6,000 updates, AdamW, clip 1, decay 1e-4,
EMA 0.999. Base LR 5e-5 times sqrt(4)=1e-4, cosine to 1e-5, no warmup.
Matched cases, modes, Gaussian samples, recursion depth and LR at each update;
execution order alternates. Each model has its own optimizer and EMA.

Model seed 42; data 2026101291; batch 2026101292; training Gaussian
2026101293; evaluation Gaussian 2026101294. Independent confirmation seed
20261005 is not used. Full checkpoints and raw/EMA exports at 0, 6, 2000,
6000. Reproduce first six actual GPU updates including full optimizer state.
Restore and audit all exports. The final raw step 6000 is the only evaluated
checkpoint; no best-checkpoint selection. Training loss is descriptive.

## Evaluation fixed before training

Depth 3, one forward restoration, 192 cases x 3 masks x 2 fresh noise draws
x 2 models = 2304 model rows. Separately 192 x 3 x 2 (source/analytic inverse)
=1152 reference rows. Protected pixels must equal source exactly. Inputs,
checkpoints, source code and numerical config are hash-bound. Target-oracle
topology queries report valid and total counts, including invalid raster
queries; no assumption that all procedural graph queries survived rasterizing.

Report by split, family and mask, with paired width32-minus-width16 metrics.
F1 at alpha .1/.25/.5/.75; continuous coverage F1; opacity deficit; gray halo;
profile width error; connection cuts/bridges; protected errors; blank max
opacity and black interior opacity (distance >8px from teacher boundary).
Profiles/topology evaluated only for whole masks. Diagnostic criteria, per
applicable row: F1 at .5 >=.98, deficit <=.02, halo excess <=.003, profile width
MAE <=.2px, valid cut/bridge rates zero, blank max <=.05, black interior mean
>=.95. These are fit targets, not the old 229-check adoption gate. Report
counts and magnitude of failures, not just a binary result.

Save every prediction. Native 256px panels (target/source/16ch/32ch), noise
draw 0, fixed indices 0,8,16,24,32,40,48,56,64,72,76,80,88, for both splits;
whole .5 and partial .5. Six shape/coverage groups per split across these
masks are reviewed without thresholding, rescaling contrast or smoothing.

## Interpretation limits

Both fit and generalize: small architecture can represent this bounded task;
does not establish minimum size or real-sketch quality. Width32 fits more
accurately on both: evidence for more capacity under this schedule. Good train
but poor probe: investigate distribution/regularization before widening.
Neither fits: investigate conditioning/objective/optimization before calling
it a capacity limit. One seed and 6000 steps do not prove global optima.

Compared with CO/CP/CQ, this experiment changes source corruption, resolution,
teacher set, initialization and training objective together. Only the two
width arms are a controlled causal comparison; do not attribute improvement
over the older runs solely to removal of ambiguous source degradation. Easy,
fixed, invertible blur is not realistic rough-sketch coverage and this model
has not been validated for multi-time rollout, 512px, ONNX or Krita.
