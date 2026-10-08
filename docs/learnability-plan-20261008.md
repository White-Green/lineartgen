# Controlled learnability diagnostic

The ten ink-guide candidates failed. Before another development-set search, fit
a small fixed set of procedurally generated targets. This is a diagnostic, not a
replacement-model selection or a change to the campaign's acceptance rules.

Use only the training geometry generator: nine kinds × two variants plus a blank,
at 512px, widths .6–6px. For each clean target make clean, opacity .3, and opacity
.3 with Gaussian sigma2 sources. No dilation/displacement: this removes one source
of correspondence ambiguity. Freeze all 57 source/target pairs before training.
Make a separate 57-pair probe set with different generator seeds; it is a local
diagnostic, not the unopened campaign confirmation set. Original rough images
before Gaussian noise are the guides, always raw encoding.

Four arms, fixed before the first update:

1. Common 100k initialization, conventional clipped MSE/identity/edge/contrast.
2. Same initialization and loss, replacing only pixel MSE by existing interval MSE.
3. Random initialization, same small architecture and conventional loss.
4. Random initialization, doubled channels [8,20,40,80], conventional loss.

The first pair isolates saturated-output gradient recovery on this controlled
task. The last pair probes whether additional capacity helps fitting; the same
seed is not parameter correspondence across widths. Comparing warm vs random
initialization is descriptive. This diagnostic cannot prove sufficiency for
natural lineart or separate every possible optimization and capacity effect.

All arms see identical pair indices, noise and masks. Train exactly 1,200 updates,
batch4, one microbatch, base LR1e-4 × sqrt(4)=2e-4, warmup50, cosine to1e-5 at1200.
Reset optimizers; retain EMA .999 as a diagnostic. Cycle recursion1–6, retain
scale_loss_multiplier=0, use the existing uniform strength distribution/masks,
and add independent Gaussian noise only after downsampling. High LR, tiny fixed
dataset and repeated samples are deliberate fitting-probe changes, not a proposed
production recipe. Production settings and weights remain untouched.

Evaluate raw (primary for fitting) and EMA at steps0,400,1200: all train and probe
pairs at strengths .1/.5, whole image, three recursive scales. Save native crops,
continuous halo/ink/F1 metrics and wrong-white preclip saturation. Also evaluate
fixed partial masks at1200 and assert protected pixels exactly preserved. A
descriptive fit flag requires F1(alpha50)≥.95, ink deficit≤.05 and halo≤.02 on
each nonblank training condition; report individual failures and all families.
This flag is not campaign acceptance. No arm advances to production from this
diagnostic. No coefficient fitting, early winner selection, or extension past1200.

Before training: tests for independent train/probe generation, actual rough-guide
and protected-target semantics; all four arms through six GPU updates, complete
state save/restore exact. Pin source/config/data hashes. Resume only full grouped
checkpoints. Inspect initial/final native output and compare fitting with local
probe behavior before choosing the next full experiment. Existing development
cases and prospective confirmation cases are not used by this diagnostic.
