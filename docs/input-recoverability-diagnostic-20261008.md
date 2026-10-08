# Information retained in the fixed fitting inputs

The stored floating-point blur pairs are recoverable when the corruption is
known. This rules out total input-information loss as the sole explanation of
the neural fitting failure, but does not prove the network can infer unknown
corruption parameters. The diagnostic is not a deployable restoration method.

For the existing19 blur pairs in each local split, invert the known opacity.3
and sigma2/radius6 zero-padded separable Gaussian operator. A symmetric finite
Toeplitz eigendecomposition gives the Tikhonov solution of the linear inverse;
clip the recovered ink to[0,1]. Regularizers1e-4/1e-8/1e-12 and three input forms
were fixed in a saved plan before running: original float; rendered8bit alpha;
and8bit alpha followed by scale1.1 and clipping. For the last case, undo scaling
only on nonclipped pixels and assign zero ink to clipped white pixels. No model,
training configuration, production default or confirmation case was changed.

Means exclude the blank; these examples summarize the full retained grid:

| Input / regularizer | Train F1 / probe F1 | Train halo / probe halo |
| --- | --- | --- |
| Float /1e-12 | .9831 / .9887 | .00725 / .00524 |
| Alpha8 /1e-4 | .8031 / .9048 | .04046 / .03897 |
| Alpha8 then scale1.1 /1e-4 | .4988 / .5160 | .21642 / .20959 |

Reducing regularization to1e-8 or1e-12 on quantized inputs greatly amplifies
artifacts (F1 falls below.07), so float inversion performance must not be treated
as achievable through the same inverse in Krita. No selected nonblank image
became entirely blank at scale1.1; local signal was still clipped. The forward
operator reproduces every stored float source to maximum error2.10e-7.

Two native160px-crop sheets were inspected after the numerical results, showing
all nine first-variant geometries. Float reconstruction closely follows thin
curves, corners, crossings, gaps and parallel strokes. Alpha8 reconstruction
retains major strokes but has faint thin lines and ringing contours. Scaling
and clipping intensifies those repeated contours. These are artifacts of this
oracle inverse; they do not establish the cause of the neural model's gray fringe.

There are342 reconstructions (2splits ×19pairs ×3input forms ×3regularizers).
All results and arrays are retained in
`runs/detail-head-fitting-20261008/input-recoverability/`, including the prior
plan, exact script/data hashes, forward checks and actual visual-review record.
The input sources here were generated from already normalized clean teachers;
quantization renders those fixed sources to alpha. This is not a new sample of
natural rough drawings. The oracle knows blur/opacity/family labels unavailable
for arbitrary user inputs, and never serves as a candidate or an acceptance gate.
