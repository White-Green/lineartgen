# Conditional bridge option — research note, not implemented

The current source-feature one_pass experiments ignore the newly drawn Gaussian
noise in their features. Repeating one_pass inference therefore does not create
a new cleanup trajectory. AO shows that the small model can fit four fixed pairs;
it does not remove the gray/width errors on unfitted strokes. AP tests more broad
training first. Neither observation proves that conditional ambiguity is the
cause of the remaining errors.

A plausible alternative is to learn a path from the informative rough image to
clean lineart. I²SB explicitly studies bridges whose degraded-image endpoint
contains useful structure, including restoration/deblurring. Its reported image
results are not evidence for our tiny recursive model or Krita workloads.
[Primary paper](https://arxiv.org/abs/2302.05872).
DDBM also formulates paired-distribution transport with a non-noise input endpoint
and learned bridge scores. It is a different training/sampling construction, not
a drop-in loss term. [Primary paper](https://arxiv.org/abs/2309.16948).

Flow matching regresses vector fields along specified conditional paths. The
following simple linear conditional path is a possible diagnostic, not an I²SB
or DDBM reproduction and not a proof of globally optimal transport.
[Flow matching paper](https://arxiv.org/abs/2210.02747).

Let s be the observed rough image, y its teacher, a the spatial edit strength,
and tau an internal solver time distinct from a. At each image pyramid scale,
downsample images/strength before drawing Gaussian epsilon. Define z=s+sigma(a)
epsilon, target y*=where(a>0,y,s), x_tau=(1-tau)z+tau y*, and conditional velocity
y*-z. A network sees evolving x_tau, source s, edit strength a and solver time
tau. Inference starts at z and integrates tau0→1; protected pixels are assigned
s at every step. This preserves informative inputs rather than requiring a
pure-noise starting image. The exact sigma(a), solver and input channel schema
would require a new frozen protocol and metadata/runtime compatibility checks.

Why inspect this: for genuinely ambiguous rough inputs, point regression need
not produce one plausible crisp interpretation. A generative conditional path
could instead represent alternatives. This is a hypothesis, not a guarantee of
correct widths, connectivity, or preservation of the user's intended drawing.
L1 does not in general estimate the conditional mean; current gray output must
not be explained by the MSE conditional-mean argument alone.

Before neural experiments, an analytic two-endpoint toy should check that the
sampler approaches an endpoint, preserves masks, and converges as steps increase.
It must report ambiguity errors against both teachers: sampling a crisp alternative
cannot recover missing information or automatically pass paired reconstruction
gates. An oracle using known endpoints is not a trained restoration model.

Only then consider a matched neural pilot, retaining the user's recursion depths,
scale-loss convention and sqrt(batch) LR. Account for actual network evaluations
and latency. Keep the current adoption gates and also inspect native outputs.
No bridge model, solver, default, production export or plugin change is made by
this note. The next numerical study is not selected here.
