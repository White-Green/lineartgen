# Explicit synthetic BGRA quality check

This diagnostic exercised the actual `LineartModel.infer` BGRA API on 24 fixed,
already reused whole-image geometry cases, with 1/2/5 iterations. Models were the
matched2k EMA control and the **already rejected** raw-guide alpha1/16 EMA blend.
The trained tensors were fixed; paired ONNX files differed only in input-scale
metadata (1.1 versus1.0). The ten-model ink-guide experiment remains unchanged.

All **288 predictions** completed. Rough arrays were rendered as black transparent
scribble PNGs, with empty lineart layers, then passed through the actual input
conversion, NumPy sampler, ONNX graph, and output alpha threshold. Measurements
therefore describe displayed BGRA output, not the earlier continuous-array metric
domain. They are not a replacement acceptance test or an independent test set.

For the rejected candidate, one iteration, means across the eight selected shapes:

| Condition | Scale | Halo excess | Missing profiles | Cuts | Wrong bridges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strong blur, noise .5 | 1.1 | .016732 | .785156 | .875000 | .000000 |
| Strong blur, noise .5 | 1.0 | .026756 | .750781 | .875000 | .000000 |
| Strong strokes, noise .1 | 1.1 | .084578 | .125000 | .629167 | .500000 |
| Strong strokes, noise .1 | 1.0 | .137428 | .091146 | .408333 | .833333 |
| Strong strokes, noise .5 | 1.1 | .109397 | .140191 | .570833 | .500000 |
| Strong strokes, noise .5 | 1.0 | .168477 | .125000 | .404167 | .666667 |

Only defined probes enter each metric mean. These small, selected, overlapping
diagnostics are not estimates of general user performance. Scale1.0 retains more
line information and reduces some cuts, but increases halos and wrong bridges.
It does not solve the strongest blurred-thin-line cases. No whole case was erased
by scale1.1 in this subset; 55.96% of nonzero input-ink pixels survived preprocessing
versus100% at scale1.0. The standalone alpha-level audit separately established
the exact alpha1–11 clipping cutoff.

Four native-pixel sheets were actually inspected: the three one-step conditions
and strong-strokes .5 at five steps. Thin blurred lines still disappear, nearby
strokes/gaps merge, and repeated inference can strengthen wrong joins. This
supports testing input handling alongside learned restoration; it does not justify
changing the deployed scale or accepting the rejected model.

Artifacts: `runs/krita-input-scale-diagnostic-20261008/` contains the pre-inference
plan, exact model hashes, unchanged-graph metadata checks, 24 transparent input
PNGs, 72 native output panels, metrics, and `script-as-run.py`. The last file hashes
to the script digest in the plan; later source edits only wrapped two long text
strings for lint. Production model assets and runtime defaults were not changed.
