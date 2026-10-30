# Intended-stroke selection preview runtime

The goal is to remove retracing and stray sketch strokes while preserving intended
paths, branches and pressure-dependent width. This runtime work does not establish
that the learned model meets that goal. Model adoption requires separate output
review; the existing default assets have not been changed by this implementation.

`LineartModel` recognizes `spatial-stage-model.json`. One dynamically sized ONNX
CNN is shared by all one to six recursion stages. Image normalization, noise after
downsampling and feedback use the complete image. Within each stage, the CNN uses
512-pixel cores with a 64-pixel halo; the complete stage head is assembled before
producing feedback for the next stage. Unaligned coarse shapes use a complete CNN
call to preserve resize coordinates. Rendering retains native, variable widths.

The implementation was ported from the research branch's frozen stage runtime,
feedback and tiling scripts. Provider reporting now uses the providers of the
constructed ONNX session, including CPU fallback if DirectML initialization fails.
The public BGRA arguments and the Krita interaction remain unchanged.

Validation on 2026-10-30 used the previous 42,000-update research model solely as a
runtime fixture: all 15 saved full-image, rectangular, protected, blank and tiny
BGRA outputs at depths 1/3/6 matched byte-for-byte through the public API. JAX and
Flax were not imported. The existing spatial, curve, graph and raster runtime suites
passed (45 tests). The frozen dependency export also succeeded. These checks ran
on Linux CPU; Windows/DirectML execution requires the final packaged CI test.

The research records are `runs/intent-spatial-stage-bridge-20261030` and
`runs/intent-spatial-stage-tiling-20261030` in the research worktree. The latter
verified exact full-stage versus tiled results, including all six heads at 1024px.
