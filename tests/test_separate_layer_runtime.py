import threading

import numpy as np
import pytest
from lineartgen_runtime.graph import GraphLineartModel, downsample
from lineartgen_runtime.images import PyramidPlan, prepare_input, to_bgra
from lineartgen_runtime.layer_normalization import bgra_layer_components, normalized_layer_guide


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_actual_bgra_runtime_uses_original_layers_before_pyramid_and_preserves_protection(mode):
    model = object.__new__(GraphLineartModel)
    model.metadata = dict(
        guide_normalization=f"separate_layers_{mode}",
        halo=32,
        supersampling=1,
        max_stroke_width=None,
    )
    model._lock = threading.Lock()
    seen = {}

    def stage(guide, noise, insert, level):
        seen[level] = guide.copy()
        return np.full((*guide.shape[:-1], 9), -20, np.float32)

    model.stage = stage
    sketch = np.zeros((64, 64, 4), np.uint8)
    sketch[24:40, :, 3] = 102
    paint = np.zeros_like(sketch)
    paint[:, :32, :3] = [30, 80, 140]
    paint[:, :32, 3] = 255
    paint[28:36, 32:48, 3] = 128
    raw = model.infer(
        scribble=sketch.tobytes(),
        lineart=paint.tobytes(),
        width=64,
        height=64,
        strength=0.5,
        seed=42,
        denoise_steps=3,
    )
    rough, coverage, alpha = bgra_layer_components(sketch.tobytes(), paint.tobytes(), 64, 64)
    pads = ((0, 0), (32, 32), (32, 32), (0, 0))
    expected = normalized_layer_guide(
        np.pad(rough, pads, constant_values=1), np.pad(coverage, pads), np.pad(alpha, pads), mode
    )
    for level in range(3):
        np.testing.assert_array_equal(seen[level], expected)
        expected = downsample(expected)
    plan = PyramidPlan((64, 64), (64, 64), ((64, 64),))
    clean, noise = prepare_input(sketch.tobytes(), paint.tobytes(), plan, 0.5, image_scale=1.0)
    baseline = np.frombuffer(to_bgra(clean, plan, alpha_threshold=0), np.uint8).reshape(64, 64, 4)
    output = np.frombuffer(raw, np.uint8).reshape(64, 64, 4)
    np.testing.assert_array_equal(output[noise[0, ..., 0] == 0], baseline[noise[0, ..., 0] == 0])


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_layer_policy_rejects_heads_call_without_original_layer_cue(mode):
    model = object.__new__(GraphLineartModel)
    model.metadata = {"guide_normalization": f"separate_layers_{mode}"}
    source = np.ones((1, 64, 64, 1), np.float32)
    with pytest.raises(ValueError, match="original layer"):
        model.heads(source, source, 1)
