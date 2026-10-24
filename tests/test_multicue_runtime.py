import json
import threading

import numpy as np
import pytest
from lineartgen_runtime.graph import GraphLineartModel, downsample


def test_original_rough_intensity_is_padded_then_downsampled_without_peak_normalization():
    model = object.__new__(GraphLineartModel)
    model.metadata = dict(
        format_version=7,
        guide_normalization="separate_layers_sum",
        detail_input="painted",
        context_input="raw",
        halo=32,
        supersampling=1,
        max_stroke_width=None,
    )
    model._lock = threading.Lock()
    captured = []

    def stage(guide, noise, insert, level, *, detail, rough):
        captured.append((level, rough.copy()))
        return np.full((*guide.shape[:-1], 9), -20, np.float32)

    model.stage = stage
    sketch = np.zeros((64, 64, 4), np.uint8)
    sketch[25:39, :, 3] = 51
    sketch[29:35, :, 3] = 102
    paint = np.zeros_like(sketch)
    paint[31:33, :32, 3] = 255
    model.infer(
        scribble=sketch.tobytes(),
        lineart=paint.tobytes(),
        width=64,
        height=64,
        strength=0.5,
        seed=0,
        denoise_steps=6,
    )
    original = (1 - sketch[..., 3].astype(np.float32) / 255) * 2 - 1
    raw = 1 - 0.4 * (
        1
        - np.pad(original[None, ..., None], ((0, 0), (32, 32), (32, 32), (0, 0)), constant_values=1)
    )
    by_level = dict(captured)
    for level in range(6):
        np.testing.assert_array_equal(by_level[level], raw)
        raw = downsample(raw)


@pytest.mark.parametrize("version,context", [(7, None), (7, "unknown"), (5, "raw")])
def test_invalid_context_contract_rejected_before_onnx_loading(tmp_path, version, context):
    metadata = dict(
        format_version=version,
        representation="attraction_graph",
        levels=6,
        max_stroke_width=None,
        guide_normalization="separate_layers_sum",
        detail_input="painted",
        context_input=context,
    )
    p = tmp_path / "graph-model.json"
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="context"):
        GraphLineartModel(p)
