import json
import threading

import numpy as np
import pytest
from lineartgen_runtime.graph import GraphLineartModel, downsample
from lineartgen_runtime.images import PyramidPlan, prepare_input, to_bgra
from lineartgen_runtime.layer_normalization import bgra_layer_components, normalized_layer_guide


@pytest.mark.parametrize("kind", ["guide", "composite", "painted"])
def test_runtime_detail_matches_original_bgra_information_at_every_depth(kind):
    model = object.__new__(GraphLineartModel)
    model.metadata = dict(
        format_version=5,
        guide_normalization="separate_layers_sum",
        detail_input=kind,
        halo=32,
        supersampling=1,
        max_stroke_width=None,
    )
    model._lock = threading.Lock()
    seen = {}

    def stage(guide, noise, insert, level, *, detail):
        seen[level] = (guide.copy(), detail.copy())
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
    plan = PyramidPlan((64, 64), (64, 64), ((64, 64),))
    combined, noise = prepare_input(sketch.tobytes(), paint.tobytes(), plan, 0.5, image_scale=1.0)
    rough, coverage, alpha = bgra_layer_components(sketch.tobytes(), paint.tobytes(), 64, 64)
    pads = ((0, 0), (32, 32), (32, 32), (0, 0))
    guide = normalized_layer_guide(
        np.pad(rough, pads, constant_values=1), np.pad(coverage, pads), np.pad(alpha, pads), "sum"
    )
    if kind == "guide":
        detail = guide
    elif kind == "composite":
        detail = 1 - 0.4 * (1 - np.pad(combined, pads, constant_values=1))
    else:
        detail = 1 - 0.8 * np.pad(coverage, pads)
    for level in range(3):
        np.testing.assert_array_equal(seen[level][0], guide)
        np.testing.assert_array_equal(seen[level][1], detail)
        guide, detail = downsample(guide), downsample(detail)
    expected = np.frombuffer(to_bgra(combined, plan, alpha_threshold=0), np.uint8).reshape(
        64, 64, 4
    )
    actual = np.frombuffer(raw, np.uint8).reshape(64, 64, 4)
    mask = noise[0, ..., 0] == 0
    np.testing.assert_array_equal(actual[mask], expected[mask])


@pytest.mark.parametrize("version,kind", [(5, None), (5, "unknown"), (4, "painted")])
def test_invalid_detail_manifest_is_rejected_before_loading_onnx(tmp_path, version, kind):
    metadata = dict(
        format_version=version,
        representation="attraction_graph",
        levels=6,
        max_stroke_width=None,
        guide_normalization="separate_layers_sum",
    )
    if kind is not None:
        metadata["detail_input"] = kind
    p = tmp_path / "graph-model.json"
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="detail"):
        GraphLineartModel(p)


def test_tiled_stage_crops_the_detail_field_with_the_other_inputs():
    class Session:
        def run(self, names, feeds):
            insert = np.repeat(np.repeat(feeds["insert"], 2, axis=1), 2, axis=2)
            value = feeds["guide"] + feeds["noise_level"] + 0.3 * feeds["detail"] + insert
            return [np.repeat(value, 9, axis=-1)]

    model = object.__new__(GraphLineartModel)
    model.sessions = [Session()]
    rng = np.random.default_rng(315)
    guide, noise, detail = [rng.random((1, 320, 672, 1), dtype=np.float32) for _ in range(3)]
    insert = rng.random((1, 160, 336, 1), dtype=np.float32)
    full = model.stage(guide, noise, insert, 0, tile_size=None, detail=detail)
    tiled = model.stage(guide, noise, insert, 0, tile_size=256, detail=detail)
    np.testing.assert_array_equal(full, tiled)
