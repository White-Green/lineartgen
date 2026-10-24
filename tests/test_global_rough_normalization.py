import json

import numpy as np
import pytest
from lineartgen_runtime.global_normalization import normalized_global_rough
from lineartgen_runtime.graph import GraphLineartModel, downsample


def test_global_normalization_preserves_ratios_without_mixing_images():
    rng = np.random.default_rng(836)
    ink = rng.uniform(0.02, 1, (3, 40, 48, 1)).astype(np.float32)
    value = 1 - 0.8 * ink
    result = normalized_global_rough(value)
    np.testing.assert_allclose(
        result, normalized_global_rough(1 - 0.8 * ink * 0.4), rtol=0, atol=3e-7
    )
    np.testing.assert_array_equal(
        result, np.concatenate([normalized_global_rough(x[None]) for x in value])
    )
    np.testing.assert_array_equal(
        normalized_global_rough(np.rot90(value, axes=(1, 2))), np.rot90(result, axes=(1, 2))
    )
    padded = np.pad(value, ((0, 0), (32, 32), (32, 32), (0, 0)), constant_values=1)
    np.testing.assert_array_equal(normalized_global_rough(padded)[:, 32:-32, 32:-32], result)
    np.testing.assert_array_equal(normalized_global_rough(np.ones_like(value)), 1)


def test_global_normalization_keeps_weak_signals_below_the_peak_floor():
    value = np.full((1, 16, 16, 1), 1 - 0.8 * 0.001, np.float32)
    result = normalized_global_rough(value)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, 0.96, atol=2e-6)


@pytest.mark.parametrize("policy", [None, "identity", "global_peak"])
def test_rough_normalization_runs_once_before_the_pyramid(policy):
    model = object.__new__(GraphLineartModel)
    model.metadata = dict(
        guide_normalization="separate_layers_sum", detail_input="painted", context_input="raw"
    )
    if policy is not None:
        model.metadata["rough_input_normalization"] = policy
    source = np.ones((1, 128, 128, 1), np.float32)
    rough = source.copy()
    rough[:, 64, :64] = 0.68
    rough[:, 64, 64:] = 0.84
    seen = {}

    def stage(guide, noise, insert, level, *, detail, rough):
        seen[level] = rough.copy()
        return np.zeros((*guide.shape[:-1], 9), np.float32)

    model.stage = stage
    model.heads(
        source,
        np.full_like(source, 0.5),
        3,
        layer_guide=source,
        detail_guide=source,
        rough_guide=rough,
    )
    expected = normalized_global_rough(rough) if policy == "global_peak" else rough
    for level in range(3):
        np.testing.assert_array_equal(seen[level], expected)
        expected = downsample(expected)
    # Renormalizing each coarse stage would lose this subpixel coverage.
    if policy == "global_peak":
        assert seen[2][0, 16, 0, 0] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "version,context,policy",
    [
        (9, "raw", None),
        (9, "raw", "unknown"),
        (9, "control", "global_peak"),
        (7, "raw", "global_peak"),
    ],
)
def test_invalid_rough_contract_is_rejected_before_loading_onnx(tmp_path, version, context, policy):
    metadata = dict(
        format_version=version,
        representation="attraction_graph",
        levels=6,
        max_stroke_width=None,
        guide_normalization="separate_layers_sum",
        detail_input="painted",
        context_input=context,
    )
    if policy is not None:
        metadata["rough_input_normalization"] = policy
    path = tmp_path / "graph-model.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="rough normalization"):
        GraphLineartModel(path)
