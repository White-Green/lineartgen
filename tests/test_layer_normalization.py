import numpy as np
import pytest
from lineartgen_runtime.graph import normalized_guide
from lineartgen_runtime.images import PyramidPlan, prepare_input
from lineartgen_runtime.layer_normalization import bgra_layer_components, normalized_layer_guide


def test_separate_bgra_components_reconstruct_actual_composite_with_colored_paint():
    rng = np.random.default_rng(314)
    scribble = rng.integers(0, 256, (32, 40, 4), dtype=np.uint8)
    lineart = rng.integers(0, 256, (32, 40, 4), dtype=np.uint8)
    rough, paint, alpha = bgra_layer_components(scribble.tobytes(), lineart.tobytes(), 32, 40)
    plan = PyramidPlan((32, 40), (32, 40), ((32, 40),))
    unpainted, _ = prepare_input(
        scribble.tobytes(), bytes(32 * 40 * 4), plan, 0.35, image_scale=1.0
    )
    np.testing.assert_array_equal(rough, unpainted)
    clean, noise = prepare_input(scribble.tobytes(), lineart.tobytes(), plan, 0.35, image_scale=1.0)
    np.testing.assert_allclose(1 - 2 * (paint + (1 - rough) / 2 * (1 - alpha)), clean, atol=3e-7)
    np.testing.assert_array_equal(0.35 * (1 - alpha), noise)


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_full_edit_matches_old_input_exactly_and_opaque_paint_retains_coverage(mode):
    rng = np.random.default_rng(314)
    rough = rng.uniform(-1, 1, (2, 32, 40, 1)).astype(np.float32)
    zero = np.zeros_like(rough)
    np.testing.assert_array_equal(
        normalized_layer_guide(rough, zero, zero, mode), normalized_guide(rough)
    )
    paint = rng.uniform(0, 1, rough.shape).astype(np.float32)
    np.testing.assert_array_equal(
        normalized_layer_guide(rough, paint, np.ones_like(rough), mode), 1 - 0.8 * paint
    )


@pytest.mark.parametrize("mode", ["sum", "max"])
def test_layer_cues_are_bounded_d4_equivariant_and_batch_independent(mode):
    rng = np.random.default_rng(314)
    rough = rng.uniform(-1, 1, (2, 32, 40, 1)).astype(np.float32)
    alpha = rng.uniform(0, 1, rough.shape).astype(np.float32)
    paint = alpha * rng.uniform(0, 1, rough.shape).astype(np.float32)
    actual = normalized_layer_guide(rough, paint, alpha, mode)
    assert actual.min() >= 0.2 - 1e-6 and actual.max() <= 1
    np.testing.assert_array_equal(
        actual[:1], normalized_layer_guide(rough[:1], paint[:1], alpha[:1], mode)
    )
    rotated = normalized_layer_guide(
        *(np.rot90(x, axes=(1, 2)) for x in (rough, paint, alpha)), mode
    )
    np.testing.assert_array_equal(rotated, np.rot90(actual, axes=(1, 2)))
