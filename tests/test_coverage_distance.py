import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.training.coverage_distance import (
    binary_distance,
    coverage_distance,
    render_distance,
)


def test_all_256_coverage_values_round_trip_exactly():
    clean = (np.arange(256, dtype=np.float32) / 255 * 2 - 1).reshape(16, 16, 1)
    field = coverage_distance(clean)
    np.testing.assert_array_equal(render_distance(field), clean)
    assert np.max(np.abs(np.asarray(render_distance(binary_distance(clean))) - clean)) > 0.99


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_uniform_saturated_image_has_no_fictitious_image_border(value):
    clean = np.full((32, 32, 1), value, np.float32)
    field = coverage_distance(clean)
    np.testing.assert_array_equal(field, value * 8)
    np.testing.assert_array_equal(render_distance(field), clean)


def test_distance_grows_away_from_line_and_faint_line_is_preserved():
    clean = np.ones((32, 32, 1), np.float32)
    clean[:, 16] = 0.6
    field = coverage_distance(clean)
    np.testing.assert_allclose(field[16, 16:20, 0], [0.3, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(render_distance(field), clean)
    assert np.all(np.asarray(render_distance(binary_distance(clean))) == 1)


def test_renderer_protects_fractional_source_and_has_local_gradient():
    field = jnp.array([-3.0, -0.2, 0.1, 3.0]).reshape(1, 4, 1)
    source = jnp.array([0.3, 0.4, 0.5, 0.6]).reshape(1, 4, 1)
    strength = jnp.array([0.5, 0.0, 0.5, 0.5]).reshape(1, 4, 1)
    prediction = jax.jit(render_distance)(field, source, strength)
    np.testing.assert_allclose(prediction.ravel(), [-1.0, 0.4, 0.2, 1.0])
    gradient = jax.grad(lambda x: render_distance(x, source, strength).sum())(field)
    np.testing.assert_array_equal(gradient.ravel(), [0.0, 0.0, 2.0, 0.0])


@pytest.mark.parametrize(
    "clean,radius",
    [
        (np.zeros((3, 3)), 8),
        (np.full((3, 3, 1), np.nan), 8),
        (np.full((3, 3, 1), 1.1), 8),
        (np.zeros((3, 3, 1)), 0.4),
    ],
)
def test_invalid_targets_are_rejected(clean, radius):
    with pytest.raises(ValueError):
        coverage_distance(clean, radius)
