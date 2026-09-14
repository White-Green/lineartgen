import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import distance_transform_cdt

from lineartgen.config import LossConfig, load_config
from lineartgen.training.boundary import boundary_distance_loss, soft_distance
from lineartgen.training.comparison import comparison_configs
from lineartgen.training.losses import image_loss


def target():
    image = np.ones((1, 32, 32, 1), np.float32)
    image[:, 4:28, 14:17] = -1
    image[:, 4:28, 13:14] = 0.4
    image[:, 4:28, 17:18] = 0.6
    return image


def test_binary_soft_distance_matches_truncated_chebyshev_distance():
    ink = np.zeros((1, 17, 17, 1), np.float32)
    ink[0, 8, 8, 0] = 1
    expected = np.minimum(distance_transform_cdt(ink[0, ..., 0] == 0, metric="chessboard"), 4) / 4
    np.testing.assert_array_equal(soft_distance(ink, 4)[0, ..., 0], expected)


@pytest.mark.parametrize("radius", [1, 4, 8])
def test_exact_antialiased_target_has_zero_loss_and_gradient(radius):
    truth = target()
    fn = jax.jit(
        jax.value_and_grad(
            lambda x: boundary_distance_loss(x, truth, jnp.ones_like(x, bool), radius)
        )
    )
    loss, gradient = fn(truth)
    assert float(loss) == 0 and np.all(np.asarray(gradient) == 0)


def test_boundary_loss_detects_shift_width_cut_and_unwanted_bridge():
    truth = target()
    active = jnp.ones_like(truth, bool)
    score = jax.jit(lambda x, y: boundary_distance_loss(x, y, active, 4))
    edits = [np.roll(truth, 1, axis=2), np.roll(truth, 3, axis=2)]
    thick = truth.copy()
    thick[:, 4:28, 11:20] = -1
    cut = truth.copy()
    cut[:, 14:17] = 1
    missing = np.ones_like(truth)
    assert 0 < float(score(edits[0], truth)) < float(score(edits[1], truth))
    assert all(float(score(x, truth)) > 0 for x in [thick, cut, missing])
    bridged = target()
    bridged[:, 14:17] = 1
    assert float(score(truth, bridged)) > 0
    # A descent step on the local objective must improve each nonexact example.
    value_grad = jax.value_and_grad(lambda x: score(x, truth))
    for x in [edits[0], thick, cut, missing]:
        value, gradient = value_grad(x)
        assert np.isfinite(gradient).all() and np.any(np.asarray(gradient) != 0)
        assert float(score(x - 0.1 * gradient, truth)) < float(value)


def test_boundary_loss_has_no_protected_gradient_and_empty_bands_are_finite():
    truth = target()
    prediction = np.roll(truth, 2, axis=2)
    active = np.ones_like(truth, bool)
    active[:, :16] = False
    fn = jax.value_and_grad(lambda x: boundary_distance_loss(x, truth, active, 4))
    loss, gradient = fn(prediction)
    assert float(loss) > 0
    assert np.all(np.asarray(gradient)[~active] == 0)
    altered = prediction.copy()
    altered[~active] = -0.3
    np.testing.assert_array_equal(fn(altered)[0], loss)
    for image in [truth, np.ones_like(truth), -np.ones_like(truth)]:
        loss, grad = jax.value_and_grad(
            lambda x, image=image: boundary_distance_loss(x, image, np.zeros_like(active), 4)
        )(prediction)
        assert float(loss) == 0 and np.all(np.asarray(grad) == 0)
    for image in [np.ones_like(truth), -np.ones_like(truth)]:
        assert float(boundary_distance_loss(prediction, image, np.ones_like(active), 4)) == 0


def test_comparison_changes_only_loss_and_enabled_term_reaches_total_gradient():
    config = load_config("configs/boundary-loss.toml")
    control, boundary = comparison_configs(config, study="boundary_loss").values()
    assert control == boundary.model_copy(update={"loss": control.loss})
    x, y = np.roll(target(), 2, axis=2), target()
    t = np.full_like(x, 0.5)
    loss0, parts0 = image_loss(x, y, t, control.loss, 0.001)
    loss1, parts1 = image_loss(x, y, t, boundary.loss, 0.001)
    np.testing.assert_allclose(
        loss1 - loss0, boundary.loss.boundary_weight * parts1["boundary"], rtol=2e-5
    )
    assert "boundary" not in parts0
    zero = LossConfig(boundary_weight=0)
    assert np.isfinite(float(image_loss(x, y, t, zero, 0.001)[0]))
