import jax
import jax.numpy as jnp
import numpy as np

from lineartgen.config import LossConfig
from lineartgen.training.losses import image_loss
from lineartgen.training.paper import paper_peak_loss


def test_paper_stationarity_keeps_continuous_antialiasing_and_faint_ink():
    target = jnp.linspace(-1, 1, 17 * 19).reshape(1, 17, 19, 1)
    active = jnp.ones_like(target, bool)
    loss, grad = jax.value_and_grad(
        lambda p, target=target, active=active: paper_peak_loss(p, target, active)
    )(target)
    assert float(loss) == 0
    np.testing.assert_array_equal(grad, 0)


def test_localized_false_ink_is_emphasized_without_erasing_teacher_or_protected_pixels():
    target = jnp.ones((1, 16, 16, 1)).at[:, 0, :, :].set(-0.6)
    active = jnp.ones_like(target, bool).at[:, 8:, 8:, :].set(False)
    prediction = target.at[:, 5, 5, :].set(0).at[:, 0, 5, :].set(0.6)
    prediction = prediction.at[:, 12, 12, :].set(-1)
    loss, grad = jax.value_and_grad(
        lambda p, target=target, active=active: paper_peak_loss(p, target, active)
    )(prediction)
    np.testing.assert_allclose(loss, 0.25 / 3)
    assert float(grad[0, 5, 5, 0]) < 0
    assert float(grad[0, 0, 5, 0]) == 0
    assert float(grad[0, 12, 12, 0]) == 0


def test_unaligned_small_and_empty_images_have_finite_loss_and_gradients():
    for shape in [(2, 3, 5, 1), (2, 17, 19, 1)]:
        for value in (-1, 1):
            target = jnp.full(shape, value, dtype=jnp.float32)
            for protected in (True, False):
                active = jnp.full(shape, not protected)
                loss, grad = jax.value_and_grad(
                    lambda p, target=target, active=active: paper_peak_loss(p, target, active)
                )(jnp.zeros(shape))
                assert np.isfinite(loss) and np.isfinite(grad).all()
                if protected or value == -1:
                    assert float(loss) == 0
                    np.testing.assert_array_equal(grad, 0)
                else:
                    np.testing.assert_allclose(loss, 0.25)


def test_image_loss_adds_only_the_configured_paper_term():
    target = jnp.ones((1, 8, 8, 1))
    pred = target.at[:, 4, 4, :].set(0.2)
    noise = jnp.full_like(target, 0.5)
    base = LossConfig(contrast_weight=0)
    extra = base.model_copy(update={"paper_peak_weight": 0.3})
    first, _ = image_loss(pred, target, noise, base, 0.01)
    second, parts = image_loss(pred, target, noise, extra, 0.01)
    expected = paper_peak_loss(pred, target, noise > 0.01)
    np.testing.assert_allclose(second - first, 0.3 * expected, rtol=1e-6)
    np.testing.assert_array_equal(parts["paper_peak"], expected)
