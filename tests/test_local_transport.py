import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.training.local_transport import local_opacity_transport


@pytest.mark.parametrize("distance", ["squared", "smooth_absolute"])
def test_antialias_identity_saturation_and_protection(distance):
    alpha = np.zeros((1, 16, 16, 1), np.float32)
    alpha[:, :, 7, :] = 1
    alpha[:, :, 6, :] = 0.25
    target = jnp.asarray(1 - 2 * alpha)
    active = jnp.ones_like(target, dtype=bool)

    def loss(p, y, a):
        return local_opacity_transport(p, y, a, distance=distance)

    run = jax.jit(jax.value_and_grad(loss))
    value, gradient = run(target, target, active)
    assert value == 0 and np.count_nonzero(gradient) == 0
    physical = jnp.where(target == 1, 2.0, target)
    value, gradient = run(physical, target, active)
    assert value == 0 and np.count_nonzero(gradient) == 0
    physical = jnp.where(target == -1, -2.0, target)
    value, gradient = run(physical, target, active)
    assert value > 0 and gradient[:, :, 7, :].sum() < 0
    wrong = target.at[:, :, 2, :].set(-2.0)
    value, gradient = run(wrong, target, active)
    assert value > 0 and gradient[:, :, 2, :].sum() < 0
    value, gradient = run(wrong, target, jnp.zeros_like(active))
    assert value == 0 and np.count_nonzero(gradient) == 0


def test_rejects_mean_image_but_does_not_claim_spatial_geometry():
    alpha = np.zeros((1, 16, 16, 1), np.float32)
    alpha[:, :, 6, :] = 0.8
    alpha[:, :, 7, :] = 0.2
    left = jnp.asarray(1 - 2 * alpha)
    right = jnp.roll(left, 2, axis=2)
    active = jnp.ones_like(left, dtype=bool)
    assert local_opacity_transport(left, right, active) == 0
    assert local_opacity_transport((left + right) / 2, left, active) > 0
    # Deliberately shuffled pixels retain the histogram; geometry losses are necessary.
    shuffled = np.asarray(left).reshape(-1).copy()
    np.random.default_rng(91).shuffle(shuffled)
    assert local_opacity_transport(jnp.asarray(shuffled.reshape(left.shape)), left, active) == 0
