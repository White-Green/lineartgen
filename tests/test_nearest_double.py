import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.models.upsample import nearest_double


@pytest.mark.parametrize(
    "shape", [(2, 32, 32, 1), (2, 32, 32, 16), (4, 256, 256, 1), (4, 16, 16, 40)]
)
@pytest.mark.parametrize("pattern", ["ones", "integer"])
def test_exact_forward_and_analytical_adjoint(shape, pattern):
    b, h, w, c = shape
    out = (b, h * 2, w * 2, c)
    x = jnp.asarray(np.random.default_rng(42).normal(size=shape).astype(np.float32))
    weights = (
        np.ones(out, np.float32)
        if pattern == "ones"
        else (np.arange(np.prod(out)).reshape(out) % 7).astype(np.float32)
    )
    expected = (
        weights[:, 0::2, 0::2]
        + weights[:, 1::2, 0::2]
        + weights[:, 0::2, 1::2]
        + weights[:, 1::2, 1::2]
    )
    np.testing.assert_array_equal(nearest_double(x), jax.image.resize(x, out, "nearest"))
    gradient = jax.jit(jax.grad(lambda v, z: (nearest_double(v) * z).sum()))
    actual = gradient(x, jnp.asarray(weights))
    np.testing.assert_array_equal(actual, expected)
    for _ in range(3):
        np.testing.assert_array_equal(gradient(x, jnp.asarray(weights)), actual)


def test_jvp_and_transpose_with_floating_values_and_odd_input_dimensions():
    rng = np.random.default_rng(22)
    x, dx = [jnp.asarray(rng.normal(size=(1, 5, 7, 3)).astype(np.float32)) for _ in range(2)]
    dy = jnp.asarray(rng.normal(size=(1, 10, 14, 3)).astype(np.float32))
    y, tangent = jax.jvp(nearest_double, (x,), (dx,))
    np.testing.assert_array_equal(y, jax.image.resize(x, dy.shape, "nearest"))
    np.testing.assert_array_equal(tangent, jax.image.resize(dx, dy.shape, "nearest"))
    gradient = jax.vjp(nearest_double, x)[1](dy)[0]
    expected = np.asarray(dy)[:, 0::2, 0::2].astype(np.float64)
    expected += np.asarray(dy)[:, 1::2, 0::2]
    expected += np.asarray(dy)[:, 0::2, 1::2]
    expected += np.asarray(dy)[:, 1::2, 1::2]
    np.testing.assert_allclose(gradient, expected, atol=4e-7, rtol=3e-7)
    np.testing.assert_allclose(
        np.sum(np.asarray(tangent, np.float64) * np.asarray(dy, np.float64)),
        np.sum(np.asarray(dx, np.float64) * np.asarray(gradient, np.float64)),
        atol=2e-6,
        rtol=3e-7,
    )
