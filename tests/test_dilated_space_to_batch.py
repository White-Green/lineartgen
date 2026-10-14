import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.models.dilated import dilated_same, local_convolution


@pytest.mark.parametrize("size", [(4, 6), (18, 22), (32, 32)])
@pytest.mark.parametrize("dilation", [8, 16])
def test_outputs_and_both_gradients_match_direct_conv(size, dilation):
    x = jax.random.normal(jax.random.PRNGKey(1), (2, *size, 3))
    kernel = jax.random.normal(jax.random.PRNGKey(2), (3, 3, 3, 4)) * 0.1

    def direct(x, kernel):
        return jax.lax.conv_general_dilated(
            x,
            kernel,
            (1, 1),
            "SAME",
            rhs_dilation=(dilation, dilation),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    def objective(fn):
        def loss(x, kernel):
            y = fn(x, kernel)
            return jnp.mean(y * y), y

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))

    a = objective(direct)(x, kernel)
    b = objective(lambda x, k: dilated_same(x, k, dilation))(x, kernel)
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, atol=3e-6, rtol=3e-6), a, b)


@pytest.mark.parametrize("dilation", [1, 4, 16])
def test_layer_adapter_keeps_bias_and_small_dilations(dilation):
    layer = nnx.Conv(3, 4, (3, 3), padding="SAME", kernel_dilation=dilation, rngs=nnx.Rngs(3))
    layer.bias[...] = jnp.full_like(layer.bias[...], 0.07)
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 18, 22, 3))
    actual, expected = local_convolution(layer, x), layer(x)
    if dilation <= 4:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
