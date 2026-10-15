import jax.numpy as jnp
import numpy as np
from flax import nnx
from group_field_model import GroupConv, action, feature_action
from scale_dilation_model import convolve, dilation


def test_legacy_matches_actual_group_convolution():
    for size in (5, 32):
        for d in (1, 4, 16):
            layer = GroupConv(1, 2, dilation=d, rngs=nnx.Rngs(23))
            x = jnp.asarray(np.random.default_rng(15).normal(size=(1, size, size, 8)), jnp.float32)
            np.testing.assert_array_equal(convolve(layer, x, 4, "legacy"), layer(x))


def test_coarse_off_center_taps_are_restored():
    layer = GroupConv(1, 1, lifting=True, dilation=16, rngs=nnx.Rngs(14))
    k = jnp.ones((3, 3, 1, 1)).at[1, 1, 0, 0].set(0)
    layer.kernel[...] = k
    x = jnp.ones((1, 5, 5, 1))
    np.testing.assert_array_equal(convolve(layer, x, 5, "legacy"), 0)
    for mode in ("physical", "bounded"):
        value = np.asarray(convolve(layer, x, 5, mode))
        np.testing.assert_array_equal(value[:, 2, 2, :], 8)
        assert np.any(value > 0)
    assert [dilation(16, k, (1, 256 // 2**k, 256 // 2**k, 8), "physical") for k in range(6)] == [
        16,
        8,
        4,
        2,
        1,
        1,
    ]


def test_modified_group_convolutions_preserve_d4_action():
    layer = GroupConv(1, 1, dilation=16, rngs=nnx.Rngs(42))
    x = jnp.asarray(np.random.default_rng(9).normal(size=(1, 8, 8, 8)), jnp.float32)
    for mode in ("physical", "bounded"):
        y = convolve(layer, x, 3, mode).reshape(1, 8, 8, 8, 1)
        for g in range(8):
            moved = feature_action(x.reshape(1, 8, 8, 8, 1), g).reshape(x.shape)
            actual = convolve(layer, moved, 3, mode).reshape(y.shape)
            np.testing.assert_allclose(actual, feature_action(y, g), atol=2e-6, rtol=1e-5)
    # The lifting layer uses the ordinary spatial action.
    lift = GroupConv(1, 1, lifting=True, dilation=16, rngs=nnx.Rngs(45))
    z = x[..., :1]
    a = convolve(lift, action(z, 1, axes=(1, 2)), 3, "physical").reshape(1, 8, 8, 8, 1)
    b = convolve(lift, z, 3, "physical").reshape(1, 8, 8, 8, 1)
    np.testing.assert_allclose(a, feature_action(b, 1), atol=2e-6, rtol=1e-5)
