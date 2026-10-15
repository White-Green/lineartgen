"""Physical center alignment and D4 symmetry of the new transfer operator."""

import jax.numpy as jnp
import numpy as np
from field_transfer_model import pool_subpixels, transfer
from group_field_model import action


def pack(field):
    b, h, w, c = field.shape
    assert c == 1
    return (
        field.reshape(b, h // 4, 4, w // 4, 4)
        .transpose(0, 1, 3, 2, 4)
        .reshape(b, h // 4, w // 4, 16)
    )


def test_affine_signed_distance_in_finer_pixel_units():
    # Oblique plane normal has unit norm; values are /8 in coarse pixel units.
    y, x = np.meshgrid((np.arange(8) + 0.5) / 4, (np.arange(12) + 0.5) / 4, indexing="ij")
    distance = (0.6 * x + 0.8 * y - 1.2).astype(np.float32)
    actual = transfer(jnp.asarray(pack(distance[None, ..., None] / 8)), "subpixel_scaled")
    fy, fx = np.meshgrid((np.arange(4) + 0.5) / 2, (np.arange(6) + 0.5) / 2, indexing="ij")
    expected = 2 * (0.6 * fx + 0.8 * fy - 1.2) / 8
    np.testing.assert_allclose(actual[0, ..., 0], expected, atol=1e-7, rtol=0)


def test_pooling_commutes_with_all_d4_actions():
    field = jnp.asarray(
        np.random.default_rng(2026101504).normal(size=(1, 16, 24, 1)), dtype=jnp.float32
    )
    for g in range(8):
        actual = pool_subpixels(action(field, g, axes=(1, 2)))
        expected = action(pool_subpixels(field), g, axes=(1, 2))
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
