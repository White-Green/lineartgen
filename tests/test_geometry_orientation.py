import jax.numpy as jnp
import numpy as np
from geometry_orientation_study import transform

from lineartgen.diffusion import pyramid


def test_d4_inverse_and_pyramid_commutation():
    x = jnp.arange(2 * 32 * 32 * 3, dtype=jnp.float32).reshape(2, 32, 32, 3)
    for mirror in (False, True):
        for turns in range(4):
            y = transform(x, turns, mirror)
            np.testing.assert_array_equal(transform(y, turns, mirror, inverse=True), x)
            for a, b in zip(pyramid(x, 5), pyramid(y, 5), strict=True):
                np.testing.assert_array_equal(transform(a, turns, mirror), b)
