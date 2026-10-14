import jax
import jax.numpy as jnp
import numpy as np
from contour_coverage import contour_image, triangle_fraction
from scipy import ndimage
from subpixel_field_data import central_field, geometry_field
from subpixel_field_model import coverage


def test_triangle_exact_half_one_two_vertices_and_scale_invariance():
    values = jnp.array(
        [[-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [-1.0, 0.0, 1.0]]
    )
    expected = jnp.array([0.25, 0.75, 0.0, 1.0, 0.5])
    for scale in (0.01, 1.0, 100.0):
        np.testing.assert_allclose(triangle_fraction(*(values * scale).T), expected, atol=1e-7)

    def f(x):
        return triangle_fraction(x, 1.3, 2.1)

    g = float(jax.grad(f)(-0.7))
    fd = float((f(-0.699) - f(-0.701)) / 0.002)
    np.testing.assert_allclose(g, fd, atol=2e-5, rtol=1e-3)


def test_constant_fields_are_opaque_even_arbitrarily_close_to_zero():
    for value in (1e-6, 0.01, 0.1, 8.0, -1e-6, -0.01, -0.1, -8.0):
        field = jnp.full((1, 64, 64, 1), value)
        np.testing.assert_array_equal(contour_image(field), 1 if value > 0 else -1)
    assert -1 < float(coverage(jnp.full((1, 64, 64, 1), 0.01)).mean()) < 1


def test_half_plane_exact_area_not_field_magnitude_and_geometry_topology():
    x = (jnp.arange(128) + 0.5) / 4
    f = jnp.broadcast_to((x - 16.3)[None, None, :, None], (1, 128, 128, 1))
    actual = np.asarray(contour_image(f))[0, :, :, 0]
    expected = np.ones((32, 32), np.float32)
    expected[:, :16] = -1
    expected[:, 16] = 0.4
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_allclose(contour_image(f * 0.1), contour_image(f * 10), atol=2e-6)
    cases = [
        ([[[4.0, 4.0], [28.0, 28.0]], [[4.0, 28.0], [28.0, 4.0]]], 1),
        ([[[4.0, 12.0], [28.0, 12.0]], [[4.0, 20.0], [28.0, 20.0]]], 2),
        ([[[4.0, 16.0], [13.0, 16.0]], [[19.0, 16.0], [28.0, 16.0]]], 2),
    ]
    for seg, count in cases:
        field = geometry_field(jnp.asarray(seg), jnp.full((2, 2), 2.0), jnp.empty((0, 4)), size=32)[
            None
        ]
        alpha = (1 - np.asarray(contour_image(central_field(field)[..., None]))[0, :, :, 0]) / 2
        assert ndimage.label(alpha > 0.25, np.ones((3, 3)))[1] == count
