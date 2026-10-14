import jax
import jax.numpy as jnp
import numpy as np
from opaque_geometry import compose, render
from scipy import ndimage

from lineartgen.diffusion import pyramid


def draw(segments, widths, rectangles=(), size=32):
    return render(
        jnp.asarray(segments, dtype=jnp.float32).reshape(-1, 2, 2),
        jnp.asarray(widths, dtype=jnp.float32).reshape(-1, 2),
        jnp.asarray(rectangles, dtype=jnp.float32).reshape(-1, 4),
        size=size,
    )


def test_opaque_interior_boundary_coverage_empty_and_fill():
    image = np.asarray(draw([[[4, 16], [28, 16]]], [[2, 2]]))
    np.testing.assert_array_equal(image[15:17, 6:26], -1)
    np.testing.assert_array_equal(image[:14], 1)
    assert np.isfinite(image).all() and image.min() >= -1 and image.max() <= 1
    np.testing.assert_array_equal(draw([], []), 1)
    fill = np.asarray(draw([], [], [[8, 8, 24, 24]]))
    expected = np.ones((32, 32, 1), np.float32)
    expected[8:24, 8:24] = -1
    np.testing.assert_array_equal(fill, expected)


def test_crossing_union_connected_parallel_and_intentional_gap_disconnected():
    cases = [
        ([[[4, 4], [28, 28]], [[4, 28], [28, 4]]], 1),
        ([[[4, 12], [28, 12]], [[4, 20], [28, 20]]], 2),
        ([[[4, 16], [13, 16]], [[19, 16], [28, 16]]], 2),
    ]
    for seg, count in cases:
        alpha = (1 - np.asarray(draw(seg, [[2, 2]] * len(seg)))[..., 0]) / 2
        assert ndimage.label(alpha > 0.25, np.ones((3, 3)))[1] == count
    crossing = np.asarray(draw(cases[0][0], [[2, 2], [2, 2]]))
    assert crossing[15:17, 15:17].max() == -1


def test_coarse_coverage_preserves_mass_and_protection():
    raw = draw([[[4, 16], [28, 16]]], [[2, 2]])[None]
    levels = pyramid(raw, 6)
    mass = float(((1 - raw) / 2).sum())
    for i, x in enumerate(levels):
        np.testing.assert_allclose(float(((1 - x) / 2).sum()) * 4**i, mass, atol=1e-5)
    assert 0 < float(((1 - levels[-1]) / 2).max()) < 1
    source = jnp.linspace(-1, 1, 32 * 32).reshape(1, 32, 32, 1)
    t = jnp.full_like(source, 0.5).at[:, :, :16].set(0)
    output = compose(raw, source, t)
    np.testing.assert_array_equal(output[:, :, :16], source[:, :, :16])
    grad = jax.grad(lambda x: compose(x, source, t).sum())(raw)
    np.testing.assert_array_equal(grad[:, :, :16], 0)


def test_geometry_gradient_matches_finite_difference_and_changes_position():
    # Narrow coverage transitions have kinks. Use float64 and shrinking differences
    # to test the actual local derivative instead of crossing a neighboring kink.
    with jax.enable_x64(True):
        segments = jnp.array([[[5.31, 15.17], [26.22, 17.43]]], dtype=jnp.float64)
        widths = jnp.array([[2.0, 2.0]], dtype=jnp.float64)
        rect = jnp.empty((0, 4), dtype=jnp.float64)
        ramp = jnp.arange(32, dtype=jnp.float64)[:, None, None] / 32

        def loss(offset):
            image = render(segments + jnp.array([0.0, offset]), widths, rect, size=32)
            return (((1 - image) / 2) * ramp).sum()

        g = float(jax.grad(loss)(0.0))
        estimates = [float((loss(eps) - loss(-eps)) / (2 * eps)) for eps in (1e-3, 1e-4, 1e-5)]
        assert abs(g) > 0.1
        assert abs(g - estimates[-1]) < abs(g - estimates[0])
        np.testing.assert_allclose(g, estimates[-1], atol=1e-5, rtol=1e-4)


def test_taper_is_geometric_width_and_union_does_not_darkness_double():
    segment = [[[4, 16], [28, 16]]]
    image = np.asarray(draw(segment, [[0.1, 3.0]]))
    alpha = (1 - image[..., 0]) / 2
    assert alpha[:, 7].sum() < alpha[:, 24].sum()
    one = draw(segment, [[2, 2]])
    repeated = draw(segment * 2, [[2, 2]] * 2)
    np.testing.assert_array_equal(one, repeated)
