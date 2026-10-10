import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.config import SketchAugmentationConfig
from lineartgen.sketch import augment_sketch, displaced_strokes, gaussian_blur


def test_blend_breaks_the_fixed_width_and_topology_collisions_after_blur():
    key = jnp.asarray([1260038368, 1197037806], dtype=jnp.uint32)
    assert int(jax.random.randint(jax.random.fold_in(key, 1), (), 0, 3)) == 2
    for pair in range(2):
        a = jnp.zeros((1, 512, 512, 1))
        b = jnp.zeros_like(a)
        if pair == 0:
            a, b = a.at[:, 256].set(1), b.at[:, 254:259].set(1)
        else:
            a, b = a.at[:, jnp.array([254, 258])].set(1), b.at[:, 252:261].set(1)

        def source(image, radius, blend):
            rough = displaced_strokes(image, key, radius, 2, blend)
            return 1 - 2 * 0.3 * gaussian_blur(rough, jnp.array([1.5]), 2)

        old_a, old_b = source(a, 2, 1), source(b, 0, 1)
        np.testing.assert_array_equal(old_a, old_b)
        new_a, new_b = source(a, 2, 0.5), source(b, 0, 0.5)
        assert float(jnp.max(jnp.abs(new_a - new_b))) > 2 / 255
        assert not np.array_equal(np.rint((new_a + 1) * 127.5), np.rint((new_b + 1) * 127.5))
        np.testing.assert_array_equal(old_b, new_b)


@pytest.mark.parametrize("mode", ["none", "tone", "blur", "strokes", "mixed"])
def test_core_blend_preserves_other_families_enablement_and_radius_zero(mode):
    clean = jnp.ones((16, 32, 32, 1)).at[:, :, 14:16].set(-1)
    key = jax.random.PRNGKey(85)
    old = SketchAugmentationConfig(mode=mode, probability=0.8)
    new = old.model_copy(update={"dilation_blend": 0.5})
    a, b = map(np.asarray, [augment_sketch(clean, key, c) for c in [old, new]])
    enabled = np.asarray(jax.random.bernoulli(jax.random.fold_in(key, 0), 0.8, (16,)))
    family = np.asarray(jax.random.randint(jax.random.fold_in(key, 4), (16,), 0, 3))
    radius = np.asarray(
        jax.random.randint(jax.random.fold_in(jax.random.fold_in(key, 3), 1), (16,), 0, 3)
    )
    changes_allowed = enabled & (radius > 0)
    if mode == "mixed":
        changes_allowed &= family == 2
    elif mode != "strokes":
        changes_allowed[:] = False
    np.testing.assert_array_equal(a[~changes_allowed], b[~changes_allowed])
    if changes_allowed.any():
        assert np.any(a[changes_allowed] != b[changes_allowed])
    white = jnp.ones_like(clean)
    np.testing.assert_array_equal(augment_sketch(white, key, new), white)
    assert np.isfinite(b).all() and b.min() >= -1 and b.max() <= 1


def test_blend_config_bounds_and_zero_blend_matches_no_dilation():
    for value in [-0.01, 1.01]:
        with pytest.raises(ValueError):
            SketchAugmentationConfig(dilation_blend=value)
    assert SketchAugmentationConfig().dilation_blend == 1
    ink = jax.random.uniform(jax.random.PRNGKey(86), (4, 32, 32, 1))
    key = jax.random.PRNGKey(87)
    np.testing.assert_array_equal(
        displaced_strokes(ink, key, 2, 2, 0), displaced_strokes(ink, key, 0, 2)
    )
