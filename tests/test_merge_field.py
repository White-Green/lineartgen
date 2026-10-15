import copy

import jax
import jax.numpy as jnp
import numpy as np
from merge_family_data import merge_allowed, merge_geometry
from merge_field_data import merged_metadata
from merge_field_objective import family_parts
from regional_field_objective import field_parts


def targets():
    x = jnp.arange(64, dtype=jnp.float32) / 4
    original = jnp.minimum(abs(x - 5), abs(x - 11)) - 1
    merged = abs(x - 8) - 1
    a, b = [jnp.broadcast_to(v[None, None, :, None], (1, 32, 64, 1)) for v in (original, merged)]
    return a, b, (jnp.ones((1, 8, 16, 1)),)


def test_whole_opaque_solutions_beat_intermediate_and_selection_is_global():
    a, b, t = targets()
    aa, bb = (a[:, None],), (b[:, None],)
    pa = family_parts((a,), aa, bb, t, jnp.array([True]))
    pb = family_parts((b,), aa, bb, t, jnp.array([True]))
    mid = family_parts(((a + b) / 2,), aa, bb, t, jnp.array([True]))
    assert float(pa["loss"]) < 1e-7 and float(pb["loss"]) < 1e-7
    assert float(mid["loss"]) > 0.01
    assert float(pa["merged_fraction"]) == 0 and float(pb["merged_fraction"]) == 1
    # Inconsistent stages cannot pick their own perfect family independently.
    p = family_parts((a, b), aa * 2, bb * 2, t * 2, jnp.array([True]))
    assert float(p["loss"]) > 0.01


def test_ineligible_loss_and_gradient_match_original():
    a, b, t = targets()
    ff = (a * 0.7,)
    aa = (a[:, None],)
    bb = (b[:, None],)
    result = jax.jit(family_parts)(ff, aa, bb, t, jnp.array([False]))
    expected = jax.jit(field_parts)(ff, aa, t)
    for k in expected:
        np.testing.assert_array_equal(result[k], expected[k])
    g1 = jax.jit(jax.grad(lambda x: family_parts((x,), aa, bb, t, jnp.array([False]))["loss"]))(
        ff[0]
    )
    g2 = jax.jit(jax.grad(lambda x: field_parts((x,), aa, t)["loss"]))(ff[0])
    np.testing.assert_array_equal(g1, g2)
    assert np.isfinite(g1).all()


def test_merge_requires_input_support_and_preserves_metadata():
    record = dict(
        kind="parallel",
        metadata=dict(
            paths=2,
            path_geometry=[
                dict(kind="bezier", points=[[48.0, 65.0], [144.0, 65.0]], widths=[2.0, 2.0]),
                dict(kind="bezier", points=[[48.0, 71.0], [144.0, 71.0]], widths=[2.0, 2.0]),
            ],
        ),
    )
    old = copy.deepcopy(record)
    g = merge_geometry(record)
    source = np.ones((256, 256, 1), np.float32)
    source[94:107, 75:182] = 0.2
    assert merge_allowed(source, g)
    source[98:103] = 1
    assert not merge_allowed(source, g)
    merged = merged_metadata(record)
    np.testing.assert_array_equal(merged["path_geometry"][0]["points"], [[48, 68], [144, 68]])
    assert record == old
