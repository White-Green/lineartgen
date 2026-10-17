import jax
import jax.numpy as jnp
import numpy as np
from ambiguous_shape_data import render
from consistent_axis_data import records
from consistent_axis_objective import candidate_mask, field_parts, preferred_original
from quartic_field_objective import field_parts as previous_parts


def test_global_direction_and_partial_guard():
    normals = jnp.array([[1.0, 0.0], [0.0, -1.0], [1.0, -1.0], [1.0, 1.0]])
    assert np.asarray(preferred_original(normals)).tolist() == [False, True, True, False]
    t = jnp.ones((4, 2, 2, 1)) * 0.5
    a = np.asarray(candidate_mask(normals, t, "consistent"))
    assert (a.sum(1) == 9).all()
    assert a[:, 0].tolist() == [False, True, True, False]
    assert np.asarray(candidate_mask(normals, t, "winner")).all()
    t = t.at[0, 0, 0, 0].set(0)
    for policy in ("winner", "consistent"):
        mask = np.asarray(candidate_mask(normals, t, policy))[0]
        assert mask[:9].all() and not mask[9:].any()


def test_same_source_family_swap_keeps_preferred_teacher():
    for angle in (0, 30, 45, 60, 90, -45):
        r = records("train")[4] | dict(angle=float(np.deg2rad(angle)), center=[128.0, 128.0])
        q = r | dict(width=r["length"], length=r["width"], angle=r["angle"] + np.pi / 2)
        a, b = render(r), render(q)
        np.testing.assert_allclose(a["source"], b["source"], atol=1e-7, rtol=0)
        ca = a["clean"] if bool(preferred_original(a["direction"])) else a["axis_clean"]
        cb = b["clean"] if bool(preferred_original(b["direction"])) else b["axis_clean"]
        np.testing.assert_array_equal(ca, cb)


def toy():
    rng = np.random.default_rng(192)
    ff = tuple(jnp.asarray(rng.normal(0, 0.4, (2, s, s, 1)).astype(np.float32)) for s in (8, 4))
    tt = tuple(jnp.asarray(rng.normal(0, 0.8, (2, 18, s, s, 1)).astype(np.float32)) for s in (8, 4))
    strengths = tuple(jnp.ones((2, s // 4, s // 4, 1)) * 0.5 for s in (8, 4))
    return ff, tt, strengths


def test_unrestricted_cost_and_gradient_match_previous():
    ff, tt, t = toy()
    a = previous_parts(ff, tt, t)
    b = field_parts(ff, tt, t, candidate_mask=jnp.ones((2, 18), bool))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    ga = jax.grad(lambda f: previous_parts(f, tt, t)["loss"])(ff)
    gb = jax.grad(lambda f: field_parts(f, tt, t, candidate_mask=jnp.ones((2, 18), bool))["loss"])(
        ff
    )
    for x, y in zip(ga, gb, strict=True):
        np.testing.assert_array_equal(x, y)


def test_mask_selects_complete_candidate_after_common_costs():
    ff, tt, t = toy()
    # Restrict each image to a different half. Brute-force one complete candidate
    # at a time uses the SAME target tensor and normalization, across both stages.
    mask = jnp.array([[True] * 9 + [False] * 9, [False] * 9 + [True] * 9])
    chosen = field_parts(ff, tt, t, candidate_mask=mask)["loss"]
    expected = []
    for i in range(2):
        costs = []
        for k in range(18):
            if bool(mask[i, k]):
                one = jnp.arange(18)[None, :] == k
                costs.append(
                    float(
                        field_parts(
                            tuple(f[i : i + 1] for f in ff),
                            tuple(v[i : i + 1] for v in tt),
                            tuple(v[i : i + 1] for v in t),
                            candidate_mask=one,
                        )["loss"]
                    )
                )
        expected.append(min(costs))
    np.testing.assert_allclose(chosen, np.mean(expected), atol=1e-7, rtol=1e-6)
    assert float(field_parts(ff, tt, t, candidate_mask=mask)["selected_alternate_fraction"]) == 0.5
