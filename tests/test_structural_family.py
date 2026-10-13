import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import make_arm
from flax import nnx
from structural_family_data import structural_region
from structural_family_objective import structural_costs, structural_gradient
from structural_family_study import update
from style_family_objective import style_gradient
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture
from test_source_family import source_config

from lineartgen.diffusion import pyramid
from lineartgen.models import create_model
from lineartgen.training import checkpoints, lineart_state


def test_region_covers_true_junction_and_endpoints_without_duplicate_weight():
    paths = [dict(points=[(64, 64), end]) for end in [(64, 8), (120, 64), (64, 120), (8, 64)]]
    region = structural_region(dict(path_geometry=paths), size=128, margin=0)
    assert region.dtype == np.uint8 and region.max() == 1
    assert region[64, 64, 0] == region[8, 64, 0] == region[64, 120, 0] == 1
    assert region[35, 64, 0] == 0
    np.testing.assert_array_equal(
        region, structural_region(dict(path_geometry=paths * 2), size=128, margin=0)
    )
    assert structural_region({}).sum() == 0
    assert (
        structural_region(dict(path_geometry=[dict(points=[(-100, -100), (-80, -80)])])).sum() == 0
    )


def example():
    a, b, v = candidates()
    region = jnp.zeros_like(a).at[:, 3:9].set(1).at[:, 23:29].set(1)
    return a, b, v, dict(normalizers=v, region=region)


def test_global_clean_choice_not_gray_or_hybrid_and_corrective_gradient():
    a, b, v, ann = example()
    t = jnp.full_like(a, 0.5)

    def cost(z):
        return structural_costs(pyramid(z, 2), v, pyramid(t, 2), ann)

    assert float(cost(a).min()) == float(cost(b).min()) == 0
    assert float(cost((a + b) / 2).min()) > 0
    assert float(cost(a.at[:, 16:].set(b[:, 16:])).min()) > 0
    np.testing.assert_allclose(
        cost(a), structural_costs(pyramid(a, 2), v[:, ::-1], pyramid(t, 2), ann)[:, ::-1]
    )
    broken = a.at[:, 4:7, 10:12].set(1)
    g = jax.grad(
        lambda z: structural_costs((z,), v[:, :1], (t,), ann | dict(normalizers=v[:, :1])).sum()
    )(broken)
    assert np.all(np.asarray(g)[:, 4:7, 10:12] > 0)
    assert np.all(np.asarray(g)[:, 12:20] == 0)
    wrong = a.at[:, 5, 5].set(-3)
    g = jax.grad(
        lambda z: structural_costs((z,), v[:, :1], (t,), ann | dict(normalizers=v[:, :1])).sum()
    )(wrong)
    assert float(g[0, 5, 5, 0]) < 0  # A wrong saturated dark region still receives correction.
    protected = t.at[:, :, :16].set(0)
    g = jax.grad(lambda z: structural_costs((z,), v, (protected,), ann).min())((a + b) / 2)
    assert np.all(np.asarray(g)[:, :, :16] == 0)
    assert float(structural_costs((a,), v, (t,), ann | dict(region=jnp.zeros_like(a))).sum()) == 0


def test_stage_weights_and_normalization_are_shared_across_candidates():
    a, b, v, ann = example()
    ts = pyramid(jnp.full_like(a, 0.5), 2)
    bs = pyramid(b, 2)
    fine = structural_costs((a,), v, (ts[0],), ann)
    vc = pyramid(v.reshape(2, 32, 32, 1), 2)[1].reshape(1, 2, 16, 16, 1)
    coarse = structural_costs(
        (bs[1],), vc, (ts[1],), dict(normalizers=vc, region=pyramid(ann["region"], 2)[1])
    )
    np.testing.assert_allclose(
        structural_costs((a, bs[1]), v, ts, ann), (fine + 0.5 * coarse) / 1.5
    )


def test_zero_coefficient_preserves_existing_objective_and_designated_target_unused():
    c, p, _ = fixture()
    m = create_model(source_config(c).model, rngs=nnx.Rngs(31))
    m.head.kernel[...] = jnp.full_like(m.head.kernel[...], 0.01)
    _, _, v, ann = example()
    ann |= annotation()
    parts, grads = structural_gradient(0)(m, p, v, ann)
    old, old_grad = style_gradient()(m, p, v, ann)
    assert parts["structure"] == 0
    jax.tree.map(np.testing.assert_array_equal, {k: parts[k] for k in old}, old)
    jax.tree.map(np.testing.assert_array_equal, grads, old_grad)
    fn = structural_gradient(2)
    result = fn(m, p, v, ann)
    other = p._replace(targets=tuple(-z for z in p.targets))
    jax.tree.map(np.testing.assert_array_equal, result, fn(m, other, v, ann))
    assert (
        float(
            sum(
                jnp.sum(abs(x - y))
                for x, y in zip(jax.tree.leaves(grads), jax.tree.leaves(result[1]), strict=True)
            )
        )
        > 0
    )


def test_real_update_full_restore_and_next_update(tmp_path):
    c, p, _ = fixture()
    c = source_config(c)
    m = create_model(c.model, rngs=nnx.Rngs(31))
    arms = {}
    _, _, v, ann = example()
    ann |= annotation()
    for name in ("control", "structure"):
        arms[name] = make_arm(c, m)
        arms[name]["gradient"] = structural_gradient(2 if name == "structure" else 0)
        stats = update(arms[name], p, v, ann)
        assert stats["gradient_norm"] > 0
        assert (
            abs(
                stats["loss"]
                - stats["family"]
                - 0.25 * stats["profile"]
                - 2 * stats["structure"]
                - 0.05 * stats["range"]
            )
            < 1e-6
        )
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, arms, 1)
        restored = {name: make_arm(c, m) for name in arms}
        for name in restored:
            restored[name]["gradient"] = structural_gradient(2 if name == "structure" else 0)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    for name in arms:
        assert update(arms[name], p, v, ann) == update(restored[name], p, v, ann)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
