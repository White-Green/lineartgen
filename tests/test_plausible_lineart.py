import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import forward, prepare_inputs
from evaluate_plausible import specification
from flax import nnx
from local_recursive_fit import fresh_model
from plausible_data import CHOICES, ENDPOINTS, make_case
from plausible_objective import generator_gradient, geometry_loss, geometry_parts, prior_loss, views
from plausible_study import coordinates, make_arm, update

from lineartgen.config import load_config
from lineartgen.training import checkpoints, lineart_state


def fixture():
    c = load_config(
        "../jax-cue-preserving-capacity-20261012/runs/cue-capacity-20261012/configs/width16.json"
    )
    c = c.model_copy(update={"train": c.train.model_copy(update={"batch_size": 1})})
    source = jnp.ones((1, 32, 32, 1)).at[:, 5:27, 14:18].set(0.6)
    target = jnp.ones_like(source).at[:, 5:27, 15:17].set(-1)
    p = prepare_inputs(target, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(7), depth=2)
    return c, p, coordinates(p, 0)


def test_specification_accepts_alternatives_rejects_average_and_missing_content(tmp_path):
    specification(tmp_path)


def test_ambiguity_and_disjoint_shapes():
    for i in (0, 6, 12, 18, 24, 30, 36, 42, 48, 58):
        a, ar = make_case("train", i)
        b, br = make_case("probe", i)
        assert ar["variants_sha256"] != br["variants_sha256"]
        assert a["variants"].shape == (len(CHOICES), 256, 256, 1)
        assert np.mean(abs(a["variants"][ENDPOINTS[0]] - a["variants"][ENDPOINTS[1]])) > 1e-4
        # Changing the chosen teacher never changes this single stored source.
        before = a["source"].copy()
        for choice in (0, 13, 26):
            p = prepare_inputs(
                a["variants"][choice][None],
                a["source"][None],
                jnp.full((1, 256, 256, 1), 0.5),
                jax.random.PRNGKey(9),
                depth=3,
            )
            if choice == 0:
                initial = p
            else:
                jax.tree.map(np.testing.assert_array_equal, p[1:], initial[1:])
        np.testing.assert_array_equal(before, a["source"])


def test_candidate_gradients_use_no_designated_teacher_and_keep_critic_fixed():
    c, p, coords = fixture()
    arm = make_arm(c, fresh_model(c), "prior")
    state = jax.tree.map(lambda v: np.array(v, copy=True), nnx.state(arm["critic"]))
    fn = generator_gradient(adversarial=True)
    parts, grads = fn(arm["model"], p, coords, arm["critic"], jnp.asarray(0.05))
    other = p._replace(targets=tuple(-x for x in p.targets))
    jax.tree.map(
        np.testing.assert_array_equal,
        (parts, grads),
        fn(arm["model"], other, coords, arm["critic"], jnp.asarray(0.05)),
    )
    jax.tree.map(np.testing.assert_array_equal, state, nnx.state(arm["critic"]))
    _, zero = fn(arm["model"], p, coords, arm["critic"], jnp.asarray(0.0))
    assert any(
        np.any(np.asarray(a) != np.asarray(b))
        for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(zero), strict=True)
    )
    ordinary = generator_gradient(adversarial=False)(
        arm["model"], p, coords, None, jnp.asarray(0.0)
    )[1]
    jax.tree.map(np.testing.assert_array_equal, ordinary, zero)
    pred, raw = forward(arm["model"], p)
    stage = [
        geometry_parts(y, s, t, n)["geometry"]
        for n, (y, s, t) in enumerate(zip(pred, p.guides, p.noise_levels, strict=True))
    ]
    expected = (stage[0] + 0.5 * stage[1]) / 1.5
    np.testing.assert_allclose(geometry_loss(pred, raw, p)["geometry"], expected, rtol=2e-5)


def test_protected_area_has_no_prior_gradient_and_stage_scale_is_preserved():
    c, p, coords = fixture()
    arm = make_arm(c, fresh_model(c), "prior")
    p = p._replace(noise_levels=tuple(jnp.zeros_like(t) for t in p.noise_levels))
    pred, _ = forward(arm["model"], p._replace(noisy=p.guides))
    jax.tree.map(np.testing.assert_array_equal, pred, p.guides)
    gradient = jax.grad(lambda values: prior_loss(arm["critic"], values, p, coords))(pred)
    assert all(np.all(np.asarray(v) == 0) for v in jax.tree.leaves(gradient))
    local, whole = views(p.guides[0], p.noise_levels[0], coords[0], 3)
    assert local[2] == whole[2] == 3 / 5
    assert not np.any(np.asarray(local[1])) and not np.any(np.asarray(whole[1]))


def test_real_generator_critic_updates_and_full_state_restore(tmp_path):
    c, p, coords = fixture()
    arm = make_arm(c, fresh_model(c), "prior")
    a = update(arm, "prior", p, coords, 0)
    assert a["gradient_norm"] > 0 and a["critic_gradient_norm"] > 0 and a["critic_r1"] >= 0
    with checkpoints.manager(tmp_path / "saved") as manager:
        lineart_state.save(manager, {"prior": arm}, 1)
        restored = make_arm(c, fresh_model(c), "prior")
        assert lineart_state.restore(manager, {"prior": restored}) == 1
    assert update(arm, "prior", p, coords, 1) == update(restored, "prior", p, coords, 1)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"prior": arm}, 2),
        lineart_state.payload({"prior": restored}, 2),
    )
