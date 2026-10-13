import jax
import jax.numpy as jnp
import numpy as np
import optax
from cue_capacity import make_arm
from flax import nnx
from local_recursive_fit import fresh_model
from plausible_objective import LinePrior, prior_loss, views
from style_family_objective import style_gradient
from style_prior_data import coordinates, structure_anchors
from style_prior_objective import style_prior_gradient
from style_prior_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture
from test_source_family import source_config

from lineartgen.training import checkpoints, lineart_state


def inputs():
    c, p, coords = fixture()
    _, _, v = candidates()
    ann = annotation() | dict(
        normalizers=v,
        coordinates=coords,
        critic_targets=p.targets,
        prior_weight=jnp.asarray(0.2),
    )
    return source_config(c), p, v, ann


def make():
    c, _, _, _ = inputs()
    m = fresh_model(c)
    m.head.kernel[...] = jnp.full_like(m.head.kernel[...], 0.01)
    arm = make_arm(c, m)
    arm["gradient"] = style_prior_gradient(adversarial=True)
    arm["critic"] = LinePrior(rngs=nnx.Rngs(3))
    arm["critic_optimizer"] = nnx.Optimizer(
        arm["critic"],
        optax.chain(optax.clip_by_global_norm(1), optax.adam(1e-4, b1=0.0, b2=0.99)),
        wrt=nnx.Param,
    )
    return arm


def test_zero_weight_matches_style_objective_and_prior_is_real_generator_gradient():
    _, p, v, ann = inputs()
    arm = make()
    state = jax.tree.map(lambda x: np.array(x, copy=True), nnx.state(arm["critic"]))
    plain, base_grad = style_gradient()(arm["model"], p, v, ann)
    zero, zero_grad = arm["gradient"](
        arm["model"], p, v, ann | dict(prior_weight=jnp.asarray(0.0)), arm["critic"]
    )
    jax.tree.map(np.testing.assert_array_equal, plain, {k: zero[k] for k in plain})
    jax.tree.map(np.testing.assert_array_equal, base_grad, zero_grad)
    parts, grad = arm["gradient"](arm["model"], p, v, ann, arm["critic"])
    assert parts["prior"] > 0
    assert any(
        np.any(np.asarray(a) != np.asarray(b))
        for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(zero_grad), strict=True)
    )
    jax.tree.map(np.testing.assert_array_equal, state, nnx.state(arm["critic"]))
    changed = p._replace(targets=tuple(-x for x in p.targets))
    jax.tree.map(
        np.testing.assert_array_equal,
        (parts, grad),
        arm["gradient"](arm["model"], changed, v, ann, arm["critic"]),
    )


def test_actual_generator_and_critic_update_checkpoint_continuation(tmp_path):
    _, p, v, ann = inputs()
    arm = make()
    old_g = np.array(arm["model"].head.kernel[...], copy=True)
    old_d = np.array(arm["critic"].head.kernel[...], copy=True)
    stats = update(arm, p, v, ann)
    assert stats["gradient_norm"] > 0 and stats["critic_gradient_norm"] > 0
    assert stats["critic_r1"] >= 0
    assert not np.array_equal(old_g, np.asarray(arm["model"].head.kernel[...]))
    assert not np.array_equal(old_d, np.asarray(arm["critic"].head.kernel[...]))
    with checkpoints.manager(tmp_path / "saved") as manager:
        lineart_state.save(manager, {"prior": arm}, 1)
        restored = make()
        assert lineart_state.restore(manager, {"prior": restored}) == 1
    assert update(arm, p, v, ann) == update(restored, p, v, ann)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"prior": arm}, 2),
        lineart_state.payload({"prior": restored}, 2),
    )


def test_structural_coordinates_geometry_order_and_protected_prior_exclusion():
    metadata = dict(
        path_geometry=[
            dict(points=[[0, 0], [100, 0]], widths=[2, 2]),
            dict(points=[[50, -50], [50, 50]], widths=[2, 2]),
        ]
    )
    a = structure_anchors(metadata)
    xy = a["anchors"][a["anchor_valid"]]
    assert len(xy) == 7
    np.testing.assert_array_equal(xy[[0, 2, 6]], [[32, 32], [132, 32], [82, 32]])
    assert not structure_anchors({})["anchor_valid"].any()
    _, p, _, ann = inputs()
    args = (p, a["anchors"][None], a["anchor_valid"][None], 7, 13)
    coords = coordinates(*args)
    jax.tree.map(np.testing.assert_array_equal, coords, coordinates(*args))
    for level, c in enumerate(coords):
        assert c.shape == (1, 8, 2)
        assert np.all(np.asarray(c) >= 0)
        assert np.all(np.asarray(c) <= max(0, p.guides[level].shape[1] - 64))
    protected = p._replace(noise_levels=tuple(jnp.zeros_like(t) for t in p.noise_levels))
    arm = make()
    grad = jax.grad(lambda xs: prior_loss(arm["critic"], xs, protected, coords))(p.guides)
    assert all(np.all(np.asarray(x) == 0) for x in grad)
    image = jnp.ones((1, 128, 128, 1))
    strength = jnp.ones_like(image).at[:, :, :64].set(0)
    local, whole = views(image, strength, jnp.array([[[0, 0], [0, 64]]]), 0)
    np.testing.assert_array_equal(local[1], [0, 1])
    np.testing.assert_array_equal(whole[1], [0])
