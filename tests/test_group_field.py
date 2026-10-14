import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from group_field_model import (
    INVERSE,
    PRODUCT,
    GroupConv,
    GroupFieldCNN,
    GroupHead,
    action,
    feature_action,
)
from regional_field_objective import regional_gradient
from scale_style_balance_study import update
from subpixel_field_data import central_field
from subpixel_field_model import coverage, field_forward, infer, load_field, save_field
from test_full_source_normalization import config
from test_subpixel_field import fixture

from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state


def cfg(width=6):
    c, _ = config()
    return c.model_copy(
        update=dict(
            model=c.model.model_copy(
                update=dict(
                    channels=(width,),
                    local_dilations=(1, 4, 16),
                    pyramid_conditioning="fourier",
                )
            )
        )
    )


def random_head(model):
    model.head.linear.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(11), model.head.linear.kernel.shape) * 0.1
    )
    model.head.linear.bias[...] = jnp.linspace(-0.1, 0.1, 16)


def test_group_composition_and_regular_feature_action():
    x = jnp.arange(1 * 7 * 7 * 8 * 2).reshape(1, 7, 7, 8, 2)
    for g in range(8):
        np.testing.assert_array_equal(feature_action(feature_action(x, g), INVERSE[g]), x)
        for h in range(8):
            np.testing.assert_array_equal(
                feature_action(feature_action(x, h), g), feature_action(x, PRODUCT[g][h])
            )


def test_nonzero_lifting_and_group_convolutions_commute_with_d4():
    source = jax.random.normal(jax.random.PRNGKey(2), (1, 19, 19, 2))
    lift = GroupConv(2, 3, lifting=True, rngs=nnx.Rngs(3))
    base = lift(source).reshape(1, 19, 19, 8, 3)
    for g in range(8):
        actual = lift(action(source, g, (1, 2))).reshape(base.shape)
        np.testing.assert_allclose(actual, feature_action(base, g), atol=3e-5, rtol=3e-5)
    for dilation in (1, 4, 16):
        layer = GroupConv(3, 2, dilation=dilation, rngs=nnx.Rngs(5))
        reference = layer(base.reshape(1, 19, 19, 24)).reshape(1, 19, 19, 8, 2)
        for g in range(8):
            actual = layer(feature_action(base, g).reshape(1, 19, 19, 24))
            np.testing.assert_allclose(
                actual.reshape(reference.shape),
                feature_action(reference, g),
                atol=3e-5,
                rtol=3e-5,
            )


def test_subpixel_head_rotates_local_coordinates_with_feature_orientation():
    from subpixel_field_model import unpack

    x = jax.random.normal(jax.random.PRNGKey(9), (1, 7, 7, 8, 3))
    head = GroupHead(3, rngs=nnx.Rngs(10))
    head.linear.kernel[...] = jax.random.normal(jax.random.PRNGKey(11), (3, 16))
    head.linear.bias[...] = jnp.linspace(-1, 1, 16)
    reference = unpack(head(x.reshape(1, 7, 7, 24)))
    assert float(reference.std()) > 0.1
    for g in range(8):
        actual = unpack(head(feature_action(x, g).reshape(1, 7, 7, 24)))
        np.testing.assert_allclose(actual, action(reference, g, (1, 2)), atol=2e-6, rtol=2e-6)


def test_recursive_fields_equivariant_at_all_six_depths_with_nonzero_weights():
    model = GroupFieldCNN(cfg(2).model, rngs=nnx.Rngs(12))
    random_head(model)
    source = jax.random.uniform(jax.random.PRNGKey(13), (1, 64, 64, 1)) * 2 - 1
    t = jnp.full_like(source, 0.5).at[:, 7:28, 5:17].set(0)
    forward = nnx.jit(field_forward)
    for depth in range(1, 7):
        p = prepare_inputs(source, source, t, jax.random.PRNGKey(14), depth)
        fields = forward(model, p)
        assert float(fields[0].std()) > 0.001
        for g in range(8):
            transformed = jax.tree.map(lambda x, g=g: action(x, g, (1, 2)), p)
            actual = forward(model, transformed)
            for a, b in zip(actual, fields, strict=True):
                np.testing.assert_allclose(a, action(b, g, (1, 2)), atol=1e-4, rtol=1e-4)


def test_real_regional_update_checkpoint_export_and_protected_region(tmp_path):
    c = cfg()
    model = GroupFieldCNN(c.model, rngs=nnx.Rngs(8))
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 16402
    _, base = fixture()
    clean = coverage(central_field(base)[..., None])
    source = 1 - (1 - clean) * 0.4
    t = jnp.full_like(source, 0.5).at[:, :, :8].set(0)
    p = prepare_inputs(clean, source, t, jax.random.PRNGKey(3), 3)
    direction = jnp.array([[0.0, 1.0]])
    arm = make_arm(c, model)
    arm["gradient"] = regional_gradient()
    before = float(arm["gradient"](arm["model"], p, base, direction)[0]["loss"])
    assert update(arm, p, base, direction)["gradient_norm"] > 0
    after = float(arm["gradient"](arm["model"], p, base, direction)[0]["loss"])
    assert after < before
    assert np.any(np.asarray(arm["model"].head.linear.kernel[...]) != 0)
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, {"group": arm}, 1)
        restored = make_arm(c, model)
        restored["gradient"] = regional_gradient()
        assert lineart_state.restore(saved, {"group": restored}, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"group": arm}, 1),
        lineart_state.payload({"group": restored}, 1),
    )
    assert update(arm, p, base, direction) == update(restored, p, base, direction)
    save_field(arm["model"], c, tmp_path / "export", provenance="D4 group field test")
    loaded, cc = load_field(tmp_path / "export")
    assert cc == c and isinstance(loaded, GroupFieldCNN)
    np.testing.assert_array_equal(infer(loaded, p), infer(arm["model"], p))
    with pytest.raises(ValueError, match="Unsupported inference weight format"):
        load_inference(tmp_path / "export")
    pred = np.asarray(infer(loaded, p))
    np.testing.assert_array_equal(pred[np.asarray(t) == 0], np.asarray(source)[np.asarray(t) == 0])
