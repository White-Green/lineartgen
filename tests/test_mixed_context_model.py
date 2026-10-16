import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import action
from mixed_context_model import TAG, add_context, infer, load_field, save_field
from physical_field_model import PhysicalGroupFieldCNN
from physical_field_model import load_field as old_load
from plausible_study import exact
from subpixel_field_model import field_forward, unpack
from test_group_field import cfg, random_head


def fixture(width=1, dilation=(1, 32)):
    config = cfg(width)
    old = PhysicalGroupFieldCNN(config.model, rngs=nnx.Rngs(41))
    random_head(old)
    new = add_context(old, config, dilation=dilation)
    source = jax.random.uniform(jax.random.PRNGKey(17), (1, 64, 64, 1)) * 2 - 1
    strength = jnp.full_like(source, 0.5).at[:, :16].set(0)
    return config, old, new, source, strength


def activate(model):
    second = model.context_block.second
    second.kernel[...] = jax.random.normal(jax.random.PRNGKey(28), second.kernel.shape) * 0.1


@pytest.mark.parametrize("dilation", [(1, 32)])
def test_transfer_preserves_all_scales_and_parameter_count(dilation):
    _, old, new, source, strength = fixture(dilation=dilation)
    state = nnx.to_pure_dict(nnx.state(new, nnx.Param))
    del state["context_block"]
    exact(nnx.to_pure_dict(nnx.state(old, nnx.Param)), state)
    forward = nnx.jit(field_forward)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(18), 6)
    for a, b in zip(forward(old, p), forward(new, p), strict=True):
        np.testing.assert_array_equal(a, b)
    _, old, new, _, _ = fixture(6, dilation)
    assert sum(x.size for x in jax.tree.leaves(nnx.state(old, nnx.Param))) == 16402
    assert sum(x.size for x in jax.tree.leaves(nnx.state(new, nnx.Param))) == 21718


def test_nonzero_context_preserves_d4_and_reaches_beyond_old_radius():
    _, old, new, source, strength = fixture()
    activate(new)
    insert = jnp.zeros((1, 32, 32, 1))

    @nnx.jit
    def call(m, guide, t, coarse):
        return unpack(m(guide, t, coarse, guide, jnp.ones_like(guide), return_network_output=True))

    expected = call(new, source, strength, insert)
    for g in range(8):
        actual = call(
            new, action(source, g, (1, 2)), action(strength, g, (1, 2)), action(insert, g, (1, 2))
        )
        np.testing.assert_allclose(actual, action(expected, g, (1, 2)), atol=3e-5, rtol=3e-5)
    # Output pixel (8, 8) cannot see (56, 8) in the old radius-43 feature network.
    changed = source.at[:, 56:60, 8:12].set(1)
    crop = (slice(None), slice(32, 36), slice(32, 36))
    np.testing.assert_array_equal(
        call(old, source, strength, insert)[crop], call(old, changed, strength, insert)[crop]
    )
    assert (
        float(jnp.max(jnp.abs(expected[crop] - call(new, changed, strength, insert)[crop]))) > 1e-8
    )


@pytest.mark.parametrize("dilation", [(1, 32)])
def test_zero_branch_has_learning_gradient_and_nonzero_roundtrip(tmp_path, dilation):
    config, _, new, source, strength = fixture(dilation=dilation)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(19), 1)
    gradient = nnx.jit(nnx.grad(lambda m: jnp.mean(field_forward(m, p)[0] ** 2)))
    grad = gradient(new)
    assert float(jnp.linalg.norm(grad.context_block.second.kernel)) > 0
    assert float(jnp.linalg.norm(grad.context_block.first.kernel)) == 0
    activate(new)
    grad = gradient(new)
    assert float(jnp.linalg.norm(grad.context_block.first.kernel)) > 0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    expected = infer(new, p)
    save_field(new, config, tmp_path / "model", provenance="test")
    restored, cc = load_field(tmp_path / "model")
    assert cc == config and restored.research_model == TAG
    assert (
        restored.context_block.first.dilation,
        restored.context_block.second.dilation,
    ) == dilation
    exact(nnx.state(new, nnx.Param), nnx.state(restored, nnx.Param))
    np.testing.assert_array_equal(expected, infer(restored, p))
    protected = np.asarray(strength) == 0
    np.testing.assert_array_equal(np.asarray(expected)[protected], np.asarray(source)[protected])
    with pytest.raises(AssertionError):
        old_load(tmp_path / "model")
