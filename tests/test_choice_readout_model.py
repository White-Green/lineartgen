import jax
import jax.numpy as jnp
import numpy as np
import pytest
from choice_readout_model import TAG, infer, load_field, save_field, select_readout
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import action
from mixed_context_model import MixedContextFieldCNN
from mixed_context_model import load_field as old_load
from plausible_study import exact
from subpixel_field_model import field_forward
from test_group_field import cfg, random_head


def fixture(width=1):
    config = cfg(width)
    old = MixedContextFieldCNN(config.model, rngs=nnx.Rngs(41))
    random_head(old)
    new = select_readout(old, config)
    source = jax.random.uniform(jax.random.PRNGKey(17), (1, 64, 64, 1)) * 2 - 1
    strength = jnp.full_like(source, 0.5).at[:, :16].set(0)
    return config, old, new, source, strength


class ManualFirst(nnx.Module):
    def __init__(self, linear):
        self.linear = linear

    def __call__(self, x):
        b, h, w, c = x.shape
        # Independent explicit branch: first orientation, including its16 subpixels.
        return self.linear(x.reshape(b, h, w, 8, c // 8)[..., 0, :])


def test_parameter_identity_and_all_scale_manual_readout():
    _, old, new, source, strength = fixture()
    exact(nnx.state(old, nnx.Param), nnx.state(new, nnx.Param))
    reference = nnx.clone(old)
    reference.head = ManualFirst(reference.head.linear)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(18), 6)
    forward = nnx.jit(field_forward)
    for a, b in zip(forward(reference, p), forward(new, p), strict=True):
        np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4)
    assert float(jnp.max(jnp.abs(forward(old, p)[0] - forward(new, p)[0]))) > 1e-4
    _, old, new, _, _ = fixture(6)
    for m in (old, new):
        assert sum(x.size for x in jax.tree.leaves(nnx.state(m, nnx.Param))) == 21718


def test_component_set_equivariance_through_all_recursion_levels():
    _, _, new, source, strength = fixture()
    key = jax.random.PRNGKey(19)
    p = prepare_inputs(source, source, strength, key, 6)
    forward = nnx.jit(field_forward)
    expected = forward(new, p)
    for g in range(8):
        transformed = nnx.clone(new)
        transformed.head.component = g
        s, t = action(source, g, (1, 2)), action(strength, g, (1, 2))
        actual = forward(transformed, prepare_inputs(s, s, t, key, 6))
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, action(b, g, (1, 2)), atol=3e-5, rtol=3e-5)


def test_gradient_roundtrip_and_protected_pixels(tmp_path):
    config, _, new, source, strength = fixture()
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(20), 3)
    grad = nnx.jit(nnx.grad(lambda m: jnp.mean(field_forward(m, p)[0] ** 2)))(new)
    assert float(jnp.linalg.norm(grad.head.linear.kernel)) > 0
    assert float(jnp.linalg.norm(grad.stem.kernel)) > 0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    expected = infer(new, p)
    save_field(new, config, tmp_path / "model", provenance="test")
    restored, cc = load_field(tmp_path / "model")
    assert cc == config and restored.research_model == TAG and restored.head.component == 0
    exact(nnx.state(new, nnx.Param), nnx.state(restored, nnx.Param))
    np.testing.assert_array_equal(expected, infer(restored, p))
    protected = np.asarray(strength) == 0
    np.testing.assert_array_equal(np.asarray(expected)[protected], np.asarray(source)[protected])
    with pytest.raises(AssertionError):
        old_load(tmp_path / "model")
