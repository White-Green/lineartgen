import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import action, feature_action
from learned_coarse_axis_model import TAG, add_coarse, coarse_average, infer, load_field, save_field
from plausible_study import exact
from pooled_axis_model import PooledAxisCNN
from pooled_axis_model import load_field as old_load
from pooled_context_model import pooled_cue
from subpixel_field_model import field_forward
from test_group_field import cfg, random_head


def fixture(width=1, mode="choice"):
    config = cfg(width)
    old = PooledAxisCNN(config.model, mode=mode, rngs=nnx.Rngs(41))
    random_head(old)
    old.head.residual.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(42), old.head.residual.kernel.shape) * 0.03
    )
    new = add_coarse(old, config)
    source = jax.random.uniform(jax.random.PRNGKey(17), (1, 128, 128, 1)) * 2 - 1
    strength = jnp.full_like(source, 0.5).at[:, :16].set(0)
    return config, old, new, source, strength


def test_zero_branch_preserves_nonzero_readout_all_scales_and_count():
    _, old, new, source, strength = fixture()
    state = nnx.to_pure_dict(nnx.state(new, nnx.Param))
    added = state.pop("coarse_branch")
    exact(nnx.to_pure_dict(nnx.state(old, nnx.Param)), state)
    assert all(np.count_nonzero(np.asarray(v)) == 0 for v in jax.tree.leaves(added["project"]))
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(18), 6)
    f = nnx.jit(field_forward)
    for a, b in zip(f(old, p), f(new, p), strict=True):
        np.testing.assert_array_equal(a, b)
    _, old, new, _, _ = fixture(6)
    assert sum(x.size for x in jax.tree.leaves(nnx.state(old, nnx.Param))) == 21890
    assert sum(x.size for x in jax.tree.leaves(nnx.state(new, nnx.Param))) == 22508


def test_coarse_average_branch_d4_and_information_beyond_mean():
    _, _, new, source, _ = fixture()
    v = jnp.arange(256, dtype=jnp.float32).reshape(1, 16, 16, 1)
    np.testing.assert_allclose(
        coarse_average(v, 2), np.asarray(v - 1).reshape(1, 4, 4, 4, 4, 1).mean((2, 4))
    )
    new.coarse_branch.project.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(28), new.coarse_branch.project.kernel.shape) * 0.1
    )

    @nnx.jit
    def call(branch, guide):
        out = branch(guide, 0)
        return out.reshape(*out.shape[:-1], 8, out.shape[-1] // 8)

    expected = call(new.coarse_branch, source)
    for g in range(8):
        actual = call(new.coarse_branch, action(source, g, (1, 2)))
        np.testing.assert_allclose(actual, feature_action(expected, g), atol=3e-5, rtol=3e-5)
    a = jnp.ones((1, 256, 256, 1)).at[:, 120:128, 88:96].set(-1)
    b = jnp.ones_like(a).at[:, 120:128, 128:136].set(-1)
    region = (slice(None), slice(120, 136), slice(120, 136))
    np.testing.assert_array_equal(pooled_cue(a, 0)[region], pooled_cue(b, 0)[region])
    assert (
        float(
            jnp.max(
                jnp.abs(call(new.coarse_branch, a)[region] - call(new.coarse_branch, b)[region])
            )
        )
        > 1e-8
    )


def test_projection_and_hidden_layers_receive_gradients():
    _, _, new, source, strength = fixture()
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(19), 1)
    gradient = nnx.jit(nnx.grad(lambda m: jnp.mean(field_forward(m, p)[0] ** 2)))
    grad = gradient(new)
    assert float(jnp.linalg.norm(grad.coarse_branch.project.kernel)) > 0
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad))
    new.coarse_branch.project.kernel[...] -= 0.01 * grad.coarse_branch.project.kernel
    new.coarse_branch.project.bias[...] -= 0.01 * grad.coarse_branch.project.bias
    grad = gradient(new)
    for name in ("lift", "first", "second"):
        assert float(jnp.linalg.norm(getattr(grad.coarse_branch, name).kernel)) > 0
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad))


@pytest.mark.parametrize("mode", ["mean", "choice"])
def test_nonzero_roundtrip_and_protection(tmp_path, mode):
    config, _, new, source, strength = fixture(mode=mode)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(19), 3)
    new.coarse_branch.project.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(28), new.coarse_branch.project.kernel.shape) * 0.1
    )
    expected = infer(new, p)
    save_field(new, config, tmp_path / "model", provenance="test")
    restored, cc = load_field(tmp_path / "model")
    assert cc == config and restored.research_model == TAG and restored.head.mode == mode
    exact(nnx.state(new, nnx.Param), nnx.state(restored, nnx.Param))
    np.testing.assert_array_equal(expected, infer(restored, p))
    mask = np.asarray(strength) == 0
    np.testing.assert_array_equal(np.asarray(expected)[mask], np.asarray(source)[mask])
    with pytest.raises(AssertionError):
        old_load(tmp_path / "model")
