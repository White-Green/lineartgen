import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import action
from mixed_context_model import MixedContextFieldCNN
from plausible_study import exact
from pooled_context_model import TAG, add_pool, infer, load_field, pooled_cue, save_field
from subpixel_field_model import field_forward, unpack
from test_group_field import cfg, random_head


def fixture(width=1):
    config = cfg(width)
    old = MixedContextFieldCNN(config.model, rngs=nnx.Rngs(41))
    random_head(old)
    new = add_pool(old, config)
    source = jax.random.uniform(jax.random.PRNGKey(17), (1, 128, 128, 1)) * 2 - 1
    strength = jnp.full_like(source, 0.5).at[:, :16].set(0)
    return config, old, new, source, strength


def test_zero_pool_preserves_all_scales_and_count():
    _, old, new, source, strength = fixture()
    state = nnx.to_pure_dict(nnx.state(new, nnx.Param))
    del state["pooled_stem"]
    exact(nnx.to_pure_dict(nnx.state(old, nnx.Param)), state)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(18), 6)
    forward = nnx.jit(field_forward)
    for a, b in zip(forward(old, p), forward(new, p), strict=True):
        np.testing.assert_array_equal(a, b)
    _, old, new, _, _ = fixture(6)
    assert sum(x.size for x in jax.tree.leaves(nnx.state(old, nnx.Param))) == 21718
    assert sum(x.size for x in jax.tree.leaves(nnx.state(new, nnx.Param))) == 21778


def test_pool_definition_d4_and_distant_influence():
    _, old, new, source, strength = fixture()
    # At level5 the physical radius is2: compare to a direct square sum.
    sample = jnp.arange(49, dtype=jnp.float32).reshape(1, 7, 7, 1) / 50
    direct = (
        jax.lax.reduce_window(sample - 1, 0.0, jax.lax.add, (1, 5, 5, 1), (1, 1, 1, 1), "SAME") / 25
    )
    np.testing.assert_allclose(pooled_cue(sample, 5), direct, atol=1e-7, rtol=1e-6)
    new.pooled_stem.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(28), new.pooled_stem.kernel.shape) * 0.1
    )
    insert = jnp.zeros((1, 64, 64, 1))

    @nnx.jit
    def call(m, guide, t, coarse):
        return unpack(m(guide, t, coarse, guide, jnp.ones_like(guide), return_network_output=True))

    expected = call(new, source, strength, insert)
    for g in range(8):
        actual = call(
            new, action(source, g, (1, 2)), action(strength, g, (1, 2)), action(insert, g, (1, 2))
        )
        np.testing.assert_allclose(actual, action(expected, g, (1, 2)), atol=3e-5, rtol=3e-5)
    # The old mixed model's maximum radius76 cannot see this displacement92.
    changed = source.at[:, 100:104, 8:12].set(1)
    crop = (slice(None), slice(32, 36), slice(32, 36))
    np.testing.assert_array_equal(
        call(old, source, strength, insert)[crop], call(old, changed, strength, insert)[crop]
    )
    assert (
        float(jnp.max(jnp.abs(expected[crop] - call(new, changed, strength, insert)[crop]))) > 1e-8
    )


def test_gradient_and_nonzero_roundtrip(tmp_path):
    config, _, new, source, strength = fixture()
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(19), 1)
    grad = nnx.jit(nnx.grad(lambda m: jnp.mean(field_forward(m, p)[0] ** 2)))(new)
    assert float(jnp.linalg.norm(grad.pooled_stem.kernel)) > 0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    new.pooled_stem.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(28), new.pooled_stem.kernel.shape) * 0.1
    )
    expected = infer(new, p)
    save_field(new, config, tmp_path / "model", provenance="test")
    restored, cc = load_field(tmp_path / "model")
    assert cc == config and restored.research_model == TAG
    exact(nnx.state(new, nnx.Param), nnx.state(restored, nnx.Param))
    np.testing.assert_array_equal(expected, infer(restored, p))
    protected = np.asarray(strength) == 0
    np.testing.assert_array_equal(np.asarray(expected)[protected], np.asarray(source)[protected])
