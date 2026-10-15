import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import prepare_inputs
from flax import nnx
from physical_field_model import PhysicalGroupFieldCNN, infer, load_field, save_field
from plausible_study import exact
from test_group_field import cfg, random_head
from widen_physical_model import widen


def test_embedding_preserves_nonzero_film_and_recursive_prediction(tmp_path):
    config = cfg(2)
    old = PhysicalGroupFieldCNN(config.model, rngs=nnx.Rngs(42))
    random_head(old)
    for layer in [old.stem_modulation] + [b.modulation for b in old.blocks]:
        layer.linear.kernel[...] = (
            jax.random.normal(jax.random.PRNGKey(7), layer.linear.kernel.shape) * 0.05
        )
        layer.linear.bias[...] = jnp.linspace(-0.2, 0.2, 4)
    new, wc = widen(old, config, width=4)
    source = jax.random.uniform(jax.random.PRNGKey(1), (1, 32, 32, 1)) * 2 - 1
    p = prepare_inputs(source, source, jnp.full_like(source, 0.5), jax.random.PRNGKey(2), 3)
    a, b = np.asarray(infer(old, p)), np.asarray(infer(new, p))
    assert np.ptp(a) > 0
    np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4)
    for block in new.blocks:
        for layer in (block.first, block.second):
            np.testing.assert_array_equal(layer.kernel[...][..., 2:, :2], 0)
    save_field(new, wc, tmp_path / "model", provenance="test")
    restored, actual = load_field(tmp_path / "model")
    assert actual == wc
    exact(nnx.state(new, nnx.Param), nnx.state(restored, nnx.Param))
    np.testing.assert_array_equal(infer(restored, p), b)


def test_new_head_paths_can_receive_gradient_and_change_output():
    config = cfg(2)
    old = PhysicalGroupFieldCNN(config.model, rngs=nnx.Rngs(42))
    random_head(old)
    new, _ = widen(old, config, width=4)
    source = jax.random.normal(jax.random.PRNGKey(4), (1, 12, 12, 1))

    def loss(m):
        return m(
            source,
            jnp.ones_like(source),
            jnp.zeros((1, 6, 6, 1)),
            source,
            jnp.ones_like(source),
            return_network_output=True,
        ).mean()

    g = nnx.grad(loss)(new)
    assert np.isfinite(np.asarray(g.head.linear.kernel[...])).all()
    assert np.max(abs(np.asarray(g.head.linear.kernel[...][2:]))) > 0
    before = float(loss(new))
    new.head.linear.kernel[...] = (
        new.head.linear.kernel[...].at[2:].add(-0.01 * g.head.linear.kernel[...][2:])
    )
    assert float(loss(new)) < before
