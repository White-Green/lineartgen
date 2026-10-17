import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import action
from plausible_study import exact
from pooled_axis_model import add_readout
from pooled_context_model import PooledContextFieldCNN
from subpixel_field_model import field_forward, infer
from test_group_field import cfg, random_head


def fixture():
    old = PooledContextFieldCNN(cfg(1).model, rngs=nnx.Rngs(41))
    random_head(old)
    old.pooled_stem.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(42), old.pooled_stem.kernel.shape) * 0.1
    )
    source = jax.random.uniform(jax.random.PRNGKey(17), (1, 64, 64, 1)) * 2 - 1
    strength = jnp.full_like(source, 0.5).at[:, :16].set(0)
    return old, source, strength


def test_initial_function_and_gradient():
    old, source, strength = fixture()
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(18), 6)
    forward = nnx.jit(field_forward)
    expected = forward(old, p)
    arms = {m: add_readout(old, cfg(1), mode=m) for m in ("mean", "choice")}
    exact(nnx.state(arms["mean"], nnx.Param), nnx.state(arms["choice"], nnx.Param))
    for new in arms.values():
        for a, b in zip(forward(new, p), expected, strict=True):
            np.testing.assert_array_equal(a, b)
        gradient = nnx.jit(nnx.grad(lambda m: jnp.mean(field_forward(m, p)[0] ** 2)))(new)
        assert float(jnp.linalg.norm(gradient.head.residual.kernel)) > 0
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
        output = np.asarray(infer(new, p))
        mask = np.asarray(strength) == 0
        np.testing.assert_array_equal(output[mask], np.asarray(source)[mask])
    large = PooledContextFieldCNN(cfg(6).model, rngs=nnx.Rngs(41))
    for m in ("mean", "choice"):
        assert (
            sum(
                x.size
                for x in jax.tree.leaves(nnx.state(add_readout(large, cfg(6), mode=m), nnx.Param))
            )
            == 21890
        )


def test_nonzero_component_set_through_recursive_feedback():
    old, source, strength = fixture()
    model = add_readout(old, cfg(1), mode="choice")
    model.head.residual.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(21), model.head.residual.kernel.shape) * 0.1
    )
    model.head.residual.bias[...] = jax.random.normal(jax.random.PRNGKey(22), (16,)) * 0.1
    key = jax.random.PRNGKey(19)
    forward = nnx.jit(field_forward)
    expected = forward(model, prepare_inputs(source, source, strength, key, 6))
    for g in range(8):
        transformed = nnx.clone(model)
        transformed.head.component = g
        s, t = action(source, g, (1, 2)), action(strength, g, (1, 2))
        actual = forward(transformed, prepare_inputs(s, s, t, key, 6))
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, action(b, g, (1, 2)), atol=3e-5, rtol=3e-5)
