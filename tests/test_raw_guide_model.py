import jax
import jax.numpy as jnp
import numpy as np
from context_dropout_objective import dropout_forward
from cue_capacity import prepare_inputs
from flax import nnx
from group_field_model import action, feature_action
from physical_field_model import PhysicalGroupFieldCNN, infer, load_field, save_field
from plausible_study import exact
from raw_guide_model import RawLift, add_raw_guide
from subpixel_field_model import field_forward
from test_group_field import cfg, random_head


def fixture():
    config = cfg(1)
    old = PhysicalGroupFieldCNN(config.model, rngs=nnx.Rngs(41))
    random_head(old)
    raw = add_raw_guide(old, config)
    source = jax.random.uniform(jax.random.PRNGKey(17), (1, 64, 64, 1)) * 2 - 1
    strength = jnp.full_like(source, 0.5).at[:, :16].set(0)
    return config, old, raw, source, strength


def test_initial_fields_all_scales_and_auxiliary_gradient():
    _, old, raw, source, strength = fixture()
    forward = nnx.jit(field_forward)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(18), 6)
    for a, b in zip(forward(old, p), forward(raw, p), strict=True):
        np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4)
    keep = jnp.ones((6, 1, 1, 1, 1)).at[2].set(0)
    for a, b in zip(dropout_forward(old, p, keep), dropout_forward(raw, p, keep), strict=True):
        np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4)
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(18), 1)
    grad = nnx.grad(lambda m: jnp.mean(field_forward(m, p)[0] ** 2))(raw)
    assert np.isfinite(np.asarray(grad.raw_stem.kernel)).all()
    assert float(jnp.linalg.norm(grad.raw_stem.kernel)) > 0


def test_raw_lift_equivariance():
    lift = RawLift(2)
    lift.kernel[...] = jax.random.normal(jax.random.PRNGKey(3), lift.kernel.shape)
    source = jax.random.normal(jax.random.PRNGKey(4), (1, 11, 11, 1))
    base = lift(source).reshape(1, 11, 11, 8, 2)
    for g in range(8):
        actual = lift(action(source, g, (1, 2))).reshape(base.shape)
        np.testing.assert_allclose(actual, feature_action(base, g), atol=3e-5, rtol=3e-5)


def test_nonzero_cue_serialization_and_noise_independence(tmp_path):
    config, _, raw, source, strength = fixture()
    raw.raw_stem.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(5), raw.raw_stem.kernel.shape) * 0.1
    )
    p = prepare_inputs(source, source, strength, jax.random.PRNGKey(6), 3)
    q = prepare_inputs(source, source, strength, jax.random.PRNGKey(7), 3)
    expected = infer(raw, p)
    np.testing.assert_array_equal(expected, infer(raw, q))
    protected = np.asarray(strength) == 0
    np.testing.assert_array_equal(np.asarray(expected)[protected], np.asarray(source)[protected])
    save_field(raw, config, tmp_path / "model", provenance="test")
    restored, cc = load_field(tmp_path / "model")
    assert cc == config and restored.uses_raw_guide
    exact(nnx.state(raw, nnx.Param), nnx.state(restored, nnx.Param))
    np.testing.assert_array_equal(expected, infer(restored, p))


def test_raw_channel_changes_fields_with_normalized_guide_held_fixed():
    _, old, raw, source, strength = fixture()
    raw.raw_stem.kernel[...] = (
        jax.random.normal(jax.random.PRNGKey(8), raw.raw_stem.kernel.shape) * 0.1
    )
    insert = jnp.zeros((1, 32, 32, 1))

    def call(m, raw_input):
        return m(
            raw_input, strength, insert, source, jnp.ones_like(source), return_network_output=True
        )

    np.testing.assert_array_equal(call(old, source), call(old, source * 0.25 + 0.75))
    assert float(jnp.max(jnp.abs(call(raw, source) - call(raw, source * 0.25 + 0.75)))) > 1e-5
