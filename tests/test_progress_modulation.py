import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import endpoints, source_gradient
from diagnose_learnability import update
from flax import nnx
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import batch, velocity_inputs

from lineartgen.config import ModelConfig, load_config
from lineartgen.models import create_model
from lineartgen.models.blocks import ConvBlock
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.models.unet import progress_basis
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


def modulated(tmp_path):
    old, p, _ = fixture(tmp_path)
    c = load_config(tmp_path / "configs/velocity.json")
    c = c.model_copy(
        update={"model": c.model.model_copy(update={"progress_modulation": "fourier_film"})}
    )
    m = create_model(c.model, rngs=nnx.Rngs(42))
    nnx.update(m, nnx.state(old, nnx.Param))
    return old, m, c, p


def test_modulation_rejects_missing_progress_and_keeps_default():
    assert ModelConfig().progress_modulation == "none"
    with pytest.raises(ValueError, match="requires progress_conditioning"):
        ModelConfig(progress_modulation="fourier_film")


def test_basis_and_feature_gain_bias_before_relu():
    u = jnp.array([0.0, 0.5, 1.0])[None, None, :, None]
    basis = np.asarray(progress_basis(u))
    np.testing.assert_array_equal(basis[..., 0], 1 - np.asarray(u[..., 0]))
    assert basis.shape == (1, 1, 3, 9) and np.abs(basis).max() <= 1
    block = ConvBlock(1, 1, kernel_size=1, depth=1, rngs=nnx.Rngs(3))
    block.layers[0].kernel[...] = 1
    block.layers[0].bias[...] = 0
    x = jnp.array([-2.0, 1.0])[None, None, :, None]
    actual = block(
        x, first_bias=jnp.ones_like(x), first_modulation=(jnp.ones_like(x), jnp.full_like(x, -1))
    )
    np.testing.assert_array_equal(actual, jax.nn.relu((x + 1) * 2 - 1))


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_zero_modulation_preserves_both_anchor_functions_and_inherited_params(tmp_path, depth):
    old, m, _, _ = modulated(tmp_path)
    state = nnx.to_pure_dict(nnx.state(m, nnx.Param))
    extra = state.pop("progress_modulators")
    jax.tree.map(lambda x: np.testing.assert_array_equal(x, 0), extra)
    jax.tree.map(np.testing.assert_array_equal, state, nnx.to_pure_dict(nnx.state(old, nnx.Param)))
    p = velocity_inputs(batch(2, depth, count=2)[0])
    args = {k: p[k] for k in ("images", "levels", "guides", "progress")}
    for source_anchor in (False, True):
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, atol=2e-6, rtol=0),
            endpoints(m, **args, source_anchor=source_anchor),
            endpoints(old, **args, source_anchor=source_anchor),
        )


def test_modulation_receives_gradient_and_full_state_next_update_restores(tmp_path):
    _, m, c, p = modulated(tmp_path)
    arm = create_arm(c, m)
    arm["gradient"] = source_gradient
    _, grads = source_gradient(arm["model"], p)
    norms = [float(jnp.linalg.norm(x)) for x in jax.tree.leaves(grads["progress_modulators"])]
    assert np.isfinite(norms).all() and sum(v > 1e-8 for v in norms) >= 3
    stats = update(arm, p)
    assert stats["gradient_norm"] > 0
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, {"x": arm}, 1)
        restored = create_arm(c, m)
        restored["gradient"] = source_gradient
        assert lineart_state.restore(saved, {"x": restored}, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"x": arm}, 1),
        lineart_state.payload({"x": restored}, 1),
    )
    assert update(arm, p) == update(restored, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"x": arm}, 2),
        lineart_state.payload({"x": restored}, 2),
    )
    path = tmp_path / "export"
    save_inference(arm["model"], c, path, provenance="roundtrip")
    loaded, cfg = load_inference(path)
    assert cfg == c
    jax.tree.map(
        np.testing.assert_array_equal,
        nnx.state(arm["model"], nnx.Param),
        nnx.state(loaded, nnx.Param),
    )
    args = {k: p[k] for k in ("images", "levels", "guides", "progress")}
    jax.tree.map(
        np.testing.assert_array_equal,
        endpoints(loaded, **args, source_anchor=True),
        endpoints(arm["model"], **args, source_anchor=True),
    )
