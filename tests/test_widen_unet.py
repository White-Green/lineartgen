import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ModelConfig
from lineartgen.diffusion import pyramid, recursive_forward_with_preclip
from lineartgen.models import create_model
from lineartgen.models.widen import widen_unet


@pytest.mark.parametrize("kind", ["encoder", "output", "both"])
@pytest.mark.parametrize("skip_mode", ["post_pool", "pre_pool"])
def test_widen_preserves_recursive_function_and_new_paths_can_learn(skip_mode, kind):
    config = ModelConfig(
        channels=(2, 3),
        convolutions_per_block=2,
        detail_channels=3,
        output_mode="residual",
        source_conditioning=True,
        residual_base="source",
        feature_input="source",
        sampling_mode="one_pass",
        skip_mode=skip_mode,
    )
    model = create_model(config, rngs=nnx.Rngs(17))
    # Exercise learned guide and detail paths, not just zero-initialized additions.
    for name in ["guide_encoder", "guide_head"]:
        layer = getattr(model, name)
        layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(13), layer.kernel.shape) * 0.05
    layer = model.detail_head.layers[-1]
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(15), layer.kernel.shape) * 0.05
    changes = {}
    if kind in ("encoder", "both"):
        changes.update(channels=(4, 6), detail_channels=6)
    if kind in ("output", "both"):
        changes.update(final_decoder_channels=8, head_hidden_channels=12)
    wide_config = config.model_copy(update=changes)
    wide = widen_unet(model, config, wide_config, seed=29)
    branch = "head" if kind == "output" else "detail_head"
    index = len(getattr(wide, branch).layers) - 1
    before = np.asarray(getattr(wide, branch).layers[-1].kernel[...]).copy()
    guide = jax.random.uniform(jax.random.PRNGKey(1), (1, 32, 32, 1), minval=-1, maxval=1)
    strength = jnp.full_like(guide, 0.5).at[:, :8, :].set(0)
    for depth in [1, 2, 3, 4]:
        guides = pyramid(guide, depth)
        levels = pyramid(strength, depth)
        noisy = guides
        expected = recursive_forward_with_preclip(model, noisy, levels, guides=guides)
        actual = recursive_forward_with_preclip(wide, noisy, levels, guides=guides)
        jax.tree.map(
            lambda x, y: np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-6), expected, actual
        )
        np.testing.assert_array_equal(
            np.asarray(actual[0][0])[:, :8, :], np.asarray(guide)[:, :8, :]
        )

    def objective(m):
        prediction, _ = recursive_forward_with_preclip(
            m, pyramid(guide, 2), pyramid(strength, 2), guides=pyramid(guide, 2)
        )
        return ((prediction[0] - guide * 0.5) ** 2).mean()

    grads = nnx.grad(objective)(wide)
    extra = np.asarray(grads[branch]["layers"][index]["kernel"])[..., 3:, :]
    assert np.isfinite(extra).all() and np.any(extra != 0)
    np.testing.assert_array_equal(before[..., 3:, :], 0)


def test_widen_rejects_other_architecture_changes():
    config = ModelConfig(channels=(2, 3), detail_channels=3, source_conditioning=True)
    model = create_model(config, rngs=nnx.Rngs(1))
    with pytest.raises(ValueError, match="Only channel widths"):
        widen_unet(model, config, config.model_copy(update={"kernel_size": 5}), seed=1)
    with pytest.raises(ValueError, match="preserved or increased"):
        widen_unet(model, config, config.model_copy(update={"channels": (1, 3)}), seed=1)


@pytest.mark.parametrize(
    "changes",
    [
        dict(final_decoder_channels=1),
        dict(head_hidden_channels=0),
        dict(architecture="nested", final_decoder_channels=8),
        dict(architecture="nested", head_hidden_channels=8),
    ],
)
def test_output_width_validation(changes):
    with pytest.raises(ValueError):
        ModelConfig(**changes)


def test_widen_rejects_smaller_head_and_decoder():
    config = ModelConfig(channels=(2, 3), final_decoder_channels=8, head_hidden_channels=12)
    model = create_model(config, rngs=nnx.Rngs(1))
    for changes in [dict(final_decoder_channels=4), dict(head_hidden_channels=6)]:
        with pytest.raises(ValueError, match="cannot shrink"):
            widen_unet(model, config, config.model_copy(update=changes), seed=2)


def test_default_model_matches_frozen_R_implementation_exactly():
    import importlib.util
    from pathlib import Path

    path = Path("../jax-source-interval-l1-20261010/src/lineartgen/models/unet.py").resolve()
    spec = importlib.util.spec_from_file_location("lineartgen.models.frozen_R_unet", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    config = ModelConfig(channels=(2, 3), source_conditioning=True, detail_channels=4)
    old = legacy.UNet(config, rngs=nnx.Rngs(17))
    new = create_model(config, rngs=nnx.Rngs(17))
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(old, nnx.Param), nnx.state(new, nnx.Param)
    )
    x = jax.random.uniform(jax.random.PRNGKey(1), (1, 32, 32, 1), minval=-1, maxval=1)
    t = jnp.ones_like(x) * 0.5
    insert = jnp.zeros((1, 16, 16, 1))
    np.testing.assert_array_equal(old(x, t, insert, guide=x), new(x, t, insert, guide=x))
