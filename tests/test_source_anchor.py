import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.diffusion import pyramid, sample_prepared
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference


def configuration(base="source"):
    return ExperimentConfig(
        model=ModelConfig(
            channels=(2, 4),
            convolutions_per_block=2,
            output_mode="residual",
            source_conditioning=True,
            residual_base=base,
            detail_channels=4,
        )
    )


def test_source_anchor_changes_only_physical_base_and_preserves_protected_gradient():
    c = configuration()
    source = create_model(c.model, rngs=nnx.Rngs(12))
    noisy = create_model(configuration("noisy").model, rngs=nnx.Rngs(12))
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(source, nnx.Param), nnx.state(noisy, nnx.Param)
    )
    x = jax.random.uniform(jax.random.PRNGKey(4), (1, 16, 16, 1), minval=-0.5, maxval=0.5)
    guide = jnp.full_like(x, 0.2)
    level = jnp.full_like(x, 0.4).at[:, :4].set(0)
    insert = jnp.zeros((1, 8, 8, 1))
    _, a = source(x, level, insert, guide, return_preclip=True)
    _, b = noisy(x, level, insert, guide, return_preclip=True)
    np.testing.assert_allclose(a - b, jnp.where(level > 0, guide - x, 0), atol=1e-6)
    np.testing.assert_array_equal(a[:, :4], x[:, :4])
    gradient = nnx.grad(lambda m: m(x, level, insert, guide, return_preclip=True)[1][:, :4].sum())(
        source
    )
    for leaf in jax.tree.leaves(gradient):
        np.testing.assert_array_equal(leaf, 0)
    gradient = nnx.grad(lambda m: m(x, level, insert, guide, return_preclip=True)[1][:, 4:].sum())(
        source
    )
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradient))


def test_zero_head_has_no_artificial_noise_and_roundtrips_recursive_sampler(tmp_path):
    c = configuration()
    m = create_model(c.model, rngs=nnx.Rngs(8))
    for block in [m.head, m.detail_head]:
        block.layers[-1].kernel[...] = 0
        block.layers[-1].bias[...] = 0
    source = np.linspace(-0.8, 0.8, 32 * 32, dtype=np.float32).reshape(1, 32, 32, 1)
    level = np.full_like(source, 0.5)
    level[:, :8] = 0
    guide = pyramid(jnp.asarray(source), 3)
    save_inference(m, c, tmp_path / "model", provenance="source anchor contract test")
    restored, saved = load_inference(tmp_path / "model")
    assert saved.model.residual_base == "source"
    for seed in (7, 21):
        inputs = runtime_inputs(source, level, seed, 3)
        for steps in (1, 2, 5):
            actual = sample_prepared(m, *inputs, steps=steps, guides=guide)
            np.testing.assert_allclose(actual, source, atol=2e-7)
            np.testing.assert_array_equal(actual[:, :8], source[:, :8])
            np.testing.assert_array_equal(
                actual, sample_prepared(restored, *inputs, steps=steps, guides=guide)
            )


def test_source_anchor_rejects_incompatible_configs_and_old_default_is_noisy():
    assert ModelConfig().residual_base == "noisy"
    for updates in [
        dict(source_conditioning=False),
        dict(output_mode="direct"),
        dict(architecture="nested"),
        dict(progress_conditioning=True),
    ]:
        data = configuration().model.model_dump()
        data.update(updates)
        with pytest.raises(ValueError, match="Source residual base"):
            ModelConfig.model_validate(data)
