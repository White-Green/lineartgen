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


def configuration():
    return ExperimentConfig(
        model=ModelConfig(
            channels=(2, 4),
            convolutions_per_block=2,
            source_conditioning=True,
            output_mode="residual",
            residual_base="source",
            feature_input="source",
            detail_channels=4,
        )
    )


def test_source_features_ignore_sampled_noise_but_keep_protected_input_and_trainable_head():
    c = configuration()
    m = create_model(c.model, rngs=nnx.Rngs(2))
    x = jnp.full((1, 16, 16, 1), -0.4)
    guide = jnp.full_like(x, 0.3)
    t = jnp.full_like(x, 0.5).at[:, :4].set(0)
    insert = jnp.zeros((1, 8, 8, 1))

    def physical(image):
        return m(image, t, insert, guide, return_preclip=True)[1]

    a, b = physical(x), physical(-x)
    np.testing.assert_array_equal(a[:, 4:], b[:, 4:])
    np.testing.assert_array_equal(a[:, :4], x[:, :4])
    derivative = jax.grad(lambda image: physical(image).sum())(x)
    np.testing.assert_array_equal(derivative, jnp.where(t == 0, 1, 0))
    gradient = nnx.grad(
        lambda model: model(x, t, insert, guide, return_preclip=True)[1][:, 4:].sum()
    )(m)
    assert any(np.any(np.asarray(v) != 0) for v in jax.tree.leaves(gradient))
    # Rough image content still changes the prediction; this is not constant output.
    changed = m(x, t, insert, -guide, return_preclip=True)[1]
    assert np.max(np.abs(np.asarray(changed[:, 4:] - a[:, 4:]))) > 0.1


def test_source_signal_seed_invariance_and_serialized_multistep_mask(tmp_path):
    c = configuration()
    m = create_model(c.model, rngs=nnx.Rngs(12))
    source = np.linspace(-0.7, 0.7, 32 * 32, dtype=np.float32).reshape(1, 32, 32, 1)
    noise = np.full_like(source, 0.5)
    noise[:, :8] = 0
    guides = pyramid(jnp.asarray(source), 3)
    save_inference(m, c, tmp_path / "model", provenance="source signal contract")
    restored, saved = load_inference(tmp_path / "model")
    assert saved.model.feature_input == "source"
    for steps in (1, 2, 5):
        a = sample_prepared(m, *runtime_inputs(source, noise, 7, 3), steps=steps, guides=guides)
        b = sample_prepared(
            restored, *runtime_inputs(source, noise, 19, 3), steps=steps, guides=guides
        )
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, :8], source[:, :8])
    assert ModelConfig().feature_input == "noisy"
    with pytest.raises(ValueError, match="Source feature input"):
        ModelConfig(feature_input="source")
