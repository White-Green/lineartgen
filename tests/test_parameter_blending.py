import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ModelConfig
from lineartgen.diffusion import pyramid, recursive_forward
from lineartgen.models import create_model
from lineartgen.models.blending import blend_parameters, compatible_config


def models():
    config = ModelConfig(channels=(2, 4), convolutions_per_block=1)
    other = config.model_copy(update={"output_clip_gradient": "straight_through"})
    compatible_config(config, other)
    return [create_model(c, rngs=nnx.Rngs(seed)) for seed, c in enumerate([config, other])]


def test_endpoints_reproduce_parameters_and_recursive_outputs_exactly():
    a, b = models()
    inputs = pyramid(jax.random.normal(jax.random.PRNGKey(9), (1, 16, 16, 1)), 3)
    strengths = tuple(jnp.full_like(x, 0.4).at[:, :, :3, :].set(0) for x in inputs)
    for alpha, reference in [(0, a), (1, b)]:
        model = blend_parameters(a, b, alpha)
        jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(reference))
        jax.tree.map(
            np.testing.assert_array_equal,
            recursive_forward(model, inputs, strengths),
            recursive_forward(reference, inputs, strengths),
        )


def test_interpolation_keeps_one_model_and_does_not_mutate_endpoints():
    a, b = models()
    before = [jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(m)) for m in [a, b]]
    model = blend_parameters(a, b, 0.25)
    jax.tree.map(np.testing.assert_array_equal, *[before[0], nnx.state(a)])
    jax.tree.map(np.testing.assert_array_equal, *[before[1], nnx.state(b)])
    expected = jax.tree.map(
        lambda x, y: (0.75 * x.astype(np.float64) + 0.25 * y.astype(np.float64)).astype(x.dtype),
        *before,
    )
    jax.tree.map(np.testing.assert_array_equal, expected, nnx.state(model))
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model))) == sum(
        x.size for x in jax.tree.leaves(before[0])
    )


def test_rejects_unaligned_models_and_invalid_coefficients():
    a, b = models()
    for alpha in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            blend_parameters(a, b, alpha)
    with pytest.raises(ValueError, match="Forward"):
        compatible_config(ModelConfig(), ModelConfig(convolutions_per_block=1))
    with pytest.raises(ValueError):
        blend_parameters(
            a,
            create_model(ModelConfig(channels=(3, 5), convolutions_per_block=1), rngs=nnx.Rngs(1)),
            0.5,
        )
