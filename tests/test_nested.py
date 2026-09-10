import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.diffusion import (
    TrainingInputs,
    pyramid,
    recursive_forward,
    recursive_forward_all,
    sample,
)
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training.steps import make_steps


def config(depth=2):
    return ExperimentConfig.model_validate(
        {
            "model": {
                "architecture": "nested",
                "channels": [2, 3],
                "nested_channels": [[3, 5], [4, 7]],
                "convolutions_per_block": depth,
            }
        }
    )


def inputs(size, levels):
    image = jax.random.uniform(jax.random.PRNGKey(1), (1, size, size, 1), minval=-1, maxval=1)
    noise = jnp.full_like(image, 0.3).at[:, :, : size // 3].set(0)
    return pyramid(image, levels), pyramid(noise, levels)


@pytest.mark.parametrize("levels", [1, 3, 6])
def test_every_extracted_prefix_matches_shared_full_recursion(levels):
    model = create_model(config(depth=1).model, rngs=nnx.Rngs(42))
    images, noises = inputs(max(16, 4 * 2 ** (levels - 1)), levels)
    shared = nnx.jit(recursive_forward_all)(model, images, noises)
    counts = []
    for index in range(3):
        subnet = model.extract(index)
        actual = nnx.jit(recursive_forward)(subnet, images, noises)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5),
            actual,
            shared[index],
        )
        counts.append(sum(x.size for x in jax.tree.leaves(nnx.state(subnet, nnx.Param))))
    assert counts[0] < counts[1] < counts[2]


def test_large_parameters_and_insert_cannot_affect_small_outputs_or_gradients():
    model = create_model(config().model, rngs=nnx.Rngs(42))
    images, noises = inputs(16, 3)
    before = nnx.jit(recursive_forward_all)(model, images, noises)
    altered = nnx.clone(model)
    for block in (*altered.encoders, *altered.decoders):
        for layer in block.layers:
            layer.groups[2].kernel[...] += 0.25
            layer.groups[2].bias[...] += 0.3
    for layer in altered.heads[2].layers:
        layer.kernel[...] += 0.2
        layer.bias[...] += 0.2
    after = nnx.jit(recursive_forward_all)(altered, images, noises)
    for index in (0, 1):
        jax.tree.map(np.testing.assert_array_equal, before[index], after[index])
    assert not np.allclose(before[2][0], after[2][0])
    insert = jnp.zeros((1, 8, 8, 3))
    a = model(images[0], noises[0], insert)
    b = model(images[0], noises[0], insert.at[..., 1:].set(0.8))
    np.testing.assert_array_equal(a[..., :1], b[..., :1])

    derivative = nnx.jit(nnx.grad(lambda m: recursive_forward_all(m, images, noises)[0][0].mean()))(
        model
    )
    for side in ("encoders", "decoders"):
        for block in derivative[side].values():
            for layer in block["layers"].values():
                for index in (1, 2):
                    for value in jax.tree.leaves(layer["groups"][index]):
                        np.testing.assert_array_equal(value, np.zeros(value.shape))
    for index in (1, 2):
        for value in jax.tree.leaves(derivative["heads"][index]):
            np.testing.assert_array_equal(value, np.zeros(value.shape))
    assert any(np.any(np.asarray(x)) for x in jax.tree.leaves(derivative["heads"][0]))


def test_fused_and_block_execution_have_same_outputs_and_training_gradients():
    settings = config()
    fused = create_model(settings.model, rngs=nnx.Rngs(42))
    blocks = nnx.clone(fused)
    blocks.set_execution("blocks")
    images, noises = inputs(16, 2)
    prepared = TrainingInputs(images, noises, images)
    gradients, _, _ = make_steps(settings)
    values = [gradients(model, prepared) for model in (fused, blocks)]
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=2e-5, rtol=1e-4), *values)


def test_core_has_identical_initialization_and_is_an_ordinary_unet():
    settings = config()
    full = create_model(settings.model, rngs=nnx.Rngs(42))
    core = full.extract(0)
    independent = create_model(
        settings.model.model_copy(update={"nested_channels": ()}), rngs=nnx.Rngs(42)
    )
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(core, nnx.Param), nnx.state(independent, nnx.Param)
    )
    ordinary, _ = core.to_unet()
    images, noises = inputs(16, 3)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=2e-6),
        recursive_forward(core, images, noises),
        recursive_forward(ordinary, images, noises),
    )
    assert sum(x.size for x in jax.tree.leaves(nnx.state(core, nnx.Param))) == sum(
        x.size for x in jax.tree.leaves(nnx.state(ordinary, nnx.Param))
    )


def test_prefix_archive_is_autonomous_for_multiple_denoising_steps(tmp_path):
    settings = config(depth=1)
    full = create_model(settings.model, rngs=nnx.Rngs(42))
    images, noises = inputs(16, 2)
    for index in range(3):
        subnet = full.extract(index)
        subset_config = settings.model_copy(
            update={
                "model": settings.model.model_copy(
                    update={"nested_channels": settings.model.nested_channels[:index]}
                )
            }
        )
        destination = tmp_path / str(index)
        save_inference(subnet, subset_config, destination, provenance="test")
        restored, restored_config = load_inference(destination)
        assert restored_config == subset_config
        assert restored.num_subnets == index + 1
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(subnet, nnx.Param),
            nnx.state(restored, nnx.Param),
        )

        @nnx.jit
        def run(model):
            return sample(
                model, images[0], jax.random.PRNGKey(2), strength=noises[0], steps=3, levels=2
            )

        np.testing.assert_array_equal(run(subnet), run(restored))


class AnalyticNestedModel(nnx.Module):
    num_subnets = 2

    def __init__(self):
        self.small = nnx.Param(jnp.array(0.1))
        self.extension = nnx.Param(jnp.array(0.2))

    def __call__(self, image, noise, insert):
        # Make each size's previous-resolution prediction affect only itself/upwards.
        values = jnp.stack((self.small[...], self.small[...] + self.extension[...]))
        return jnp.broadcast_to(values + insert.mean((1, 2), keepdims=True), (*image.shape[:-1], 2))


@pytest.mark.parametrize("multiplier", [0.0, 0.5])
def test_subnet_losses_average_without_changing_scale_weights_or_recursive_gradients(multiplier):
    settings = ExperimentConfig.model_validate(
        {
            "diffusion": {"scale_loss_multiplier": multiplier},
            "loss": {"balance_tones": False, "edge_weight": 0, "contrast_weight": 0},
        }
    )
    images, _ = inputs(16, 2)
    targets = tuple(jnp.zeros_like(x) for x in images)
    prepared = TrainingInputs(targets, tuple(jnp.ones_like(x) for x in images), targets)
    metrics, grads = make_steps(settings)[0](AnalyticNestedModel(), prepared)
    # At the top scale S=2*s and L=2*(s+e); at the lower scale S=s, L=s+e.
    coefficient = (4 + multiplier) / (1 + multiplier)
    expected = coefficient * (0.1**2 + 0.3**2) / 2
    np.testing.assert_allclose(metrics["loss"], expected, rtol=1e-5)
    np.testing.assert_allclose(grads["small"][...], coefficient * 0.4, rtol=1e-5)
    np.testing.assert_allclose(grads["extension"][...], coefficient * 0.3, rtol=1e-5)
    np.testing.assert_allclose(
        metrics["loss"], (metrics["subnet_0/loss"] + metrics["subnet_1/loss"]) / 2
    )


@pytest.mark.parametrize("widths", [[[3]], [[2, 4]], [[3, 2]], [[3, 4], [3, 5]]])
def test_invalid_nested_widths_are_rejected(widths):
    with pytest.raises(ValueError):
        ModelConfig(architecture="nested", channels=(2, 3), nested_channels=widths)
