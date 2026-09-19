import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig, ModelConfig
from lineartgen.diffusion import pyramid, recursive_forward_all
from lineartgen.models import create_model
from lineartgen.models.nested import CausalConv
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training.comparison import manifest_configs


@pytest.mark.parametrize(
    "skip,dilation", [("post_pool", 1), ("pre_pool", 1), ("post_pool", 2), ("pre_pool", 2)]
)
def test_architecture_prefix_independence_and_serialization(tmp_path, skip, dilation):
    config = ExperimentConfig(
        model=ModelConfig(
            architecture="nested",
            channels=(2, 3),
            nested_channels=((3, 5),),
            convolutions_per_block=1,
            output_mode="residual",
            skip_mode=skip,
            bottleneck_dilation=dilation,
        )
    )
    model = create_model(config.model, rngs=nnx.Rngs(42))
    image = jax.random.uniform(jax.random.PRNGKey(3), (1, 128, 128, 1), minval=-1, maxval=1)
    noise = jnp.full_like(image, 0.4).at[:, :, :32].set(0)
    forward = nnx.jit(recursive_forward_all)
    for levels in (1, 3, 6):
        inputs = pyramid(image, levels), pyramid(noise, levels)
        result = forward(model, *inputs)
        core = model.extract(0)
        ordinary, settings = core.to_unet()
        assert settings.skip_mode == skip and settings.bottleneck_dilation == dilation
        for candidate in (core, ordinary):
            jax.tree.map(
                lambda a, b: np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5),
                result[0],
                forward(candidate, *inputs)[0],
            )
    altered = nnx.clone(model)
    for block in (*altered.encoders, *altered.decoders):
        for layer in block.layers:
            layer.groups[1].kernel[...] += 0.4
    changed = forward(altered, *inputs)
    jax.tree.map(np.testing.assert_array_equal, result[0], changed[0])
    gradient = nnx.jit(nnx.grad(lambda m: forward(m, *inputs)[0][0].mean()))(model)
    for side in ("encoders", "decoders"):
        for block in gradient[side].values():
            for layer in block["layers"].values():
                for leaf in jax.tree.leaves(layer["groups"][1]):
                    np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))
    path = tmp_path / "model"
    save_inference(model, config, path, provenance="architecture roundtrip")
    restored, saved = load_inference(path)
    assert saved == config
    jax.tree.map(np.testing.assert_array_equal, result, forward(restored, *inputs))
    blocks = nnx.clone(model)
    blocks.set_execution("blocks")
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5),
        result,
        forward(blocks, *inputs),
    )


def test_pre_pool_skip_preserves_positions_hidden_by_max_pool():
    config = ModelConfig(channels=(1, 1), convolutions_per_block=1)
    post = create_model(config, rngs=nnx.Rngs(0))
    pre = create_model(config.model_copy(update={"skip_mode": "pre_pool"}), rngs=nnx.Rngs(0))
    for model in (post, pre):
        # Only the learned skip path reaches the output; the raw-image head input is zeroed.
        for block in (*model.encoders, *model.decoders, model.head):
            conv = block.layers[0]
            conv.kernel[...] = (
                jnp.zeros_like(conv.kernel).at[1, 1, 1 if block is model.head else 0, 0].set(1)
            )
            conv.bias[...] = jnp.zeros_like(conv.bias)
    a = jnp.tile(jnp.array([[0.2, 0.8], [0.8, 0.2]]), (8, 8))[None, ..., None]
    b = jnp.where(a == 0.2, 0.8, 0.2)
    np.testing.assert_array_equal(nnx.max_pool(a, (2, 2), (2, 2)), nnx.max_pool(b, (2, 2), (2, 2)))
    masks, insert = jnp.full_like(a, 0.5), jnp.zeros((1, 8, 8, 1))
    np.testing.assert_array_equal(post(a, masks, insert), post(b, masks, insert))
    np.testing.assert_array_equal(pre(a, masks, insert), a)
    np.testing.assert_array_equal(pre(b, masks, insert), b)


def test_dilation_changes_spatial_support_without_changing_weights():
    layers = [
        CausalConv((1,), (1,), kernel_size=3, rngs=(nnx.Rngs(3),), execution="fused", dilation=d)
        for d in (1, 2)
    ]
    for layer in layers:
        layer.groups[0].kernel[...] = jnp.ones_like(layer.groups[0].kernel)
        layer.groups[0].bias[...] = jnp.zeros_like(layer.groups[0].bias)
    pulse = jnp.zeros((1, 9, 9, 1)).at[0, 4, 4, 0].set(1)
    for layer, distance in zip(layers, (1, 2), strict=True):
        result = np.asarray(layer((pulse,))[0])[0, ..., 0]
        expected = np.zeros((9, 9))
        expected[np.ix_([4 - distance, 4, 4 + distance], [4 - distance, 4, 4 + distance])] = 1
        np.testing.assert_array_equal(result, expected)
        layer.execution = "blocks"
        np.testing.assert_array_equal(layer((pulse,))[0][0, ..., 0], expected)


def test_variant_manifest_rejects_non_architecture_changes(tmp_path):
    base = ExperimentConfig()
    path = tmp_path / "variants.json"
    record = base.model_dump(mode="json")
    path.write_text(json.dumps({"model": record}))
    assert manifest_configs(path, base)["model"] == base
    record["train"]["learning_rate"] *= 2
    path.write_text(json.dumps({"model": record}))
    with pytest.raises(ValueError, match="non-model"):
        manifest_configs(path, base)
