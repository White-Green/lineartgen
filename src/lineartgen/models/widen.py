"""Embed a UNet in a wider UNet while preserving its initial function.

Inherited output channels ignore new input channels at initialization. New
output channels retain random kernels, so paths into inherited channels can
start learning on the first update. Floating-point convolution order may change.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from .unet import UNet


def _conv(source, target, inputs, outputs):
    old = np.asarray(source.kernel[...])
    kernel = np.array(target.kernel[...])
    inputs, outputs = np.asarray(inputs), np.asarray(outputs)
    assert old.shape[:2] == kernel.shape[:2]
    assert old.shape[2:] == (len(inputs), len(outputs))
    assert len(set(inputs)) == len(inputs) and len(set(outputs)) == len(outputs)
    kernel[..., outputs] = 0
    kernel[:, :, inputs[:, None], outputs[None, :]] = old
    target.kernel[...] = jnp.asarray(kernel)
    if source.use_bias:
        bias = np.array(target.bias[...])
        bias[outputs] = np.asarray(source.bias[...])
        target.bias[...] = jnp.asarray(bias)


def _block(source, target, inputs, outputs):
    assert len(source.layers) == len(target.layers)
    for i, (old, new) in enumerate(zip(source.layers, target.layers, strict=True)):
        mapping = outputs if i == len(source.layers) - 1 else range(old.out_features)
        _conv(old, new, inputs, mapping)
        inputs = mapping


def widen_unet(model, old_config, new_config, *, seed):
    """Only channel widths may change; no layers, activations or inputs are added."""
    if not isinstance(model, UNet):
        raise ValueError("Widening requires a non-nested UNet")
    old, new = old_config.model_dump(), new_config.model_dump()
    for key in ("channels", "detail_channels", "final_decoder_channels", "head_hidden_channels"):
        old.pop(key)
        new.pop(key)
    if old != new:
        raise ValueError("Only channel widths may change")
    a, b = old_config.channels, new_config.channels
    if len(a) != len(b) or any(x > y for x, y in zip(a, b, strict=True)):
        raise ValueError("Every channel width must be preserved or increased")
    if bool(old_config.detail_channels) != bool(new_config.detail_channels):
        raise ValueError("The detail branch must already exist in both models")
    if old_config.detail_channels > new_config.detail_channels:
        raise ValueError("Detail width cannot shrink")
    if old_config.final_decoder_channels > new_config.final_decoder_channels:
        raise ValueError("Final decoder width cannot shrink")
    old_hidden = old_config.head_hidden_channels or (1 + old_config.final_decoder_channels)
    new_hidden = new_config.head_hidden_channels or (1 + new_config.final_decoder_channels)
    if old_hidden > new_hidden:
        raise ValueError("Head hidden width cannot shrink")
    widened = UNet(new_config, rngs=nnx.Rngs(seed))
    for i, (source, target) in enumerate(zip(model.encoders, widened.encoders, strict=True)):
        inputs = range(2) if i == 0 else list(range(a[i - 1]))
        if i == 1:
            inputs.append(b[0])  # The recursive insert follows the enlarged first encoder.
        _block(source, target, inputs, range(a[i]))
    old_skips, new_skips = (a[0] + 1, *a[1:]), (b[0] + 1, *b[1:])
    previous = 0
    for i, (source, target) in enumerate(zip(model.decoders, widened.decoders, strict=True)):
        stage = len(a) - i - 1
        skip = list(range(a[0])) + [b[0]] if stage == 0 else list(range(old_skips[stage]))
        inputs = skip + [new_skips[stage] + j for j in range(previous)]
        outputs = source.layers[-1].out_features
        _block(source, target, inputs, range(outputs))
        previous = outputs
    _block(model.head, widened.head, range(1 + previous), range(1))
    if old_config.detail_channels:
        inputs = list(range(3 + a[0])) + [3 + b[0] + j for j in range(previous)]
        _block(model.detail_head, widened.detail_head, inputs, range(1))
    for name in (
        "guide_encoder",
        "guide_head",
        "progress_encoder",
        "progress_head",
        "progress_detail",
    ):
        if hasattr(model, name):
            source, target = getattr(model, name), getattr(widened, name)
            _conv(source, target, range(source.in_features), range(source.out_features))
    return widened
