"""A training-only distance head; its values never enter image recursion or inference."""

import jax.numpy as jnp
from flax import nnx

from lineartgen.models.blocks import ConvBlock

from .distance_supervision import distance_supervision
from .losses import image_loss


class AuxiliaryDistance(nnx.Module):
    def __init__(self, core, config, *, rngs):
        if not config.source_conditioning or not config.detail_channels:
            raise ValueError("Expected source conditioning and a full-resolution detail head")
        self.core = nnx.clone(core)
        self.auxiliary = ConvBlock(
            3 + config.channels[0] + config.final_decoder_channels,
            1,
            kernel_size=3,
            depth=3,
            hidden_channels=16,
            rngs=rngs,
        )
        self.auxiliary.layers[-1].kernel[...] = 0
        self.auxiliary.layers[-1].bias[...] = 0

    def __call__(self, *args, **kwargs):
        return self.core(*args, **kwargs)


def forward_with_auxiliary(model, prepared):
    smallest = prepared["noisy"][-1]
    insert = jnp.zeros(
        (smallest.shape[0], smallest.shape[1] // 2, smallest.shape[2] // 2, 1),
        dtype=smallest.dtype,
    )
    predictions, physical = [], []
    for level in reversed(range(len(prepared["noisy"]))):
        args = (
            prepared["noisy"][level],
            prepared["levels"][level],
            insert,
            prepared["guides"][level],
        )
        if level == 0:
            (insert, raw), features = model.core(*args, return_preclip=True, return_features=True)
            distance = model.auxiliary(features, linear_output=True)
        else:
            insert, raw = model.core(*args, return_preclip=True)
        predictions.append(insert)
        physical.append(raw)
    return tuple(reversed(predictions)), tuple(reversed(physical)), distance


def objective(model, prepared, config, weight):
    predictions, physical, distance = forward_with_auxiliary(model, prepared)
    base, parts = image_loss(
        predictions[0],
        prepared["target"],
        prepared["levels"][0],
        config.loss,
        config.diffusion.min_noise,
        preclip=physical[0],
    )
    field = distance_supervision(
        16 * distance,
        prepared["distance"],
        prepared["target"],
        prepared["levels"][0] > config.diffusion.min_noise,
    )
    loss = base + weight * field
    return loss, dict(parts, loss=loss, auxiliary_distance=field)
