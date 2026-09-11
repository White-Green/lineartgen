"""One-way channel groups: larger models reuse the smaller models' activations.

Group i only reads groups 0..i. This includes decoder skips and the previous
resolution's predictions, so every prefix is an autonomous recursive model.
All groups keep features through the final decoder; separate heads produce
one prediction per prefix. Inputs/outputs pack recursive predictions in NHWC
channel order [smallest, ..., largest].

Only allowed connections are parameters. `fused` pads their kernels with
constant zeros for one dense convolution; `blocks` evaluates the same weights
as separate convolutions. Both modes share each group's activations once.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .blocks import ConvBlock
from .output import denoised_output


def concatenate_groups(left, right):
    """Keep each group's skip/features together, rather than mixing prefixes."""
    return tuple(jnp.concatenate((a, b), -1) for a, b in zip(left, right, strict=True))


class CausalConv(nnx.Module):
    def __init__(self, inputs, outputs, *, kernel_size, rngs, execution):
        self.execution = execution
        self.groups = nnx.List(
            [
                nnx.Conv(sum(inputs[: i + 1]), width, (kernel_size, kernel_size), rngs=rng)
                for i, (width, rng) in enumerate(zip(outputs, rngs, strict=True))
            ]
        )

    def __call__(self, groups):
        if self.execution == "blocks":
            return tuple(
                conv(jnp.concatenate(groups[: i + 1], -1)) for i, conv in enumerate(self.groups)
            )
        inputs = jnp.concatenate(groups, -1)
        kernels = [
            jnp.pad(
                conv.kernel[...],
                ((0, 0), (0, 0), (0, inputs.shape[-1] - conv.kernel.shape[-2]), (0, 0)),
            )
            for conv in self.groups
        ]
        output = jax.lax.conv_general_dilated(
            inputs,
            jnp.concatenate(kernels, -1),
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + jnp.concatenate([conv.bias[...] for conv in self.groups])
        boundaries = tuple(np.cumsum([conv.bias.shape[0] for conv in self.groups])[:-1])
        return tuple(jnp.split(output, boundaries, axis=-1))


class CausalConvBlock(nnx.Module):
    def __init__(self, inputs, outputs, *, depth, kernel_size, rngs, execution):
        # Match max(in, out) at each prefix, including the original smallest block.
        middle = tuple(
            int(width)
            for width in np.diff([0, *np.maximum(np.cumsum(inputs), np.cumsum(outputs)).tolist()])
        )
        widths = [inputs] + [middle] * (depth - 1) + [outputs]
        self.layers = nnx.List(
            [
                CausalConv(a, b, kernel_size=kernel_size, rngs=rngs, execution=execution)
                for a, b in zip(widths[:-1], widths[1:], strict=True)
            ]
        )

    def __call__(self, groups):
        for layer in self.layers:
            groups = tuple(nnx.relu(x) for x in layer(groups))
        return groups


class NestedUNet(nnx.Module):
    def __init__(self, config, *, rngs):
        widths = (config.channels, *config.nested_channels)
        self.num_subnets = len(widths)
        self.minimum_size = config.minimum_size
        self.output_mode = config.output_mode
        # A group's initialization is independent of how many extensions exist.
        key = rngs.params()
        group_rngs = tuple(nnx.Rngs(jax.random.fold_in(key, i)) for i in range(self.num_subnets))
        increments = tuple(
            tuple(int(x) for x in np.diff([0, *(width[stage] for width in widths)]))
            for stage in range(len(config.channels))
        )

        def block(inputs, outputs):
            return CausalConvBlock(
                inputs,
                outputs,
                depth=config.convolutions_per_block,
                kernel_size=config.kernel_size,
                rngs=group_rngs,
                execution=config.nested_execution,
            )

        # The external image/noise map belongs to the common input group.
        encoder_inputs = [(2, *([0] * (self.num_subnets - 1)))] + [
            tuple(c + (1 if stage == 0 else 0) for c in channels)
            for stage, channels in enumerate(increments[:-1])
        ]
        skips = [tuple(c + 1 for c in increments[0]), *increments[1:]]
        self.encoders = nnx.List(
            [block(a, b) for a, b in zip(encoder_inputs, increments, strict=True)]
        )
        previous = (0,) * self.num_subnets
        decoders = []
        for stage in reversed(range(len(increments))):
            # Every extension retains two final features, not just the smallest.
            outputs = encoder_inputs[stage] if stage else (2,) * self.num_subnets
            inputs = tuple(a + b for a, b in zip(skips[stage], previous, strict=True))
            decoders.append(block(inputs, outputs))
            previous = outputs
        self.decoders = nnx.List(decoders)
        self.heads = nnx.List(
            [
                ConvBlock(
                    1 + sum(previous[: i + 1]),
                    1,
                    kernel_size=config.kernel_size,
                    depth=config.convolutions_per_block,
                    rngs=rng,
                )
                for i, rng in enumerate(group_rngs)
            ]
        )

    def __call__(self, image, noise_level, insert):
        if insert.shape[-1] != self.num_subnets:
            raise ValueError("insert must contain one prediction per retained subnet")
        empty = image[..., :0]
        groups = (jnp.concatenate((image, noise_level), -1),) + (empty,) * (self.num_subnets - 1)
        skips = []
        for stage, encoder in enumerate(self.encoders):
            groups = tuple(
                nnx.max_pool(x, (2, 2), (2, 2), padding="VALID") for x in encoder(groups)
            )
            if stage == 0:
                groups = concatenate_groups(
                    groups, tuple(insert[..., i : i + 1] for i in range(self.num_subnets))
                )
            skips.append(groups)
        for stage, (decoder, skip) in enumerate(zip(self.decoders, reversed(skips), strict=True)):
            groups = skip if stage == 0 else concatenate_groups(skip, groups)
            groups = tuple(
                jax.image.resize(
                    x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), "nearest"
                )
                for x in groups
            )
            groups = decoder(groups)
        return jnp.concatenate(
            [
                denoised_output(
                    head(jnp.concatenate((image, *groups[: i + 1]), -1), linear_output=True),
                    image,
                    noise_level,
                    self.output_mode,
                )
                for i, head in enumerate(self.heads)
            ],
            -1,
        )

    def extract(self, index):
        """Copy a prefix, removing every parameter belonging only to larger models."""
        if not 0 <= index < self.num_subnets:
            raise ValueError(f"subnet index must be in [0, {self.num_subnets})")
        result = nnx.clone(self)
        result.num_subnets = index + 1
        for block in (*result.encoders, *result.decoders):
            for layer in block.layers:
                layer.groups = nnx.List(list(layer.groups)[: index + 1])
        result.heads = nnx.List(list(result.heads)[: index + 1])
        return result

    def set_execution(self, execution):
        if execution not in ("fused", "blocks"):
            raise ValueError("execution must be 'fused' or 'blocks'")
        for block in (*self.encoders, *self.decoders):
            for layer in block.layers:
                layer.execution = execution

    def to_unet(self):
        """The smallest prefix is exactly the original dense UNet architecture."""
        from lineartgen.config import ModelConfig

        from .unet import UNet

        if self.num_subnets != 1:
            raise ValueError("Only the smallest prefix has the one-channel UNet contract")
        config = ModelConfig(
            channels=tuple(block.layers[-1].groups[0].bias.shape[0] for block in self.encoders),
            convolutions_per_block=len(self.encoders[0].layers),
            kernel_size=self.encoders[0].layers[0].groups[0].kernel.shape[0],
            output_mode=self.output_mode,
        )
        model = UNet(config, rngs=nnx.Rngs(0))
        for source, target in zip(
            (*self.encoders, *self.decoders), (*model.encoders, *model.decoders), strict=True
        ):
            for a, b in zip(source.layers, target.layers, strict=True):
                nnx.update(b, nnx.state(a.groups[0], nnx.Param))
        nnx.update(model.head, nnx.state(self.heads[0], nnx.Param))
        return model, config
