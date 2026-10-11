"""Small NHWC convolution blocks, shared by the encoder, decoder and output head."""

from flax import nnx


class ConvBlock(nnx.Module):
    def __init__(
        self, inputs, outputs, *, kernel_size=3, depth=3, dilation=1, hidden_channels=None, rngs
    ):
        middle = max(inputs, outputs) if hidden_channels is None else hidden_channels
        channels = [inputs] + [middle] * (depth - 1) + [outputs]
        self.layers = nnx.List(
            [
                nnx.Conv(
                    a,
                    b,
                    (kernel_size, kernel_size),
                    padding="SAME",
                    kernel_dilation=(dilation, dilation),
                    rngs=rngs,
                )
                for a, b in zip(channels[:-1], channels[1:], strict=True)
            ]
        )

    def __call__(self, x, *, linear_output=False, first_bias=None, first_modulation=None):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index == 0 and first_bias is not None:
                x = x + first_bias
            if index == 0 and first_modulation is not None:
                scale, shift = first_modulation
                x = x * (1 + scale) + shift
            if not linear_output or index < len(self.layers) - 1:
                x = nnx.relu(x)
        return x
