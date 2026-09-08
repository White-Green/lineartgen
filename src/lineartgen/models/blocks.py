"""Small NHWC convolution blocks, shared by the encoder, decoder and output head."""

from flax import nnx


class ConvBlock(nnx.Module):
    def __init__(self, inputs, outputs, *, kernel_size=3, depth=3, rngs):
        middle = max(inputs, outputs)
        channels = [inputs] + [middle] * (depth - 1) + [outputs]
        self.layers = nnx.List(
            [
                nnx.Conv(a, b, (kernel_size, kernel_size), padding="SAME", rngs=rngs)
                for a, b in zip(channels[:-1], channels[1:], strict=True)
            ]
        )

    def __call__(self, x, *, linear_output=False):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if not linear_output or index < len(self.layers) - 1:
                x = nnx.relu(x)
        return x
