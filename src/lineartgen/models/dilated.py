"""Equivalent dense convolutions on residue classes for spatial dilations."""

import jax
import jax.numpy as jnp


def dilated_same(inputs, kernel, dilation, *, precision=None, preferred_element_type=None):
    """NHWC, odd square kernel, stride one, symmetric zero padding.

    Each residue class modulo dilation is an independent ordinary convolution.
    Padding to a multiple of dilation is removed after interleaving the outputs.
    This avoids expensive dilated filter gradients on the current ROCm backend.
    """
    n, height, width, channels = inputs.shape
    ph, pw = (-height) % dilation, (-width) % dilation
    h, w = (height + ph) // dilation, (width + pw) // dilation
    x = jnp.pad(inputs, ((0, 0), (0, ph), (0, pw), (0, 0)))
    x = x.reshape(n, h, dilation, w, dilation, channels)
    x = x.transpose(0, 2, 4, 1, 3, 5).reshape(n * dilation**2, h, w, channels)
    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    y = y.reshape(n, dilation, dilation, h, w, kernel.shape[-1])
    y = y.transpose(0, 3, 1, 4, 2, 5).reshape(n, height + ph, width + pw, kernel.shape[-1])
    return y[:, :height, :width]


def local_convolution(layer, inputs):
    """Keep undilated kernels exact; specialize supported dilated local kernels."""
    ds = layer.kernel_dilation
    ds = (ds, ds) if isinstance(ds, int) else tuple(ds)
    if (
        ds[0] == 1
        or ds[0] != ds[1]
        or inputs.ndim != 4
        or layer.kernel_size != (3, 3)
        or layer.padding != "SAME"
        or layer.strides not in (1, (1, 1), None)
        or layer.input_dilation not in (1, (1, 1), None)
        or layer.feature_group_count != 1
        or layer.mask is not None
    ):
        return layer(inputs)
    bias = layer.bias[...] if layer.bias is not None else None
    x, kernel, bias = layer.promote_dtype((inputs, layer.kernel[...], bias), dtype=layer.dtype)
    y = dilated_same(
        x,
        kernel,
        ds[0],
        precision=layer.precision,
        preferred_element_type=layer.preferred_element_type,
    )
    return y + bias if layer.use_bias else y
