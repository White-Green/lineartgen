"""Include antialiased painted pixels when isolating an existing line's contrast."""

from .masked_normalization import normalized_masked_guide


def normalized_painted_guide(source, noise):
    if source.shape != noise.shape or source.ndim != 4 or source.shape[-1] != 1:
        raise ValueError("Expected matching NHWC single-channel source and noise")
    maximum = noise.max(axis=(1, 2), keepdims=True)
    # Krita noise = strength * (1 - lineart alpha). Half of a BGRA alpha
    # quantum separates unpainted pixels from even partially painted ones.
    unpainted = noise >= maximum * (1 - 0.5 / 255)
    return normalized_masked_guide(source, unpainted.astype(noise.dtype))
