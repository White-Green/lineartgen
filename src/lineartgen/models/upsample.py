"""Keep nearest-neighbor pixels while summing the four input adjoints explicitly."""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def nearest_double(image):
    """Double NHWC spatial dimensions; preserve the existing forward operation."""
    if image.ndim != 4:
        raise ValueError("nearest_double expects an NHWC array")
    b, h, w, c = image.shape
    return jax.image.resize(image, (b, h * 2, w * 2, c), "nearest")


@nearest_double.defjvp
def _nearest_double_jvp(primals, tangents):
    (image,), (tangent,) = primals, tangents
    b, h, w, c = image.shape
    # The transpose reduces the four copies with a fixed reduction, rather
    # than accumulating repeated gather indices through a scatter addition.
    repeated = jnp.broadcast_to(tangent.reshape(b, h, 1, w, 1, c), (b, h, 2, w, 2, c))
    return nearest_double(image), repeated.reshape(b, 2 * h, 2 * w, c)
