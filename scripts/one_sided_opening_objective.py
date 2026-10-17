"""Opening residual with a detached reference, to remove rather than thicken ink.

The scalar value is identical to opening_consistency. This deliberately uses a
surrogate gradient: the opened prediction is a fixed reference in each update.
At the field level its gradient is nonpositive, so descent can only increase
signed distance (reduce alpha). Shared CNN parameters can still couple pixels;
this is not a guarantee that every model output gets lighter after an update.
"""

import jax
import jax.numpy as jnp
from opening_prior_objective import disk_reduce, soft_opening
from subpixel_field_model import SUBPIXELS


def one_sided_consistency(pred, target, strengths):
    b, k, h, w, c = target.shape
    assert pred.shape == (b, h, w, c)
    pa = jnp.clip(0.5 - SUBPIXELS * pred, 0, 1)
    ta = jnp.clip(0.5 - SUBPIXELS * target, 0, 1)
    opened_target = soft_opening(ta.reshape(b * k, h, w, c)).reshape(ta.shape)
    allowance = jnp.maximum(ta - opened_target, 0).max(1)
    reference = jax.lax.stop_gradient(soft_opening(pa))
    residual = jax.nn.relu(pa - reference - allowance)
    active = jnp.repeat(jnp.repeat(strengths > 0, SUBPIXELS, 1), SUBPIXELS, 2).astype(pred.dtype)
    valid = disk_reduce(disk_reduce(active, maximum=False), maximum=False)
    denominator = jnp.maximum((ta * valid[:, None]).sum((2, 3, 4)).max(1), 1)
    mean_square = (residual * residual * valid).sum((1, 2, 3)) / denominator
    return (jnp.sqrt(mean_square + 0.01**2) - 0.01).mean()
