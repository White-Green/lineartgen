"""Equal per-image importance for present paper, antialias, and opaque-ink regions."""

import jax.numpy as jnp


def region_masks(clean, active):
    return tuple(mask & active for mask in (clean >= 1, (clean > -1) & (clean < 1), clean <= -1))


def region_velocity_mse(prediction, target_velocity, clean, active):
    error = jnp.square(prediction - target_velocity)
    losses, present = [], []
    for mask in region_masks(clean, active):
        count = mask.sum((1, 2, 3))
        losses.append((error * mask).sum((1, 2, 3)) / jnp.maximum(count, 1))
        present.append(count > 0)
    count_present = jnp.stack(present).astype(error.dtype).sum(0)
    per_image = jnp.stack(losses).sum(0) / jnp.maximum(count_present, 1)
    return per_image.mean()
