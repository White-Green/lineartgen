"""Teacher-normalized first differences of interval opacity residuals."""

import jax
import jax.numpy as jnp


def edge_energy_loss(preclip, target, active):
    """Preserve AA targets and valid saturation margins, with no protected force.

    This is an auxiliary, not a standalone reconstruction loss: constant residual
    fields are in its null space. Pixel/opacity terms still supervise their level.
    Neither the teacher nor its normalization receives gradients.
    """
    teacher = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    mask = jax.lax.stop_gradient(active.astype(bool))
    residual = (1 - preclip) / 2 - teacher
    residual = jnp.where(teacher == 0, jax.nn.relu(residual), residual)
    residual = jnp.where(teacher == 1, -jax.nn.relu(-residual), residual)
    residual = jnp.where(mask, residual, 0)
    numerator = jnp.zeros((preclip.shape[0],), preclip.dtype)
    energy = jnp.zeros_like(numerator)
    count = jnp.zeros_like(numerator)
    for axis in (1, 2):
        before = [slice(None)] * 4
        after = list(before)
        before[axis], after[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(before), tuple(after)
        valid = mask[left] & mask[right]
        difference = residual[right] - residual[left]
        target_difference = teacher[right] - teacher[left]
        numerator += (difference**2 * valid).sum((1, 2, 3))
        energy += (target_difference**2 * valid).sum((1, 2, 3))
        count += valid.sum((1, 2, 3))
    # Constant teachers use a per-edge mean rather than amplifying every blank pixel.
    denominator = jnp.where(energy > 0, jnp.maximum(energy, 1), jnp.maximum(count, 1))
    return (numerator / jax.lax.stop_gradient(denominator)).mean()
