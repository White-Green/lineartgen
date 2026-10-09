"""One-sided target opacity deficit, normalized by each image's editable ink."""

import jax
import jax.numpy as jnp


def opacity_deficit_loss(prediction, target, active, *, unclipped=False):
    """Keep weak/antialiased ink without rewarding excess darkness or background ink.

    With clipped inputs, nonempty images match the evaluation ink-deficit fraction.
    With physical preclip inputs, unclipped=True extends the same one-sided term
    beyond white, where its direct derivative would otherwise be blocked.
    Empty editable ink
    contributes zero to the batch mean. This is a pixel retention term, not a
    guarantee of topology or a conditional-mean-preserving flow objective.
    """
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    core = active & (truth > 0.02)
    ink = (1 - prediction) / 2
    if not unclipped:
        ink = jnp.clip(ink, 0, 1)
    deficit = jnp.where(core, jax.nn.relu(truth - ink), 0.0)
    axes = (1, 2, 3)
    numerator = deficit.sum(axes)
    denominator = jnp.where(core, truth, 0.0).sum(axes)
    return (numerator / jnp.maximum(denominator, 1e-6)).mean()
