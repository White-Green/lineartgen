"""Two-task PCGrad projection, with mean aggregation for both comparison arms."""

from functools import partial

import jax
import jax.numpy as jnp


def dot(a, b):
    return sum(jnp.sum(x * y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


@partial(jax.jit, static_argnames=("project",))
def combine(focus, background, *, project):
    product = dot(focus, background)
    ff, bb = dot(focus, focus), dot(background, background)
    negative = jnp.minimum(product, 0.0) if project else jnp.float32(0)
    # Use original task gradients in both projections. Zero gradients stay finite.
    cf = negative / jnp.maximum(bb, 1e-12)
    cb = negative / jnp.maximum(ff, 1e-12)
    result = jax.tree.map(lambda a, b: ((a - cf * b) + (b - cb * a)) * 0.5, focus, background)
    return result, dict(
        gradient_dot=product,
        gradient_cosine=product / jnp.maximum(jnp.sqrt(ff * bb), 1e-12),
        focus_gradient_norm=jnp.sqrt(ff),
        background_gradient_norm=jnp.sqrt(bb),
        conflicting=(product < 0).astype(jnp.float32),
        gradient_norm=jnp.sqrt(dot(result, result)),
        combined_dot_focus=dot(result, focus),
        combined_dot_background=dot(result, background),
    )
