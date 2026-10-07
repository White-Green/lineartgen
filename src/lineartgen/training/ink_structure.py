"""Target-relative ink retention and local weak-spot supervision.

These terms keep antialiasing targets continuous. They are local error objectives,
not a claim of topology preservation or a reproduction of segmentation SCNP.
"""

import jax
import jax.numpy as jnp


def _maximum(x, radius):
    # The max reduction's identity is needed for JAX's max-pool transpose rule.
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        (1, radius * 2 + 1, radius * 2 + 1, 1),
        (1, 1, 1, 1),
        "SAME",
    )


def _mean(values, weights):
    axes = (1, 2, 3)
    numerator = (values * weights).sum(axes)
    denominator = weights.sum(axes)
    return (numerator / jnp.maximum(denominator, 1e-6)).mean()


def ink_structure_losses(prediction, target, active, *, halo_radius=6, neighbor_radius=1):
    """Return separately normalized missing ink and spurious near-background ink.

    The maximum is taken over *errors relative to each pixel's own teacher*,
    never over predicted ink itself. Thus a correct gray/tapered target is a
    zero-loss stationary point. Pool sources and destination anchors are both
    restricted to editable pixels of the teacher-defined class.
    """
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    ink = jnp.where(active, jnp.clip((1 - prediction) / 2, 0, 1), truth)
    support = truth > 0.02
    core = active & support
    near = _maximum(support.astype(jnp.float32), halo_radius) > 0
    paper = active & near & ~support
    deficit = jnp.where(core, jax.nn.relu(truth - ink) ** 2, 0.0)
    excess = jnp.where(paper, jax.nn.relu(ink - truth) ** 2, 0.0)
    return {
        "ink_keep": _mean(deficit, core * truth),
        "neighbor_keep": _mean(_maximum(deficit, neighbor_radius), core * truth),
        "neighbor_paper": _mean(_maximum(excess, neighbor_radius), paper),
    }
