"""Grayscale centerline coverage using the soft skeleton of Shit et al., CVPR 2021.

This adaptation replaces product intersections by minimum intersections, written
as ReLU deficits so exact grayscale targets also have zero gradient. It is not
the paper's binary-target soft-clDice and carries no topology guarantee.
"""

import jax
import jax.numpy as jnp


def _max_pool(image, height, width):
    return jax.lax.reduce_window(
        image, -jnp.inf, jax.lax.max, (1, height, width, 1), (1, 1, 1, 1), "SAME"
    )


def soft_erode(image):
    """Cross-shaped erosion; padding is excluded, matching max-pool padding."""
    return jnp.minimum(-_max_pool(-image, 3, 1), -_max_pool(-image, 1, 3))


def soft_skeleton(image, iterations):
    """Initial soft opening residue, then `iterations` erosion/residue unions.

    Erosion never increases intensity and opening includes the erosion itself.
    Thus the sum of residues is bounded by the input. Cap rounding overshoot to
    retain that bound exactly, including faint antialiased lines.
    """

    def residue(x):
        return jax.nn.relu(x - _max_pool(soft_erode(x), 3, 3))

    def body(_, state):
        current, skeleton = state
        current = soft_erode(current)
        delta = residue(current)
        return current, skeleton + jax.nn.relu(delta - skeleton * delta)

    _, skeleton = jax.lax.fori_loop(0, iterations, body, (image, residue(image)))
    return jnp.minimum(skeleton, image)


def centerline_loss(prediction, target, active, iterations):
    """Harmonic coverage of predicted/target soft centerlines inside editable pixels.

    Morphology sees protected context copied from the target; only editable
    centerline deficits contribute. Images without an editable target skeleton
    leave supervision to the base objective. Per-image ratios are batch-averaged.
    """
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    predicted = jnp.where(active, jnp.clip((1 - prediction) / 2, 0, 1), truth)
    reference = jax.lax.stop_gradient(soft_skeleton(truth, iterations))
    estimated = soft_skeleton(predicted, iterations)
    axes = (1, 2, 3)
    target_mass = (reference * active).sum(axes)
    predicted_mass = (estimated * active).sum(axes)
    # One pixel of smoothing bounds gradients for near-empty skeletons. For binary
    # masks these equal the paper's smoothed product-intersection ratios.
    precision = 1 - (jax.nn.relu(estimated - truth) * active).sum(axes) / (predicted_mass + 1)
    recall = 1 - (jax.nn.relu(reference - predicted) * active).sum(axes) / (target_mass + 1)
    loss = 1 - 2 * precision * recall / jnp.maximum(precision + recall, 1e-6)
    return jnp.where(target_mass > 1e-6, loss, 0).mean()
