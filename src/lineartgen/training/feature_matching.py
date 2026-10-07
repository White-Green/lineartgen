"""Teacher-relative features from a frozen conditional patch critic.

This is a diagnostic building block, not wired into the active generator study.
The caller freezes critic parameters; features preserve their spatial coordinates.
"""

import jax
import jax.numpy as jnp


def feature_distance(
    critic, prediction, target, condition, active, *, metric="smooth_l1", layers=(0, 1, 2)
):
    """Per-example mean feature error; protected prediction pixels have zero gradient.

    The three hidden layers are compared before the classifier head. Each layer
    contributes its mean over spatial positions and channels. Correct continuous
    grayscale targets have zero loss and gradient, including antialiasing.
    """
    if metric not in ("smooth_l1", "l2") or not layers or any(i not in (0, 1, 2) for i in layers):
        raise ValueError("Unsupported feature distance")
    target, condition = jax.lax.stop_gradient((target, condition))
    prediction = jnp.where(active, prediction, target)
    fake = jnp.concatenate((condition, prediction), -1)
    real = jnp.concatenate((condition, target), -1)
    values = []
    for index, layer in enumerate(critic.layers):
        fake = jax.nn.leaky_relu(layer(fake), negative_slope=0.2)
        real = jax.nn.leaky_relu(layer(real), negative_slope=0.2)
        if index in layers:
            difference = fake - jax.lax.stop_gradient(real)
            error = (
                jnp.sqrt(jnp.square(difference) + 1e-6) - 0.001
                if metric == "smooth_l1"
                else jnp.square(difference)
            )
            values.append(error.mean((1, 2, 3)))
    return jnp.stack(values).mean(0)
