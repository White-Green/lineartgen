"""Squared distance to the preimage of a clean teacher under clipping to [-1, 1]."""

import jax.numpy as jnp


def interval_squared_error(preclip, clean):
    """Endpoint teachers allow saturated margins; antialiasing targets stay exact.

    For white the valid interval is [1, infinity), for black (-infinity, -1],
    and for interior gray the valid set is the singleton {clean}. This equals
    ordinary squared error inside [-1, 1] and has true, inward gradients at
    incorrect saturation. It does not use a surrogate clipping derivative.
    """
    difference = preclip - clean
    difference = jnp.where(clean == 1, jnp.minimum(difference, 0), difference)
    difference = jnp.where(clean == -1, jnp.maximum(difference, 0), difference)
    return jnp.square(difference)


def interval_absolute_error(preclip, clean):
    """Absolute distance to the same valid set, with zero subgradient at a match."""
    difference = preclip - clean
    difference = jnp.where(clean == 1, jnp.minimum(difference, 0), difference)
    difference = jnp.where(clean == -1, jnp.maximum(difference, 0), difference)
    return jnp.where(difference == 0, jnp.zeros_like(difference), jnp.abs(difference))
