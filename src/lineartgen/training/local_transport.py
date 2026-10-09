"""Experimental local opacity-distribution loss; not enabled in any trainer.

Sorted samples give the1D empirical transport pairing. Teacher-dependent rank
weights and separate line/blank tile means are our adaptation, not a Wasserstein
metric. This constrains opacity amount/distribution, not geometry or topology.
Physical preclip opacity supplies a correcting gradient outside output clipping.
"""

import jax
import jax.numpy as jnp


def _tiles(image, size):
    batch, height, width, channels = image.shape
    if channels != 1 or height % size or width % size:
        raise ValueError("Local transport requires grayscale images divisible by tile size")
    return (
        image.reshape(batch, height // size, size, width // size, size)
        .transpose(0, 1, 3, 2, 4)
        .reshape(batch, -1, size * size)
    )


def local_opacity_transport(preclip, target, active, *, tile=16, distance="squared"):
    """Match each tile's sorted opacity; retain AA targets and protected pixels.

    White targets represent a saturation interval. Gray and black targets
    remain finite exact opacity values, so an overblack spur cannot occupy a
    free unbounded black rank. Line ranks and paper ranks get equal
    weight within a line tile; blank tiles use their paper error. Means across
    line tiles and blank tiles are separate, with blank weight.25. Equality
    has zero loss and gradient; excessively black physical outputs are penalized
    even when their clipped black appearance is correct.
    """
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    prediction = jnp.where(active, (1 - preclip) / 2, truth)
    p, y = jnp.sort(_tiles(prediction, tile), axis=-1), jnp.sort(_tiles(truth, tile), axis=-1)
    difference = jnp.where(y == 0, jax.nn.relu(p), p - y)
    if distance == "squared":
        error = difference**2
    elif distance == "smooth_absolute":
        error = jnp.sqrt(difference**2 + 1e-6) - 0.001
    else:
        raise ValueError(f"Unknown local transport distance: {distance}")
    ink = y > 0.02
    core = jnp.where(ink, error, 0).sum(-1) / jnp.maximum(ink.sum(-1), 1)
    paper = jnp.where(~ink, error, 0).sum(-1) / jnp.maximum((~ink).sum(-1), 1)
    line = ink.any(-1)
    editable = _tiles(active, tile).any(-1)
    line = line & editable
    blank = ~ink.any(-1) & editable
    line_mean = jnp.where(line, (core + paper) / 2, 0).sum(-1) / jnp.maximum(line.sum(-1), 1)
    blank_mean = jnp.where(blank, paper, 0).sum(-1) / jnp.maximum(blank.sum(-1), 1)
    return jnp.mean(line_mean + 0.25 * blank_mean)
