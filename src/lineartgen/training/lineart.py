"""Teacher-local ink supervision and true normal profiles of generated curves.

Grayscale targets remain continuous. Geometry coordinates never come from predictions.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates


class Profiles(NamedTuple):
    coordinates: object  # [batch, profiles, samples, 2], y/x pixel coordinates
    valid: object  # [batch, profiles]


def make_profiles(records, batch, size, *, seed, step, count=16, samples=49):
    coordinates = np.zeros((batch, count, samples, 2), np.float32)
    valid = np.zeros((batch, count), np.float32)
    allowed = {"straight", "thin", "bezier", "taper", "parallel", "gap"}
    for record in records:
        item = record["item"]
        rng = np.random.default_rng([seed, step, item, 0x50524F46])
        candidates = []
        for path in record.get("path_geometry", []):
            if path["kind"] not in allowed:
                continue
            points, widths = np.asarray(path["points"]), np.asarray(path["widths"])
            tangent = np.gradient(points, axis=0)
            length = np.linalg.norm(tangent, axis=1)
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.maximum(
                length[:, None], 1e-8
            )
            for index in range(8, len(points) - 8):
                if length[index] < 1e-8:
                    continue
                offsets = np.linspace(-1, 1, samples) * (widths[index] / 2 + 6)
                xy = points[index] + offsets[:, None] * normal[index]
                if (xy >= 1).all() and (xy <= size - 2).all():
                    candidates.append(xy[:, ::-1].copy())
        if candidates:
            selected = rng.choice(len(candidates), min(count, len(candidates)), replace=False)
            coordinates[item, : len(selected)] = np.asarray(candidates)[selected]
            valid[item, : len(selected)] = 1
    return Profiles(coordinates, valid)


def _smooth_absolute(x):
    return jnp.sqrt(x * x + 1e-6) - 0.001


def _per_image_mean(values, weights):
    axes = tuple(range(1, values.ndim))
    return ((values * weights).sum(axes) / jnp.maximum(weights.sum(axes), 1)).mean()


def line_region_loss(prediction, target, active, radius=6):
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    ink = jnp.clip((1 - prediction) / 2, 0, 1)
    support = truth > 0.02
    near = (
        jax.lax.reduce_window(
            support.astype(jnp.float32),
            0.0,
            jax.lax.max,
            (1, radius * 2 + 1, radius * 2 + 1, 1),
            (1, 1, 1, 1),
            "SAME",
        )
        > 0
    )
    core = _per_image_mean(_smooth_absolute(ink - truth), active & support)
    halo = _per_image_mean(jax.nn.relu(ink - truth), active & near & ~support)
    background = _per_image_mean(_smooth_absolute(ink - truth), active & ~near)
    return core + halo + 0.25 * background


def profile_loss(prediction, target, active, profiles, *, level=0):
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    ink = jnp.where(active, jnp.clip((1 - prediction) / 2, 0, 1), truth)
    # Block-average pyramid pixels are centered at 2**level*j+(2**level-1)/2.
    scale = 2**level
    coordinates = (profiles.coordinates - (scale - 1) / 2) / scale

    def sample(image, coords):
        return map_coordinates(
            image[..., 0], jnp.moveaxis(coords, -1, 0), order=1, mode="constant", cval=0
        )

    p, y = jax.vmap(sample)(ink, coordinates), jax.vmap(sample)(truth, coordinates)
    edit = jax.vmap(sample)(active.astype(jnp.float32), coordinates)
    weights = edit * profiles.valid[..., None]
    values = _per_image_mean(_smooth_absolute(p - y), weights)
    derivative = _smooth_absolute(jnp.diff(p, axis=-1) - jnp.diff(y, axis=-1))
    edges = jnp.maximum(weights[..., 1:], weights[..., :-1])
    return values + 0.5 * _per_image_mean(derivative, edges)
