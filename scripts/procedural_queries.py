"""Fresh geometry and local graph queries, keyed only by training seed/update."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from train_connection_queries import PATCH, point_at, query_spec

from lineartgen.config import GeometryDataConfig
from lineartgen.data.geometry import KINDS, draw_geometry
from lineartgen.training.connectivity import connection_capacity


class Queries(NamedTuple):
    origins: object  # B,Q,xy
    source: object  # B,Q,H,W
    sink: object


def generate(seed, step, batch=4, size=512):
    config = GeometryDataConfig(minimum_width=0.6, maximum_width=6, supersampling=4)
    clean, origins, sources, sinks, records = [], [], [], [], []
    for item in range(batch):
        rng = np.random.default_rng([seed, step, item, 0x5147454F])
        kind = KINDS[(step * batch + item) % len(KINDS)]
        if rng.random() < 0.05:
            image = np.ones((size, size, 1), np.float32)
            meta = dict(path_geometry=[], kinds=["blank"], paths=0)
            kind = "blank"
        else:
            image, meta = draw_geometry(rng, size, config, kind=kind, include_paths=True)
            image = np.clip((image * 2 - 1) * 1.1, -1, 1)
        clean.append(image)
        paths = [np.asarray(p["points"]) for p in meta["path_geometry"]]
        pairs = [None, None]
        if paths:
            points = paths[int(rng.integers(len(paths)))]
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            center = rng.uniform(0.25, 0.75) * length
            halfspan = min(10, length * 0.2)
            pairs[0] = tuple(point_at(points, center + d) for d in (-halfspan, halfspan))
        if kind == "gap":
            ends = min(
                ((a, b) for a in paths[0][[0, -1]] for b in paths[1][[0, -1]]),
                key=lambda ab: np.linalg.norm(ab[0] - ab[1]),
            )
            pairs[1] = tuple(
                point_at(p if np.all(p[0] == end) else p[::-1], 8)
                for p, end in zip(paths, ends, strict=True)
            )
        elif kind == "parallel":
            index = int(rng.integers(25, 104))
            pairs[1] = (paths[0][index], paths[1][index])
        elif kind in ("junction", "crossing"):
            pairs[1] = (point_at(paths[0], 10), point_at(paths[-1], 10))
        oo, aa, bb = [], [], []
        valid = []
        for q, pair in enumerate(pairs):
            good = pair is not None and all((p >= 2).all() and (p < size - 2).all() for p in pair)
            if good:
                # The experiment is512px; query_spec uses that explicit contract.
                assert size == 512
                spec, masks = query_spec(item, *pair, q == 0)
                oo.append(spec["origin"])
                aa.append(masks[0])
                bb.append(masks[1])
            else:
                oo.append([0, 0])
                aa.append(np.zeros((PATCH, PATCH), bool))
                bb.append(np.zeros((PATCH, PATCH), bool))
            valid.append(good)
        origins.append(oo)
        sources.append(aa)
        sinks.append(bb)
        records.append(
            dict(
                item=item,
                kind=kind,
                query_valid=valid,
                widths=[float(np.mean(p["widths"])) for p in meta["path_geometry"]],
            )
        )
    return (
        np.stack(clean),
        Queries(np.asarray(origins, np.int32), np.asarray(sources), np.asarray(sinks)),
        records,
    )


def patches(images, queries):
    yy, xx = jnp.indices((PATCH, PATCH))
    batch = jnp.arange(images.shape[0])[:, None, None, None]
    y = queries.origins[..., 1, None, None] + yy
    x = queries.origins[..., 0, None, None] + xx
    return images[batch, y, x, 0].reshape(-1, PATCH, PATCH)


def query_loss(prediction, target, active, queries):
    # Protected output pixels remain in the connectivity graph without gradients.
    alpha = jnp.clip((1 - prediction) / 2, 0, 1)
    alpha = jnp.where(active, alpha, jax.lax.stop_gradient(alpha))
    truth = jax.lax.stop_gradient(jnp.clip((1 - target) / 2, 0, 1))
    a, b = (x.reshape(-1, PATCH, PATCH) for x in (queries.source, queries.sink))
    predicted = jax.vmap(connection_capacity)(patches(alpha, queries), a, b)
    expected = jax.lax.stop_gradient(jax.vmap(connection_capacity)(patches(truth, queries), a, b))
    valid = a.any((1, 2)) & b.any((1, 2))
    positive = valid & (expected > 0)
    negative = valid & (expected == 0)
    deficit = jnp.sum(jnp.where(positive, jax.nn.relu(expected - predicted) ** 2, 0))
    bridge = jnp.sum(jnp.where(negative, predicted**2, 0))
    deficit /= jnp.maximum(positive.sum(), 1)
    bridge /= jnp.maximum(negative.sum(), 1)
    return (deficit + bridge) / 2, dict(
        connection_deficit=deficit,
        connection_bridge=bridge,
        positive_queries=positive.sum(),
        negative_queries=negative.sum(),
    )
