"""Research graph predictor: discrete existence/connections, continuous xy/width."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from centerline_grid import encode, render
from flax import nnx
from learned_coarse_axis_model import LearnedCoarseAxisCNN

from lineartgen.diffusion import source_guides
from lineartgen.models.serialization import _arrays

DIRECTIONS = ((1, 0), (0, 1), (1, 1), (-1, 1))
CHANNELS = 9  # vertex, dx, dy, log(width), four undirected edges, opaque fill cell


def target(segments, widths, rectangles, size, level=0):
    scale = 2**level
    g = encode(segments / scale, widths / scale, 1.0)
    n = size // scale
    out = np.zeros((n, n, CHANNELS), np.float32)
    for cell, point, width in zip(g["cells"], g["vertices"], g["widths"], strict=True):
        x, y = cell
        if 0 <= x < n and 0 <= y < n:
            out[y, x, :4] = [1, *(point - cell), np.log(max(width, 1e-8))]
    for a, b in g["edges"]:
        ca, cb = g["cells"][a], g["cells"][b]
        delta = tuple(cb - ca)
        if delta not in DIRECTIONS:
            ca, cb = cb, ca
            delta = tuple(cb - ca)
        k = DIRECTIONS.index(delta)
        x, y = ca
        if min(*ca, *cb) >= 0 and max(*ca, *cb) < n:
            out[y, x, 4 + k] = 1
    yy, xx = np.mgrid[:n, :n] + 0.5
    for r in rectangles / scale:
        out[..., 8] = np.maximum(
            out[..., 8], (xx >= r[0]) & (xx < r[2]) & (yy >= r[1]) & (yy < r[3])
        )
    return out, g["stats"]


def decode(head):
    """Probability is never used as opacity; zero logit is the frozen cutoff."""
    h = np.asarray(head)
    size = h.shape[0]
    assert h.shape == (size, size, CHANNELS)
    occupied = h[..., 0] >= 0
    yy, xx = np.mgrid[:size, :size]
    xy = np.stack((xx, yy), -1) + 1 / (1 + np.exp(-np.clip(h[..., 1:3], -40, 40)))
    widths = np.exp(np.clip(h[..., 3], -8, np.log(8)))
    segments, ww = [], []
    degree = np.zeros_like(occupied, dtype=int)
    edges = []
    for k, (dx, dy) in enumerate(DIRECTIONS):
        for y, x in np.argwhere(occupied & (h[..., 4 + k] >= 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and occupied[ny, nx]:
                segments.append([xy[y, x], xy[ny, nx]])
                ww.append([widths[y, x], widths[ny, nx]])
                degree[y, x] += 1
                degree[ny, nx] += 1
                edges.append(((int(y), int(x)), (int(ny), int(nx))))
    isolated = occupied & (degree == 0)
    for y, x in np.argwhere(isolated):
        segments.append([xy[y, x], xy[y, x]])
        ww.append([widths[y, x], widths[y, x]])
    # Filled cells are hard geometry; pixel-aligned pilot rectangles only.
    s = np.asarray(segments).reshape(-1, 2, 2)
    w = np.asarray(ww).reshape(-1, 2)
    image = render(s, w, np.empty((0, 4)), size)
    image[h[..., 8] >= 0] = -1
    nodes = [tuple(map(int, p)) for p in np.argwhere(occupied)]
    parent = {p: p for p in nodes}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for a, b in edges:
        parent[find(a)] = find(b)
    return image, dict(
        vertices=len(nodes),
        edges=len(edges),
        isolated=int(isolated.sum()),
        components=len({find(p) for p in nodes}),
        fill_cells=int((h[..., 8] >= 0).sum()),
    )


class GraphCNN(LearnedCoarseAxisCNN):
    def __init__(self, config, *, rngs):
        super().__init__(config, mode="choice", rngs=rngs)
        self.research_model = "learned_centerline_grid"
        self.head = nnx.Linear(
            self.width * 8, CHANNELS, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        bias = np.zeros(CHANNELS, np.float32)
        bias[[0, 4, 5, 6, 7, 8]] = -4
        self.head.bias[...] = jnp.asarray(bias)


def convert(previous, config):
    m = GraphCNN(config.model, rngs=nnx.Rngs(2026101807))
    state = nnx.to_pure_dict(nnx.state(m, nnx.Param))
    old = nnx.to_pure_dict(nnx.state(previous, nnx.Param))
    for key in state:
        if key != "head":
            state[key] = old[key]
    restored = nnx.state(m, nnx.Param)
    nnx.replace_by_pure_dict(restored, state)
    nnx.update(m, restored)
    return m


def forward(model, prepared):
    guides = source_guides(model, prepared.guides)
    g = guides[-1]
    insert = jnp.zeros((g.shape[0], g.shape[1] // 2, g.shape[2] // 2, 1), g.dtype)
    heads = [None] * len(guides)
    for level in reversed(range(len(guides))):
        h = model(
            prepared.noisy[level],
            prepared.noise_levels[level],
            insert,
            guides[level],
            prepared.progress[level],
            pyramid_level=level,
            return_network_output=True,
        )
        heads[level] = h
        # Soft latent occupancy summary only; not the displayed raster/opacity.
        v, fill = jax.nn.sigmoid(h[..., :1]), jax.nn.sigmoid(h[..., 8:9])
        insert = 1 - 2 * (1 - (1 - v) * (1 - fill))
    return tuple(heads)


infer_heads = nnx.jit(forward)


def balanced_bce(logits, labels):
    axes = tuple(range(1, logits.ndim))
    values = optax.sigmoid_binary_cross_entropy(logits, labels)
    pos = (values * labels).sum(axes) / jnp.maximum(labels.sum(axes), 1)
    neg = (values * (1 - labels)).sum(axes) / jnp.maximum((1 - labels).sum(axes), 1)
    return (pos + neg).mean()


def stage_loss(head, truth):
    occupied = truth[..., 0]
    count = jnp.maximum(occupied.sum((1, 2)), 1)
    offset = optax.huber_loss(jax.nn.sigmoid(head[..., 1:3]), truth[..., 1:3]).mean(-1)
    width = optax.huber_loss(head[..., 3], truth[..., 3])
    parts = dict(
        vertex=balanced_bce(head[..., 0], occupied),
        edge=balanced_bce(head[..., 4:8], truth[..., 4:8]),
        fill=balanced_bce(head[..., 8], truth[..., 8]),
        position=((offset * occupied).sum((1, 2)) / count).mean(),
        width=((width * occupied).sum((1, 2)) / count).mean(),
    )
    return parts


def objective(model, prepared, targets):
    heads = forward(model, prepared)
    weights = [0.5**i for i in range(len(heads))]
    parts = [stage_loss(h, t) for h, t in zip(heads, targets, strict=True)]
    totals = {
        k: sum(w * p[k] for w, p in zip(weights, parts, strict=True)) / sum(weights)
        for k in parts[0]
    }
    totals["loss"] = sum(totals.values())
    return totals["loss"], totals


@nnx.jit
def gradient(model, prepared, targets):
    (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared, targets)
    return parts, grads


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: np.asarray(v) for k, v in _arrays(nnx.to_pure_dict(nnx.state(model, nnx.Param)))},
    )


def load_weights(model, path):
    state = nnx.state(model, nnx.Param)
    template = nnx.to_pure_dict(state)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {k for k, _ in _arrays(template)}

        def restore(tree, path=()):
            if isinstance(tree, dict):
                return {k: restore(v, (*path, k)) for k, v in tree.items()}
            a = z[json.dumps(path)]
            assert a.shape == tree.shape and a.dtype == tree.dtype
            return jnp.asarray(a)

        nnx.replace_by_pure_dict(state, restore(template))
    nnx.update(model, state)
