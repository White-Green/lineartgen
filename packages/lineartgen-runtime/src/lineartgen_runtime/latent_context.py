"""Raster connectivity and stroke decisions shared across a full diffusion canvas."""

import numpy as np

from .latent_profile import stroke_profiles


def pack(image):
    h, w, c = image.shape
    return (
        image.reshape(h // 2, 2, w // 2, 2, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(h // 2, w // 2, 4 * c)
    )


def unpack(cells):
    h, w, c = cells.shape
    return cells.reshape(h, w, 2, 2, c // 4).transpose(0, 2, 1, 3, 4).reshape(2 * h, 2 * w, c // 4)


def repeat2(x):
    return x.repeat(2, axis=0).repeat(2, axis=1)


def max_pool(x, size):
    """Centered separable maximum over non-negative input, in logarithmically many passes."""
    for axis in (0, 1):
        y = np.moveaxis(x, axis, 0)
        r = size // 2
        y = np.pad(y, [(r, r)] + [(0, 0)] * (y.ndim - 1))
        length = 1
        while length < size:
            step = min(length, size - length)
            y = np.maximum(y[:-step], y[step:])
            length += step
        x = np.moveaxis(y, 0, axis)
    return x


def mean_pool(x, size):
    r = size // 2
    prefix = np.pad(x, ((r + 1, r), (r + 1, r), (0, 0)))
    prefix = prefix.cumsum(0, dtype=np.float64).cumsum(1)
    total = (
        prefix[size:, size:]
        - prefix[:-size, size:]
        - prefix[size:, :-size]
        + prefix[:-size, :-size]
    )
    return (total / size**2).astype(np.float32)


def path_fields(cells):
    peak = cells.max(-1, keepdims=True)
    occupancy = peak > np.array([0.035, 0.10], np.float32)
    result = []
    for axis in (0, 1):
        slices = np.moveaxis(occupancy, axis, 0)

        def scan(occupied):
            distances = np.zeros(occupied.shape, np.float32)
            previous = np.zeros_like(distances[0])
            for index, row in enumerate(occupied):
                neighbor = previous.copy()
                np.maximum(neighbor[:-1], previous[1:], out=neighbor[:-1])
                np.maximum(neighbor[1:], previous[:-1], out=neighbor[1:])
                previous = distances[index] = (neighbor + 1) * row
            return distances

        left = np.moveaxis(scan(slices), 0, axis)
        right = np.moveaxis(scan(slices[::-1])[::-1], 0, axis)
        span = np.maximum(0, left + right - 1)
        result.extend(
            (
                span / np.maximum(1, span.max((0, 1))),
                span / np.maximum(1, max_pool(span, 33)),
                np.log1p(span) / 6,
                np.minimum(left, right) / np.maximum(1, span),
            )
        )
    for size in (5, 19):
        result.append(cells / np.maximum(0.045, max_pool(peak, size)))
    return np.concatenate(result, -1)


def roots(mask, diagonal):
    """Minimum raster index for each component. Pixel outside the mask uses mask.size."""
    n = mask.size
    indices = np.arange(n + 1, dtype=np.int32)
    labels = np.where(mask, indices[:-1].reshape(mask.shape), n)
    while True:
        p = np.pad(labels, 1, constant_values=n)
        h, w = labels.shape
        neighbor = labels.copy()
        for dy in range(3):
            for dx in range(3):
                if diagonal or dx == 1 or dy == 1:
                    np.minimum(neighbor, p[dy : dy + h, dx : dx + w], out=neighbor)
        parent = indices.copy()
        np.minimum.at(parent, labels[mask], neighbor[mask])
        new = parent[parent[labels]]
        if np.array_equal(new, labels):
            return labels
        labels = new


class Components:
    def __init__(self, mask, diagonal=True):
        labels = roots(mask, diagonal)
        ids, inverse, count = np.unique(labels, return_inverse=True, return_counts=True)
        self.mask = mask
        self.ids = inverse.reshape(mask.shape).astype(np.int32)
        self.count = count.astype(np.float32)
        self.active = ids < mask.size

    def sum(self, values):
        return np.bincount(self.ids.ravel(), weights=values.ravel(), minlength=len(self.count))

    def means(self, values):
        return np.stack([self.sum(v) / self.count for v in np.moveaxis(values, -1, 0)], -1).astype(
            np.float32
        )

    def features(self, peak, fields):
        ids, mask = self.ids, self.mask
        area = self.count
        yy, xx = np.indices(ids.shape)
        bbox = []
        for axis in (yy, xx):
            lo = np.full(len(area), axis.max() + 1, np.int32)
            hi = np.zeros_like(lo)
            np.minimum.at(lo, ids, axis)
            np.maximum.at(hi, ids, axis)
            bbox.append((hi - lo + 1).astype(np.float32))
        length = np.hypot(*bbox) * self.active
        local_span = length[ids] / np.maximum(1, max_pool(length[ids], 33))
        region_basic = np.stack(
            (
                length / max(1, length.max()),
                np.log1p(length) / 6,
                np.log1p(area) / 9,
                area / (bbox[0] * bbox[1]),
            ),
            -1,
        )
        # The relative local extent varies within a component; the selector consumes it per cell.
        observed = self.means(np.concatenate((peak, fields[..., :16]), -1))
        base = (
            np.concatenate(
                (
                    region_basic[:, :1][ids],
                    local_span[..., None],
                    region_basic[:, 1:][ids],
                    observed[ids],
                ),
                -1,
            )
            * mask[..., None]
        )
        p = np.pad(mask.astype(np.int8), 1)
        border = (
            self.sum(4 - (p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1])).astype(
                np.float32
            )
            * self.active
        )
        # Each nonempty 2x2 window belongs to one 8-connected foreground component.
        sentinel = len(area)
        pad = np.pad(np.where(mask, ids, sentinel), 1, constant_values=sentinel)
        corners = np.stack((pad[:-1, :-1], pad[:-1, 1:], pad[1:, :-1], pad[1:, 1:]), -1)
        present = corners < sentinel
        n = present.sum(-1)
        diag = (n == 2) & (
            (present[..., 0] & present[..., 3]) | (present[..., 1] & present[..., 2])
        )
        score = (n == 1).astype(np.int32) - (n == 3) - 2 * diag
        euler = (
            np.bincount(corners.min(-1).ravel(), weights=score.ravel(), minlength=sentinel + 1)[:-1]
            / 4
        )
        scale = max(mask.shape)
        x, y = xx.astype(np.float32) / scale, yy.astype(np.float32) / scale
        moments = self.means(np.stack((x, y, x * x, y * y, x * y), -1))
        vx = np.maximum(0, moments[:, 2] - moments[:, 0] ** 2)
        vy = np.maximum(0, moments[:, 3] - moments[:, 1] ** 2)
        cov = moments[:, 4] - moments[:, 0] * moments[:, 1]
        elongation = np.hypot(vx - vy, 2 * cov) / np.maximum(1e-6, vx + vy)
        surrounding = self.sum(border[ids] / np.maximum(1, max_pool(border[ids], 33))) / area
        contour = np.stack(
            (
                np.log1p(np.maximum(0, 1 - euler)),
                np.log1p(border) / 8,
                border / max(1, border.max()),
                surrounding,
                np.log1p(area / np.maximum(1, border)) / 3,
                np.log1p(border / np.sqrt(area)) / 4,
                elongation,
                np.sqrt(vx + vy),
            ),
            -1,
        )
        descriptor = np.concatenate((self.means(base), contour), -1).astype(np.float32)
        descriptor[~self.active] = 0
        return base, descriptor


def paper_fields(ink):
    paper = Components(~ink, diagonal=False)
    ids = paper.ids
    border = np.zeros(len(paper.count), bool)
    border[ids[[0, -1], :]] = True
    border[ids[:, [0, -1]]] = True
    inside = ~ink & ~border[ids]
    channels = np.stack(
        (inside, ~ink & border[ids], inside * np.log1p(paper.count[ids]) / 11), -1
    ).astype(np.float32)
    return [mean_pool(channels, r) for r in (3, 7, 15)]


def ray_fields(peak):
    h, w = peak.shape[:2]
    for r in (2, 6, 14, 30):
        support = max_pool(peak, 2 * (1 + r // 8) + 1)
        p = np.pad(support, ((r, r), (r, r), (0, 0)))
        diag = round(0.707 * r)
        for dx, dy in ((r, 0), (0, r), (diag, diag), (diag, -diag)):
            a = p[r + dy : r + dy + h, r + dx : r + dx + w]
            b = p[r - dy : r - dy + h, r - dx : r - dx + w]
            yield np.minimum(a, b)
            yield np.maximum(a, b)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def dense(weights, name, x):
    return x @ weights[name + "/kernel"] + weights[name + "/bias"]


def silu(x):
    return x * sigmoid(x)


def role_network(weights, name, x, blocks):
    def layer(i, values):
        return dense(weights, f"{name}/Dense_{i}", values)

    state = silu(layer(0, x))
    for i in range(blocks):
        # Flax assigns names when constructing the outer Dense before its argument.
        state += layer(2 * i + 1, silu(layer(2 * i + 2, state)))
    return np.clip(layer(2 * blocks + 1, silu(state))[..., 0], -9, 9)


def resize_linear(values, shape):
    for axis, length in enumerate(shape):
        size = values.shape[axis]
        loc = np.clip(
            (np.arange(length, dtype=np.float32) + 0.5) * (size / length) - 0.5, 0, size - 1
        )
        left = loc.astype(np.int32)
        weight = (loc - left).astype(np.float32).reshape((-1, 1, 1) if axis == 0 else (1, -1, 1))
        a = np.take(values, left, axis=axis)
        b = np.take(values, np.minimum(size - 1, left + 1), axis=axis)
        values = a + weight * (b - a)
    return values


class StrokeContext:
    """Compute global raster descriptors once per scale; reuse them during reverse diffusion."""

    def __init__(self, rough, power, weights):
        self.rough = rough
        self.weights = weights
        cells = pack(rough)
        peak = cells.max(-1, keepdims=True)
        self.fields = np.concatenate((path_fields(cells), pack(power).mean(-1, keepdims=True)), -1)
        self.regions = []
        roles, descriptors, valid = [], [], []
        for threshold in (0.035, 0.10):
            group = Components(peak[..., 0] > threshold)
            basic, stats = group.features(peak, self.fields)
            self.regions.append((group, basic[group.mask]))
            role = role_network(weights, "role_prior", stats, 1)
            roles.append(role[group.ids])
            descriptors.append(stats)
            valid.append(group.mask)
        role = np.stack(roles, -1)
        valid = np.stack(valid, -1)
        trust = (1 + np.abs(role)) * valid
        avg = np.sum(trust * role, -1) / np.maximum(1, trust.sum(-1))
        if "arc_prior/Dense_0/kernel" in weights:
            # Stream rows of large descriptor vectors through the classifier.
            paper = [v for group, _ in self.regions for v in paper_fields(group.mask)]
            rays = list(ray_fields(peak))
            if weights["arc_prior/Dense_0/kernel"].shape[0] > 140:
                rays.append(stroke_profiles(cells, np, mean_pool, max_pool))
            for y in range(0, len(peak), 32):
                rows = slice(y, y + 32)
                inputs = [cells[rows], self.fields[rows, :, :24]]
                inputs.extend(
                    stats[group.ids[rows]]
                    for (group, _), stats in zip(self.regions, descriptors, strict=True)
                )
                inputs.append(role[rows])
                inputs.extend(value[rows] for value in (*paper, *rays))
                arc = role_network(weights, "arc_prior", np.concatenate(inputs, -1), 2)
                avg[rows] = 0.15 * avg[rows] + 0.85 * arc
        self.prior = avg[..., None]
        self.observed = valid.any(-1, keepdims=True)

    def select(self, local, features, proposal, local_limit=0.0):
        w = self.weights

        def project(name, x):
            return dense(w, "region_selector/" + name, x)

        state = resize_linear(project("Dense_0", features), local.shape[:2])
        scores = [local[..., 0]]
        mixes = [np.full(local.shape[:2], -1.5, np.float32)]
        for group, basic in self.regions:
            parts = [basic, state[group.mask], local[group.mask]]
            hidden = silu(project("region_features", np.concatenate(parts, -1)))
            ids = group.ids[group.mask]
            values = np.concatenate((local[group.mask], hidden), -1)
            pooled = (
                np.stack(
                    [np.bincount(ids, weights=v, minlength=len(group.count)) for v in values.T], -1
                ).astype(np.float32)
                / group.count[:, None]
            )
            scores.append(
                (pooled[:, :1] + project("region_correction", pooled[:, 1:]))[group.ids, 0]
            )
            mixes.append(
                np.where(group.mask, project("region_weight", pooled[:, 1:])[group.ids, 0], -1e4)
            )
        blend = np.stack(mixes, -1)
        blend = np.exp(blend - blend.max(-1, keepdims=True))
        selected = (blend * np.stack(scores, -1)).sum(-1, keepdims=True) / blend.sum(
            -1, keepdims=True
        )
        if local_limit:
            selected = local_limit * np.tanh(selected / local_limit)
        if "prior_fusion/Dense_0/kernel" not in w:
            return repeat2(np.where(self.observed, 0.22 * selected + 0.78 * self.prior, selected))
        x = np.concatenate(
            (
                self.fields,
                pack(self.rough),
                pack(proposal),
                np.repeat(selected / 5, 4, -1),
                np.repeat(self.prior / 5, 4, -1),
                np.repeat(self.observed.astype(np.float32), 4, -1),
            ),
            -1,
        )
        for i in (0, 1):
            x = silu(dense(w, f"prior_fusion/Dense_{i}", x))
        adjustment = dense(w, "prior_fusion/Dense_2", x)
        trust = sigmoid(1.25 + adjustment[..., :4]) * self.observed
        return unpack(trust * self.prior + (1 - trust) * selected + adjustment[..., 4:])
