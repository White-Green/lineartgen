"""Locate opposite edges of an enclosed paper region near an existing ink stroke."""

import numpy as np

from .latent_context import Components, max_pool, repeat2


def tip_extent(ink, tip, direction, span):
    # Measure whether the two rough edges continue together beyond the enclosed paper.
    radius = min(64, max(8, round(float(span) * 0.4)))
    x, y = np.rint(tip).astype(int)
    y0, x0 = max(0, y - radius), max(0, x - radius)
    yy, xx = np.nonzero(ink[y0 : y + radius + 1, x0 : x + radius + 1])
    xy = np.column_stack((xx + x0, yy + y0)) - tip
    along = xy @ direction
    across = np.abs(xy @ np.array([-direction[1], direction[0]]))
    valid = (along >= 0) & (along < radius) & (across < 3 + 0.25 * along)
    used = np.bincount(along[valid].astype(int), minlength=radius)
    # End at the first two consecutive unoccupied stations, ignoring disconnected marks.
    stop = np.flatnonzero((used[:-1] == 0) & (used[1:] == 0))
    return np.log1p(stop[0] if len(stop) else radius) / 4


def loop_features(rough, coverage, *, tails=False, half_resolution=False):
    h, w = rough.shape
    result = np.zeros(((h + 1) // 2, (w + 1) // 2, 7 if tails else 5), np.float32)
    if np.any(coverage > 0.08):
        ph, pw = result.shape[:2]

        def peak(values):
            padded = np.pad(values, ((0, h % 2), (0, w % 2)))
            return padded.reshape(ph, 2, pw, 2).max((1, 3))

        marks = peak(coverage)
        raw = peak(rough)
        ink = raw > np.maximum(0.045, 0.3 * max_pool(raw, 9))
        paper = Components(~ink, diagonal=False)
        bounds = np.unique(
            np.concatenate((paper.ids[0], paper.ids[-1], paper.ids[:, 0], paper.ids[:, -1]))
        )
        allowed = paper.active.copy()
        allowed[bounds] = False
        touched = (max_pool(marks, 17) > 0.08) & paper.mask
        candidates = np.unique(paper.ids[touched])
        candidates = candidates[allowed[candidates] & (paper.count[candidates] >= 8)]
        eligible = np.zeros_like(allowed)
        eligible[candidates] = True
        rows, columns = np.nonzero(eligible[paper.ids])
        owners = paper.ids[rows, columns]
        order = np.argsort(owners, kind="stable")
        ends = np.cumsum(paper.count[candidates].astype(np.int64))
        for idx, positions in zip(
            candidates, np.split(order, ends[:-1]) if len(ends) else (), strict=True
        ):
            yy, xx = rows[positions], columns[positions]
            y0, x0 = max(0, yy.min() - 8), max(0, xx.min() - 8)
            crop = (slice(y0, min(ph, yy.max() + 9)), slice(x0, min(pw, xx.max() + 9)))
            hole = paper.ids[crop] == idx
            rim = max_pool(hole, 17) & ink[crop]
            if not np.any(rim & (marks[crop] > 0.08)):
                continue
            coords = np.column_stack((xx, yy)).astype(np.float32)
            center = coords.mean(0)
            coords -= center
            eig, axes = np.linalg.eigh(coords.T @ coords / len(coords))
            major = axes[:, -1]
            minor = axes[:, 0]
            along = coords @ major
            origin = along.min()
            bins = np.rint(along - origin).astype(np.int32)
            counts = np.bincount(bins)
            sums = np.bincount(bins, weights=coords @ minor)
            ticks = np.flatnonzero(counts)
            middle = np.interp(np.arange(len(counts)), ticks, sums[ticks] / counts[ticks])
            ry, rx = np.nonzero(rim)
            xy = np.column_stack((rx + x0, ry + y0)) - center
            longitudinal = xy @ major - origin
            offset = xy @ minor - np.interp(longitudinal, np.arange(len(middle)), middle)
            near = (longitudinal >= -2) & (longitudinal <= len(middle) + 1)
            side = np.clip(offset, -1, 1) * near
            paint = marks[crop][ry, rx]
            vote = float(np.sum(paint * side) / max(1.0, np.sum(paint * near)))
            score = side * vote
            ar = float(np.sqrt(max(1, eig[1]) / max(1, eig[0])))
            signal = np.column_stack(
                (
                    score,
                    np.abs(score),
                    near * np.log1p(ar) / 4,
                    near * np.log1p(eig[0]) / 6,
                    near * np.log1p(eig[1]) / 10,
                )
            )
            if tails:
                span = len(middle)
                first = center + major * origin + minor * middle[0]
                last = center + major * (origin + span - 1) + minor * middle[-1]
                extents = sorted(
                    (tip_extent(ink, first, -major, span), tip_extent(ink, last, major, span))
                )
                signal = np.column_stack((signal, near * extents[0], near * extents[1]))
            prev = result[crop][ry, rx]
            use = np.abs(score) > np.abs(prev[:, 0])
            result[crop][ry[use], rx[use]] = signal[use]
    return result if half_resolution else repeat2(result)[:h, :w]
