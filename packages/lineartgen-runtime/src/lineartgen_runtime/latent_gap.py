"""Measure opposing stroke support for a missing part of a sampled raster."""

from math import cos, pi, sin


def agreed_role(mean, rejected, xp):
    """Rescue a weak stretch only when most of the connected rough supports keeping it."""
    return xp.clip((mean - 0.62) / 0.22, 0, 1) * xp.clip((0.29 - rejected) / 0.18, 0, 1)


def selected_coverage(original, learned, low, high, xp):
    """Turn the keep evidence into a decision while retaining the predicted stroke alpha."""
    relative = learned / xp.maximum(0.003, original)
    u = xp.clip((relative - low) / (high - low), 0, 1)
    return original * u * u * (3 - 2 * u)


def paper_opacity(ink, peak, threshold, xp):
    u = xp.clip((peak - 0.015) / (threshold - 0.015), 0, 1)
    return ink * u * u * (3 - 2 * u)


def completion_mask(rough, selected, xp, peak3):
    support = opposing_support(peak3(xp.maximum(0.25 * rough, selected)), xp)
    supported = xp.clip((support - 0.025) / 0.20, 0, 1)
    blank = 1 - xp.clip((peak3(rough) - 0.012) / 0.065, 0, 1)
    return blank * supported


def facing_endpoints(rough, xp, peak3):
    """Find pairs of actual rough ends that face into a gap, excluding parallel through-lines."""
    _, h, w, _ = rough.shape

    def at(padded, y, x, pad):
        return padded[:, pad + y : pad + y + h, pad + x : pad + x + w]

    def mean(x, r):
        p = xp.pad(x, ((0, 0), (r, r), (r, r), (0, 0)))
        return (
            sum(at(p, y, xoff, r) for y in range(-r, r + 1) for xoff in range(-r, r + 1))
            / (2 * r + 1) ** 2
        )

    sm = mean(rough, 1)
    p = xp.pad(sm, ((0, 0), (1, 1), (1, 1), (0, 0)))
    gx = (at(p, 0, 1, 1) - at(p, 0, -1, 1)) * 0.5
    gy = (at(p, 1, 0, 1) - at(p, -1, 0, 1)) * 0.5
    a, b, c = xp.split(mean(xp.concatenate((gx * gx, gx * gy, gy * gy), -1), 3), 3, -1)
    theta = 0.5 * xp.arctan2(2 * b, a - c)
    tx, ty = -xp.sin(theta), xp.cos(theta)
    source = peak3(rough)
    yy = xp.arange(h)[None, :, None, None]
    xx = xp.arange(w)[None, None, :, None]
    batch = xp.arange(len(source))[:, None, None, None]

    def along(d):
        x = xx + xp.rint(tx * d).astype(xp.int32)
        y = yy + xp.rint(ty * d).astype(xp.int32)
        return xp.where(
            (x >= 0) & (x < w) & (y >= 0) & (y < h),
            source[batch, xp.clip(y, 0, h - 1), xp.clip(x, 0, w - 1), 0],
            0,
        )

    l4, l7, r4, r7 = [along(d) for d in (-4, -7, 4, 7)]

    def tip(inside, outside):
        return xp.clip((inside - 1.10 * outside - 0.018) / (inside + 0.035), 0, 1) * xp.clip(
            source / 0.09, 0, 1
        )

    right = tip(xp.minimum(r4, r7), xp.maximum(l4, l7))
    left = tip(xp.minimum(l4, l7), xp.maximum(r4, r7))
    result = xp.zeros_like(rough)
    margin = 25
    for i in range(8):
        dx, dy = cos(i * pi / 8), sin(i * pi / 8)
        dot = tx * dx + ty * dy
        align = xp.clip((xp.abs(dot) - 0.77) / 0.16, 0, 1)
        ends = []
        for sign in (-1, 1):
            tags = peak3(align * xp.where(sign * dot > 0, right, left))
            p = xp.pad(tags, ((0, 0), (margin, margin), (margin, margin), (0, 0)))
            strongest = xp.zeros_like(rough)
            for r in (2, 4, 6, 8, 10, 13, 16, 19, 22, 25):
                strongest = xp.maximum(
                    strongest, at(p, round(sign * r * dy), round(sign * r * dx), margin)
                )
            ends.append(strongest)
        result = xp.maximum(result, xp.minimum(*ends))
    return xp.clip((result - 0.04) / 0.35, 0, 1)


def opposing_support(source, xp):
    """Input is NHWC local peak ink. Look for stretches of ink on both sides."""
    _, h, w, _ = source.shape
    margin = 24
    p = xp.pad(source, ((0, 0), (margin, margin), (margin, margin), (0, 0)))
    evidence = xp.zeros_like(source)
    for i in range(8):
        dx, dy = cos(i * pi / 8), sin(i * pi / 8)
        ends = []
        for sign in (-1, 1):
            strongest = xp.zeros_like(source)
            for radius in (3, 6, 10, 14, 20):
                offsets = [
                    (margin + round(sign * dy * d), margin + round(sign * dx * d))
                    for d in (radius, radius + 3)
                ]
                values = [p[:, y : y + h, x : x + w] for y, x in offsets]
                strongest = xp.maximum(strongest, xp.minimum(*values))
            ends.append(strongest)
        evidence = xp.maximum(evidence, xp.minimum(*ends))
    return evidence
