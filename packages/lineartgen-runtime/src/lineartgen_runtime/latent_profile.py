"""Stroke-normal and stroke-tangent ink profiles, using NumPy or JAX array operations."""


def stroke_profiles(cells, xp, mean_pool, max_pool):
    """Use the local structure tensor to sample both sides of a possible retrace."""
    ink = cells.mean(-1, keepdims=True)
    smooth = mean_pool(ink, 3)[..., 0]
    padded = xp.pad(smooth, 1)
    gx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    gy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    tensor = mean_pool(xp.stack((gx * gx, gy * gy, gx * gy), -1), 9)
    a, b, c = tensor[..., 0], tensor[..., 1], tensor[..., 2]
    u, v = a - b, 2 * c
    coherence = xp.hypot(u, v) / xp.maximum(a + b, 1e-8)
    # Doubling the angle identifies a direction independently of its sign.
    axis = xp.argmax(xp.stack((u, v, -u, -v), -1), -1)
    h, w = smooth.shape
    yy, xx = xp.indices((h, w))
    peak = xp.maximum(0.045, max_pool(ink, 33)[..., 0])
    vals = [coherence, smooth / peak]
    for span in (3, 7, 15, 31):
        mean = mean_pool(ink, span)[..., 0]
        var = xp.maximum(0, mean_pool(ink**2, span)[..., 0] - mean**2)
        vals.extend((mean / peak, xp.sqrt(var) / peak))
    for radius in (2, 5, 10, 20):
        diag = round(radius * 0.70710678)
        dx = xp.asarray([radius, diag, 0, diag])[axis]
        dy = xp.asarray([0, diag, radius, -diag])[axis]
        for sx, sy in ((dx, dy), (-dy, dx)):

            def lookup(x, y):
                return xp.where(
                    (x >= 0) & (x < w) & (y >= 0) & (y < h),
                    smooth[xp.clip(y, 0, h - 1), xp.clip(x, 0, w - 1)],
                    0,
                )

            left, right = lookup(xx - sx, yy - sy), lookup(xx + sx, yy + sy)
            vals.extend((xp.minimum(left, right) / peak, xp.maximum(left, right) / peak))
    return xp.stack(vals, -1)
