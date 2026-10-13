"""Endpoint and shared-junction regions, independent of candidate selection."""

import numpy as np

RADIUS = 8


def structural_region(metadata, *, size=256, margin=32):
    # Shared endpoints (junctions) are unioned, not counted once per incident path.
    # Keep all globally shifted candidates inside the same region.
    region = np.zeros((size, size, 1), np.uint8)
    for path in metadata.get("path_geometry", []):
        for point in (path["points"][0], path["points"][-1]):
            x, y = np.rint(np.asarray(point) + margin).astype(int)
            x0, x1 = max(0, x - RADIUS), min(size, x + RADIUS + 1)
            y0, y1 = max(0, y - RADIUS), min(size, y + RADIUS + 1)
            if x0 < x1 and y0 < y1:
                region[y0:y1, x0:x1, 0] = 1
    return region
