"""Allow a single clear trace only when a close parallel pair has filled-in rough support."""

import copy

import numpy as np
from plausible_data import OFFSETS, SIZE, raster
from scipy.ndimage import map_coordinates

MAX_SEPARATION = 8.0
MIN_PEAK = 0.02
CENTER_RATIO = 0.6
REQUIRED_FRACTION = 0.75
POSITIONS = np.linspace(0.05, 0.95, 17)
CROSS_SECTION = np.linspace(-1, 1, 65)


def merge_geometry(record):
    empty = dict(
        merge_xy=np.zeros((17, 65, 2), np.float32),
        merge_valid=np.zeros(17, bool),
        merge_eligible=np.bool_(False),
        merge_separation=np.float32(0),
    )
    paths = record["metadata"].get("path_geometry", [])
    if record["kind"] != "parallel" or len(paths) != 2:
        return empty
    a, b = [np.asarray(p["points"]) for p in paths]
    if a.shape != b.shape or len(a) < 2:
        return empty
    separation = float(np.median(np.linalg.norm(a - b, axis=-1)))
    if not 0 < separation <= MAX_SEPARATION:
        return empty | dict(merge_separation=np.float32(separation))
    index = np.rint(POSITIONS * (len(a) - 1)).astype(int)
    mid, difference = (a[index] + b[index]) / 2 + 32, b[index] - a[index]
    xy = mid[:, None] + CROSS_SECTION[None, :, None] * difference[:, None]
    valid = ((xy >= 0) & (xy <= SIZE - 1)).all((1, 2))
    return dict(
        merge_xy=xy.astype(np.float32),
        merge_valid=valid,
        merge_eligible=np.bool_(valid.sum() >= 8),
        merge_separation=np.float32(separation),
    )


def merge_allowed(source, geometry):
    if not bool(geometry["merge_eligible"]):
        return False
    xy = geometry["merge_xy"]
    ink = np.clip((1 - np.asarray(source)[..., 0]) / 2, 0, 1)
    samples = map_coordinates(ink, [xy[..., 1], xy[..., 0]], order=1, mode="constant", cval=0)
    peak = samples.max(-1)
    # The whole middle fifth of the pair must have appreciable ink, not just one isolated point.
    center = samples[:, np.abs(CROSS_SECTION) <= 0.1].min(-1)
    valid = geometry["merge_valid"] & (peak >= MIN_PEAK)
    if valid.sum() < 8:
        return False
    return bool(np.mean(center[valid] / peak[valid] >= CENTER_RATIO) >= REQUIRED_FRACTION)


def merged_candidates(canonical, record):
    geometry = merge_geometry(record)
    if not bool(geometry["merge_eligible"]):
        return canonical[1::3].copy(), geometry
    metadata = copy.deepcopy(record["metadata"])
    a, b = metadata["path_geometry"]
    center = (np.asarray(a["points"]) + np.asarray(b["points"])) / 2
    metadata["path_geometry"] = [
        dict(kind="bezier", points=center.tolist(), widths=np.full(len(center), 2.0).tolist())
    ]
    metadata["paths"] = 1
    # Preserve the original global displacement convention and 2px style.
    merged = np.stack(
        [raster(metadata, np.asarray(record["direction"]), float(o), 1) for o in OFFSETS]
    )
    return merged, geometry


def allowed_candidates(canonical, merged, source, geometry):
    result = canonical.copy()
    allowed = merge_allowed(source, geometry)
    if allowed:
        # Retain all original nine choices; replace only one of three duplicate slots.
        result[1::3] = merged
    return result, allowed
