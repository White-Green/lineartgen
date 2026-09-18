"""Line-geometry probes with known paths, separate from thresholded image overlap.

Probe widths integrate antialiased ink coverage along the known normal. They
are compared with the rasterized target, not an assumed integer brush width.
Connection queries distinguish missing intended paths from unintended bridges.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

KINDS = ("straight", "thin", "curve", "corner", "junction", "parallel", "gap", "taper")

BOUNDARY_METRICS = (
    "edit_boundary_mae",
    "edit_boundary_ink_mae",
    "edit_boundary_background_mae",
    "edit_boundary_ink_f1",
)


def edit_boundary_metrics(prediction, target, strength):
    """Three-pixel editable seam; foreground scores require target ink there."""
    p, t, active = prediction[..., 0], target[..., 0], strength[..., 0] > 0
    edge = active & ndimage.binary_dilation(~active, iterations=3)
    ink, background = edge & (t < 0), edge & (t >= 0)
    error = np.abs(p - t)
    predicted = edge & (p < 0)
    return {
        "edit_boundary_mae": float(error[edge].mean()) if edge.any() else None,
        "edit_boundary_ink_mae": float(error[ink].mean()) if ink.any() else None,
        "edit_boundary_background_mae": float(error[background].mean())
        if background.any()
        else None,
        "edit_boundary_ink_f1": 2
        * int((predicted & ink).sum())
        / (int(predicted.sum()) + int(ink.sum()))
        if ink.any()
        else None,
        "edit_boundary_pixels": int(edge.sum()),
        "edit_boundary_ink_pixels": int(ink.sum()),
    }


def line_case(kind, variant, *, size=512):
    random = np.random.default_rng(20260911 + KINDS.index(kind) * 100 + variant)
    width = [1.25, 2.0, 3.0, 5.0][variant % 4]
    angle = np.deg2rad((variant % 4) * 22.5 + (variant // 4) * 90)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = size / 2 + random.uniform(-5, 5, 2)
    x = np.linspace(-170, 170, 201)
    paths = []

    def add(points, widths=width):
        points = np.asarray(points, np.float64)
        points = points @ rotation.T + center
        widths = np.broadcast_to(np.asarray(widths), len(points))
        paths.append((points, widths))

    if kind in ("straight", "thin"):
        add(
            np.column_stack((x, np.zeros_like(x))),
            [1.0, 1.25, 1.5, 1.75][variant % 4] if kind == "thin" else width,
        )
    elif kind == "curve":
        add(np.column_stack((x, 40 * np.sin(x / 70))))
    elif kind == "corner":
        a = np.column_stack((np.linspace(-160, 0, 101), np.zeros(101)))
        b = np.column_stack((np.zeros(100), np.linspace(1.6, 160, 100)))
        add(np.concatenate((a, b)))
    elif kind == "junction":
        for endpoint in [(-145, -120), (145, -120), (0, 165)]:
            add(np.linspace(endpoint, (0, 0), 101))
    elif kind == "parallel":
        separation = width + [3, 5, 8, 12][variant % 4]
        for y in (-separation / 2, separation / 2):
            add(np.column_stack((x, np.full_like(x, y))))
    elif kind == "gap":
        gap = [3, 5, 9, 16][variant % 4]
        for a, b in [(-170, -gap / 2), (gap / 2, 170)]:
            add(np.column_stack((np.linspace(a, b, 101), np.zeros(101))))
    elif kind == "taper":
        add(np.column_stack((x, 35 * np.sin(x / 110))), 1 + 7 * np.cos(x / 340 * np.pi) ** 2)
    else:
        raise ValueError(kind)
    scale = 4
    canvas = Image.new("L", (size * scale, size * scale), 255)
    draw = ImageDraw.Draw(canvas)
    probes, connections = [], []
    for path_index, (points, widths) in enumerate(paths):
        tangent = np.gradient(points, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        left, right = points + normal * widths[:, None] / 2, points - normal * widths[:, None] / 2
        polygon = np.concatenate((left, right[::-1])) * scale
        draw.polygon([tuple(p) for p in polygon], fill=0)
        # Stay away from caps and the junction, where a unique normal width is undefined.
        indices = np.unique(np.linspace(len(points) * 0.15, len(points) * 0.85, 48).astype(int))
        for i in indices:
            if kind == "corner" and abs(i - 100) < 12:
                continue
            # For nearby parallel lines, stop the probe before its neighbor.
            span = min(12.0, separation * 0.48) if kind == "parallel" else 12.0
            probes.append([*points[i], *normal[i], span, path_index])
        connections.append([*points[len(points) // 5], *points[4 * len(points) // 5], 1])
    if kind == "junction":
        for i in (1, 2):
            connections.append([*paths[0][0][20], *paths[i][0][20], 1])
    elif kind in ("gap", "parallel"):
        connections.append([*paths[0][0][50], *paths[1][0][50], 0])
    gray = (
        np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS), np.float32)[..., None]
        / 255
    )
    return gray, np.asarray(probes, np.float32), np.asarray(connections, np.float32)


def _labels_near(labels, point):
    x, y = np.rint(point).astype(int)
    patch = labels[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2]
    return set(np.unique(patch)) - {0}


def _connected(labels, query):
    return bool(_labels_near(labels, query[:2]) & _labels_near(labels, query[2:4]))


def probe_metrics(prediction, target, probes, connections):
    prediction, target = prediction[..., 0], target[..., 0]
    # A fixed quadrature density makes subpixel-width comparisons possible.
    offsets = np.linspace(-1, 1, 97)[None] * probes[:, 4:5]
    xs = probes[:, 0:1] + probes[:, 2:3] * offsets
    ys = probes[:, 1:2] + probes[:, 3:4] * offsets
    profiles = [
        ndimage.map_coordinates(np.clip((1 - value) / 2, 0, 1), [ys, xs], order=1, mode="nearest")
        for value in (prediction, target)
    ]
    widths = [np.trapezoid(p, offsets, axis=1) for p in profiles]
    locations = [
        np.trapezoid(p * offsets, offsets, axis=1) / np.maximum(w, 1e-8)
        for p, w in zip(profiles, widths, strict=True)
    ]
    visible = widths[0] >= np.maximum(0.1, widths[1] * 0.25)
    offset_error = locations[0] - locations[1]
    variation = []
    for path in np.unique(probes[:, 5]):
        selected = (probes[:, 5] == path) & visible
        values = offset_error[selected]
        if len(values) > 1:
            variation.extend(np.abs(np.diff(values)).tolist())
    cuts, bridges, cut_total, bridge_total = 0, 0, 0, 0
    validity = []
    for threshold in (-0.25, 0.0, 0.25):
        labels_p = ndimage.label(prediction < threshold, np.ones((3, 3)))[0]
        labels_t = ndimage.label(target < threshold, np.ones((3, 3)))[0]
        for query in connections:
            present = bool(_labels_near(labels_t, query[:2]) and _labels_near(labels_t, query[2:4]))
            expected = bool(query[4])
            valid = present and _connected(labels_t, query) == expected
            validity.append(valid)
            if not valid:
                continue  # The antialiased raster itself may lack a subthreshold path.
            actual = _connected(labels_p, query)
            if expected:
                cuts += not actual
                cut_total += 1
            else:
                bridges += actual
                bridge_total += 1
    return {
        "profile_width_mae_px": float(np.abs(widths[0] - widths[1]).mean()),
        "profile_missing_fraction": float((~visible).mean()),
        "profile_offset_mae_px": float(np.abs(offset_error[visible]).mean())
        if visible.any()
        else None,
        "profile_offset_variation_px": float(np.mean(variation)) if variation else None,
        "connection_cut_rate": cuts / cut_total if cut_total else None,
        "connection_bridge_rate": bridges / bridge_total if bridge_total else None,
        "connection_queries_valid": int(sum(validity)),
        "connection_queries_total": len(validity),
    }


def protected_metrics(prediction, target, strength):
    protected = strength[..., 0] == 0
    error = np.abs(prediction[..., 0] - target[..., 0])
    active = ~protected
    if active.any():
        p, t = prediction[..., 0][active] < 0, target[..., 0][active] < 0
        denominator = int(p.sum()) + int(t.sum())
        regional = {
            "active_mae": float(error[active].mean()),
            "active_ink_f1": 2 * int((p & t).sum()) / denominator if denominator else 1.0,
            "active_target_ink_fraction": float(t.mean()),
        }
    else:
        regional = {
            "active_mae": None,
            "active_ink_f1": None,
            "active_target_ink_fraction": None,
        }
    if not protected.any():
        return {**regional, "protected_max_error": None, "mask_boundary_mae": None}
    boundary = ndimage.binary_dilation(protected, iterations=3) & ndimage.binary_dilation(
        ~protected, iterations=3
    )
    return {
        **regional,
        "protected_max_error": float(error[protected].max()),
        "mask_boundary_mae": float(error[boundary].mean()) if boundary.any() else None,
    }
