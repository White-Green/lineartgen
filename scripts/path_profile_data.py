"""Uniform arc-length profiles covering every path, including both legs of corners."""

import numpy as np

from lineartgen.data.geometry import normals

FRACTIONS = np.linspace(0.05, 0.95, 17)
MAX_PATHS = 4


def path_samples(path, fractions=FRACTIONS):
    points = np.asarray(path["points"], np.float64) + 32
    distance = np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    keep = np.r_[True, np.diff(distance) > 1e-8]
    positions = distance[-1] * np.asarray(fractions)
    centers = np.stack(
        [np.interp(positions, distance[keep], points[keep, i]) for i in range(2)], -1
    )
    widths = np.interp(positions, distance[keep], np.asarray(path["widths"])[keep])
    normal = normals(points)
    normal = np.stack([np.interp(positions, distance[keep], normal[keep, i]) for i in range(2)], -1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    return centers, normal, widths


def path_probes(metadata):
    probes, connections, anchors = [], [], []
    paths = metadata.get("path_geometry", [])
    assert len(paths) <= MAX_PATHS
    for index, path in enumerate(paths):
        centers, normal, widths = path_samples(path)
        radius = np.maximum(10, widths * 1.5 + 6)
        probes.extend(np.c_[centers, normal, radius, np.full(len(centers), index)].tolist())
        # Include short segments throughout the complete path and one end-to-end query.
        for a, b in zip(centers[:-1], centers[1:], strict=True):
            connections.append([*a, *b, 1])
        connections.append([*centers[0], *centers[-1], 1])
        anchors.append(centers[len(centers) // 2])
    if len(anchors) > 1 and metadata["kinds"][0] in ("parallel", "gap", "junction", "crossing"):
        for other in anchors[1:]:
            connections.append(
                [*anchors[0], *other, int(metadata["kinds"][0] in ("junction", "crossing"))]
            )
    return np.asarray(probes, np.float32).reshape(-1, 6), np.asarray(
        connections, np.float32
    ).reshape(-1, 5)


def training_annotation(metadata):
    probes, _ = path_probes(metadata)
    padded = np.zeros((MAX_PATHS * len(FRACTIONS), 6), np.float32)
    padded[:, 4] = 10
    padded[: len(probes)] = probes
    valid = np.zeros(len(padded), np.float32)
    valid[: len(probes)] = 1
    return dict(probes=padded, valid=valid)
