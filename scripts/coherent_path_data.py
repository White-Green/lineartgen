"""Dense arc probes for continuous profile matching; does not warp the clean target."""

import numpy as np
from coherent_path_objective import PATHS, SAMPLES
from path_profile_data import path_samples


def dense_annotation(metadata):
    probes = np.zeros((PATHS, SAMPLES, 6), np.float32)
    probes[..., 4] = 10
    valid = np.zeros((PATHS, SAMPLES), np.float32)
    spacing = np.ones((PATHS, SAMPLES - 1), np.float32)
    paths = metadata.get("path_geometry", [])
    assert len(paths) <= PATHS
    for i, path in enumerate(paths):
        centers, normals, widths = path_samples(path, np.linspace(0, 1, SAMPLES))
        probes[i] = np.c_[centers, normals, np.maximum(10, widths * 1.5 + 6), np.full(SAMPLES, i)]
        valid[i] = 1
        spacing[i] = np.maximum(np.linalg.norm(np.diff(centers, axis=0), axis=-1), 1e-6)
    return dict(dense_probes=probes.reshape(-1, 6), dense_valid=valid.reshape(-1), spacing=spacing)
