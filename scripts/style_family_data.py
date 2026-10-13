"""Fixed peak-width line style, independent of rough opacity and original line width."""

import copy

import numpy as np
from plausible_data import CHOICES, raster

PEAK_WIDTH = 2.0


def styled_candidates(data, record):
    if record["kind"] in ("blank", "black", "wide"):
        return data["variants"].copy()
    metadata = copy.deepcopy(record["metadata"])
    peak = max(max(p["widths"]) for p in metadata["path_geometry"])
    assert peak > 0
    for path in metadata["path_geometry"]:
        path["widths"] = (np.asarray(path["widths"]) * (PEAK_WIDTH / peak)).tolist()
    # Three duplicated width slots keep candidate indexing and cardinality fixed.
    by_offset = {o: raster(metadata, np.asarray(record["direction"]), o, 1) for o, _ in CHOICES}
    return np.stack([by_offset[o] for o, _ in CHOICES])


def extended_data(data, record):
    """Same broad style-aware evaluation family for every method, declared in advance."""
    from plausible_data import profile_samples

    styles = styled_candidates(data, record)
    variants = np.concatenate((data["variants"], styles), axis=0)
    return data | dict(
        variants=variants,
        choices=CHOICES + CHOICES,
        profiles=np.stack([profile_samples(x, data["probes"]) for x in variants]),
    )
