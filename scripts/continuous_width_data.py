"""Change only the source width of existing straight broad training strokes."""

import copy

import numpy as np
from plausible_data import raster
from rough_family_data import BASE, rough_source


def width_choice(seed, step, item, index):
    if index < 464:
        return None
    rng = np.random.default_rng([seed, step, item, 0x57494454])
    if rng.random() < 0.25:
        return None
    return float(np.exp(rng.uniform(np.log(4), np.log(32))))


def width_source(data, record, index, width, options):
    if width is None:
        return (
            data["source"][index]
            if options == BASE
            else rough_source(data["variants"][index, [1, 25]].mean(0), options)
        )
    assert record["kind"] == "wide"
    metadata = copy.deepcopy(record["metadata"])
    peak = max(max(p["widths"]) for p in metadata["path_geometry"])
    for p in metadata["path_geometry"]:
        p["widths"] = (np.asarray(p["widths"]) * (width / peak)).tolist()
    direction = np.asarray(record["direction"])
    mixture = np.stack([raster(metadata, direction, offset, 1) for offset in [-2.0, 2.0]]).mean(0)
    return rough_source(mixture, options)
