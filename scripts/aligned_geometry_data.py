"""Keep input displacement in opaque stroke teachers, independently per component."""

import copy

import jax.numpy as jnp
import numpy as np
from aligned_field_inputs import inverse_warp
from d4_geometry_data import transform_numpy_batch
from opaque_geometry import geometry
from subpixel_field_data import geometry_field
from support_field_data import merged_metadata


def warped_metadata(metadata, options):
    result = copy.deepcopy(metadata)
    for path in result.get("path_geometry", []):
        physical = np.asarray(path["points"], np.float64) + 32
        path["points"] = (inverse_warp(physical, options) - 32).tolist()
    return result


def component_field(data, index, options, *, merged=False):
    record = data["records"][index]
    cached = data["merged_fields"][data["merge_index"][index]] if merged else data["fields"][index]
    if not options["amplitude"] or not record["metadata"].get("path_geometry"):
        return np.asarray(cached)
    metadata = merged_metadata(record) if merged else record["metadata"]
    metadata = warped_metadata(metadata, options)
    args = tuple(jnp.asarray(x) for x in geometry(metadata, kind=record["kind"], size=256))
    return np.asarray(geometry_field(*args, size=256)).astype(np.float16)


def aligned_targets(data, selection, original):
    fields, merged = [], []
    components = warped = 0
    for i, opts, overlay, eligible in zip(
        selection["indices"],
        selection["augmentation"],
        selection["overlays"],
        np.asarray(original[2]),
        strict=True,
    ):
        field = component_field(data, i, opts)
        components += 1
        warped += int(
            bool(opts["amplitude"] and data["records"][i]["metadata"].get("path_geometry"))
        )
        if overlay["enabled"]:
            assert not eligible
            j, other = overlay["index"], overlay["augmentation"]
            field = np.minimum(field, component_field(data, j, other))
            components += 1
            warped += int(bool(other["amplitude"]))
        fields.append(field)
        merged.append(component_field(data, i, opts, merged=True) if eligible else field)
    targets = tuple(
        jnp.asarray(transform_numpy_batch(np.stack(x)[..., None], selection["transforms"])[..., 0])
        for x in (fields, merged)
    ) + (original[2],)
    assert all(np.isfinite(np.asarray(x)).all() for x in targets)
    return targets, dict(components=components, warped_stroke_components=warped)
