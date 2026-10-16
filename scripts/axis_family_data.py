"""A perpendicular thin interpretation only for isolated, short broad sources."""

import copy

import jax.numpy as jnp
import numpy as np
from aligned_geometry_data import warped_metadata
from d4_geometry_data import transform_directions, transform_numpy_batch
from opaque_geometry import geometry
from subpixel_field_data import geometry_field

MINIMUM_WIDTH_LENGTH_RATIO = 0.75


def axis_spec(record, width, *, overlay=False):
    if record["kind"] != "wide" or overlay:
        return dict(allowed=False, reason="not isolated wide")
    paths = record["metadata"]["path_geometry"]
    assert len(paths) == 1
    points = np.asarray(paths[0]["points"], np.float64)
    assert len(points) >= 2
    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    width = float(width if width is not None else max(paths[0]["widths"]))
    assert length > 0 and width > 0
    ratio = width / length
    normal = np.array([-delta[1], delta[0]]) / length
    center = (points[0] + points[-1]) * 0.5
    np.testing.assert_allclose(points, np.linspace(points[0], points[-1], len(points)), atol=1e-6)
    assert np.ptp(paths[0]["widths"]) < 1e-6
    return dict(
        allowed=ratio >= MINIMUM_WIDTH_LENGTH_RATIO,
        width=width,
        length=length,
        ratio=ratio,
        points=np.linspace(
            center - 0.5 * width * normal, center + 0.5 * width * normal, len(points)
        ).tolist(),
        direction=(-delta / length).tolist(),
    )


def axis_metadata(record, spec, options):
    assert spec["allowed"]
    metadata = copy.deepcopy(record["metadata"])
    metadata["path_geometry"] = [
        dict(kind="straight", points=spec["points"], widths=[2.0] * len(spec["points"]))
    ]
    return warped_metadata(metadata, options)


def axis_payload(data, selection, width_rows, original, directions):
    bases, normals, records = [], [], []
    for item, (index, options, overlay, code) in enumerate(
        zip(
            selection["indices"],
            selection["augmentation"],
            selection["overlays"],
            selection["transforms"],
            strict=True,
        )
    ):
        record = data["records"][index]
        spec = axis_spec(record, width_rows[item]["primary"]["new"], overlay=overlay["enabled"])
        if spec["allowed"]:
            assert not bool(np.asarray(original[2])[item])
            metadata = axis_metadata(record, spec, options)
            args = tuple(jnp.asarray(x) for x in geometry(metadata, kind="wide", size=256))
            base = np.asarray(geometry_field(*args, size=256)).astype(np.float16)
            base = transform_numpy_batch(base[None, ..., None], [code])[0, ..., 0]
            normal = np.asarray(
                transform_directions(jnp.asarray([spec["direction"]]), jnp.asarray([code]))
            )[0]
        else:
            base = np.asarray(original[0])[item]
            normal = np.asarray(directions)[item]
        bases.append(base)
        normals.append(normal)
        records.append(dict(item=item, index=index, **spec))
    return (
        jnp.asarray(np.stack(bases)),
        jnp.asarray(np.stack(normals)),
        jnp.asarray([r["allowed"] for r in records]),
    ), records
