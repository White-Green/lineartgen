"""Cache one opaque center path for geometrically eligible parallel pairs."""

import copy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, write
from opaque_geometry import geometry
from subpixel_field_data import geometry_field
from support_merge_data import merge_geometry


def merged_metadata(record):
    assert bool(merge_geometry(record)["merge_eligible"])
    result = copy.deepcopy(record["metadata"])
    a, b = result["path_geometry"]
    points = (np.asarray(a["points"]) + np.asarray(b["points"])) / 2
    result["path_geometry"] = [
        dict(kind="bezier", points=points.tolist(), widths=np.full(len(points), 2.0).tolist())
    ]
    result["paths"] = 1
    return result


def build_cache(root, records):
    selected = [i for i, r in enumerate(records) if bool(merge_geometry(r)["merge_eligible"])]
    fields = []
    maximum = 0.0
    for i in selected:
        args = tuple(jnp.asarray(x) for x in geometry(merged_metadata(records[i])))
        f = np.asarray(geometry_field(*args, size=256))
        cached = f.astype(np.float16)
        maximum = max(
            maximum, float(abs(cached.astype(np.float32)[abs(f) < 8] - f[abs(f) < 8]).max())
        )
        fields.append(cached)
    assert selected
    stacked = np.stack(fields)
    assert np.isfinite(stacked).all()
    legacy = Path("../jax-merge-geometry-20261015/runs/merge-geometry-20261015")
    old_indices = read(legacy / "merge-cache.json")["indices"]
    old_fields = np.load(legacy / "merged-fields.npy", mmap_mode="r")
    for j, i in enumerate(old_indices):
        np.testing.assert_array_equal(stacked[selected.index(i)], old_fields[j])
    np.save(root / "merged-fields.npy", stacked)
    write(
        root / "merge-cache.json",
        dict(
            indices=selected,
            near8_max_abs_error=maximum,
            old_fields_exact=True,
            old_fields=len(old_indices),
        ),
    )
