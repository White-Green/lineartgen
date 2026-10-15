"""Keep input sketches fixed while asking for uniform opaque stroke widths."""

import copy

import jax
import numpy as np
from field_repair_study import inputs as relative_inputs

AUDITS = []


def uniform_plan(plan):
    out = copy.deepcopy(plan)
    for key in ("scene", "probe"):
        for path in out[key]["metadata"]["path_geometry"]:
            path["widths"] = [2.0] * len(path["points"])
    return out


def inputs(plan, case, *, depth_override=None):
    old = relative_inputs(plan, case, depth_override=depth_override)
    new = relative_inputs(uniform_plan(plan), case, depth_override=depth_override)
    # Prepared clean, guide, injected noise, strength, source and saved reference
    # all stay unchanged. Only geometry supervision and oracle rendering change.
    jax.tree.map(np.testing.assert_array_equal, old[0], new[0])
    for a, b in zip(old[2:5], new[2:5], strict=True):
        np.testing.assert_array_equal(a, b)
    difference = max(
        float(np.max(abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(old[1], new[1], strict=True)
    )
    if case.startswith("probe"):
        jax.tree.map(np.testing.assert_array_equal, old[1], new[1])
        np.testing.assert_array_equal(old[5], new[5])
    else:
        assert difference > 0
    AUDITS.append(
        dict(
            case=case,
            depth=len(new[0].guides),
            prepared_inputs_exact=True,
            probe_teacher_exact=case.startswith("probe"),
            teacher_max_abs_change=difference,
        )
    )
    return new
