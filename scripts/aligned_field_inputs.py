"""Follow the input's smooth warp with opaque vector geometry teachers."""

import jax
import numpy as np
from field_repair_study import inputs as original_inputs
from uniform_field_inputs import uniform_plan

AUDITS = []


def inverse_warp(points, options):
    points = np.asarray(points, np.float64)
    if not options["amplitude"]:
        return points.copy()
    amplitude, frequency = options["amplitude"], 2 * np.pi / options["period"]
    assert amplitude * frequency < 0.25
    phase = np.asarray([options["phase_x"], options["phase_y"]])
    result = points.copy()
    # rough_source maps output pixel indices to input pixel indices. Geometry
    # uses pixel-edge coordinates, so its index coordinates are q-0.5.
    for _ in range(32):
        result = points - amplitude * np.sin((result[..., ::-1] - 0.5) * frequency + phase)
    residual = result + amplitude * np.sin((result[..., ::-1] - 0.5) * frequency + phase) - points
    assert np.max(abs(residual), initial=0) < 1e-9
    assert np.max(abs(result - points), initial=0) <= amplitude + 1e-9
    return result


def aligned_plan(plan):
    out = uniform_plan(plan)
    for path in out["scene"]["metadata"]["path_geometry"]:
        physical = np.asarray(path["points"], np.float64) + 32
        path["points"] = (inverse_warp(physical, out["scene"]["options"]) - 32).tolist()
    return out


def inputs(plan, case, *, depth_override=None):
    old = original_inputs(uniform_plan(plan), case, depth_override=depth_override)
    new = original_inputs(aligned_plan(plan), case, depth_override=depth_override)
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
