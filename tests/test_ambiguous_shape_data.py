import numpy as np
from ambiguous_shape_data import permitted_outputs, record, render, selected, validate
from procedural_shape_data import selected as long_selected


def test_recipes_and_disjoint_depth_schedule():
    steps = [s for s in range(6000) if selected(s)]
    assert len(steps) == 750
    assert [sum(s % 6 == d for s in steps) for d in range(6)] == [125] * 6
    assert not any(long_selected(s) for s in steps)
    for step in steps:
        for item in range(4):
            r = record(step, item)
            validate(r)
            assert r == record(step, item)


def test_canonical_square_support_two_axes_and_partial_gating():
    r = record(30, 0) | dict(center=[128, 128], width=48, length=48, angle=0, square=True)
    a = render(r)
    b = render(r)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    # Separable float32 Gaussian filtering can differ by one ULP across axes.
    np.testing.assert_allclose(a["source"], np.rot90(a["source"]), atol=1e-7, rtol=0)
    np.testing.assert_array_equal(a["clean"], np.rot90(a["axis_clean"]))
    np.testing.assert_allclose(np.dot(a["direction"], a["axis_direction"]), 0, atol=1e-7)
    mask = np.full_like(a["source"], 0.5)
    p, q, allowed = permitted_outputs(a, mask)
    assert allowed and not np.array_equal(p, q)
    assert np.min(p) == np.min(q) == -1
    mask[:, :128] = 0
    p, q, allowed = permitted_outputs(a, mask)
    assert not allowed
    np.testing.assert_array_equal(p, q)
    np.testing.assert_array_equal(p[mask == 0], a["source"][mask == 0])
