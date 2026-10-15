import numpy as np
from aligned_field_inputs import aligned_plan, inverse_warp
from rough_family_data import rough_source


def test_inverse_geometry_matches_actual_image_sampler_pixel_centers():
    size = 64
    yy, xx = np.indices((size, size), dtype=np.float32)
    options = dict(
        opacity=1.0,
        sigma=0.0,
        amplitude=1.5,
        phase_x=0.9064910245618603,
        phase_y=1.998081939496715,
        period=96.0,
    )
    sampled_x = rough_source((2 * xx / (size - 1) - 1)[..., None], options)[..., 0]
    sampled_y = rough_source((2 * yy / (size - 1) - 1)[..., None], options)[..., 0]
    source_positions = np.stack(
        ((sampled_x + 1) * (size - 1) / 2 + 0.5, (sampled_y + 1) * (size - 1) / 2 + 0.5), -1
    )[4:-4, 4:-4]
    expected = np.stack((xx + 0.5, yy + 0.5), -1)[4:-4, 4:-4]
    np.testing.assert_allclose(inverse_warp(source_positions, options), expected, atol=1e-5, rtol=0)


def test_margin_shared_junctions_and_zero_warp_are_preserved():
    options = dict(amplitude=1.5, phase_x=0.9, phase_y=2.0, period=96.0)
    metadata = dict(
        path_geometry=[
            dict(points=[[10, 20], [30, 40]], widths=[4, 4]),
            dict(points=[[30, 40], [50, 60]], widths=[1, 1]),
        ]
    )
    plan = dict(scene=dict(metadata=metadata, options=options), probe=dict(metadata=metadata))
    actual = aligned_plan(plan)["scene"]["metadata"]["path_geometry"]
    np.testing.assert_array_equal(actual[0]["points"][-1], actual[1]["points"][0])
    p = np.asarray(metadata["path_geometry"][0]["points"], np.float64)
    np.testing.assert_array_equal(
        np.asarray(actual[0]["points"]), inverse_warp(p + 32, options) - 32
    )
    assert actual[0]["widths"] == [2, 2]
    assert metadata["path_geometry"][0]["widths"] == [4, 4]
    np.testing.assert_array_equal(inverse_warp(p, options | dict(amplitude=0)), p)
