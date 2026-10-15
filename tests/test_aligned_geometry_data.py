import numpy as np
from aligned_geometry_data import warped_metadata
from rough_family_data import rough_source


def test_broad_teacher_preserves_taper_and_matches_source_coordinates():
    yy, xx = np.indices((128, 128), dtype=np.float32)
    opts = dict(opacity=1.0, sigma=0.0, amplitude=1.5, phase_x=0.4, phase_y=2.1, period=64.0)
    sampled = np.stack(
        [
            (rough_source((2 * x / 127 - 1)[..., None], opts)[..., 0] + 1) * 127 / 2 + 0.5
            for x in (xx, yy)
        ],
        -1,
    )[32:96, 64]
    widths = np.linspace(0.2, 2, len(sampled)).tolist()
    original = dict(path_geometry=[dict(points=(sampled - 32).tolist(), widths=widths)])
    result = warped_metadata(original, opts)["path_geometry"][0]
    np.testing.assert_allclose(
        np.asarray(result["points"]) + 32,
        np.stack((xx + 0.5, yy + 0.5), -1)[32:96, 64],
        # Four float32 ULPs near 128px cover ramp encode/sample/decode rounding.
        atol=4 * np.spacing(np.float32(128)),
        rtol=0,
    )
    assert result["widths"] == widths
    assert original["path_geometry"][0]["points"] == (sampled - 32).tolist()
