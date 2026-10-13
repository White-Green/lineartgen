import copy

import jax.numpy as jnp
import numpy as np
from merge_family_data import allowed_candidates, merge_allowed, merge_geometry, merged_candidates
from plausible_data import CHOICES, raster
from scipy.ndimage import gaussian_filter
from style_family_objective import style_pixel_costs

from lineartgen.diffusion import pyramid


def fixture(separation=6):
    x = np.linspace(20, 160, 129)
    paths = [
        dict(
            kind="parallel",
            points=np.column_stack((x, np.full_like(x, y))).tolist(),
            widths=np.full_like(x, 2).tolist(),
        )
        for y in (60, 60 + separation)
    ]
    record = dict(
        kind="parallel",
        direction=[0.0, 1.0],
        metadata=dict(paths=2, kinds=["parallel"], path_geometry=paths),
    )
    canonical = np.stack(
        [
            raster(record["metadata"], np.asarray(record["direction"]), offset, 1)
            for offset, _ in CHOICES
        ]
    )
    ink = (1 - canonical[13]) / 2
    clear = 1 - 0.8 * ink
    blurred = 1 - 0.8 * gaussian_filter(ink, (2, 2, 0))
    return record, canonical, clear.astype(np.float32), blurred.astype(np.float32)


def test_only_close_parallel_shapes_are_eligible():
    record, _, _, _ = fixture()
    g = merge_geometry(record)
    assert g["merge_eligible"] and g["merge_valid"].all()
    np.testing.assert_allclose(g["merge_separation"], 6)
    assert not merge_geometry(fixture(10)[0])["merge_eligible"]
    for kind in ("junction", "crossing", "gap", "blank", "black"):
        assert not merge_geometry(record | dict(kind=kind))["merge_eligible"]
    incomplete = copy.deepcopy(record)
    incomplete["metadata"]["path_geometry"].pop()
    assert not merge_geometry(incomplete)["merge_eligible"]


def test_source_support_gate_preserves_clear_gaps_and_blank():
    r, _, clear, blurred = fixture()
    g = merge_geometry(r)
    assert not merge_allowed(clear, g)
    assert merge_allowed(blurred, g)
    assert merge_allowed(1 - (1 - blurred) * 0.5, g)
    assert not merge_allowed(np.ones_like(clear), g)
    assert not merge_allowed(1 - (1 - blurred) * 0.001, g)


def test_original_options_remain_exact_and_merge_is_one_opaque_line():
    r, canonical, clear, blurred = fixture()
    merged, g = merged_candidates(canonical, r)
    assert merged.shape == (9, 256, 256, 1)
    unchanged, allowed = allowed_candidates(canonical, merged, clear, g)
    assert not allowed
    np.testing.assert_array_equal(unchanged, canonical)
    changed, allowed = allowed_candidates(canonical, merged, blurred, g)
    assert allowed
    np.testing.assert_array_equal(changed[::3], canonical[::3])
    np.testing.assert_array_equal(changed[2::3], canonical[2::3])
    np.testing.assert_array_equal(changed[1::3], merged)
    assert not np.array_equal(changed, canonical)
    alpha = (1 - merged[4, :, 100, 0]) / 2
    assert alpha.max() == 1
    assert (alpha > 0.95).sum() >= 1
    assert abs(np.flatnonzero(alpha > 0.95).mean() - 95) < 1
    for k in range(9):
        np.testing.assert_array_equal(merged[k], changed[k * 3 + 1])


def test_gray_intermediate_costs_more_than_either_allowed_crisp_choice():
    r, canonical, _, blurred = fixture()
    merged, g = merged_candidates(canonical, r)
    alternatives, allowed = allowed_candidates(canonical, merged, blurred, g)
    assert allowed
    strengths = pyramid(jnp.full((1, 256, 256, 1), 0.5), 3)
    original = jnp.asarray(canonical[None])
    extended = jnp.asarray(alternatives[None])

    def costs(output, family):
        return np.asarray(
            style_pixel_costs(pyramid(jnp.asarray(output[None]), 3), family, strengths, original)
        )[0]

    assert costs(merged[4], original).min() > 0.01
    assert abs(costs(merged[4], extended).min()) < 1e-7
    assert abs(costs(canonical[13], extended).min()) < 1e-7
    assert costs((merged[4] + canonical[13]) / 2, extended).min() > 0.01
