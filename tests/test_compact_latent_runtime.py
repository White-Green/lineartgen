"""Regression checks for the compact pixel editor exported with the Krita model."""

import numpy as np
import pytest
from lineartgen_runtime.latent_compact_merge import compact_merge
from lineartgen_runtime.latent_context import Components, roots
from lineartgen_runtime.latent_edit import PixelEdit
from lineartgen_runtime.latent_painted_loops import loop_features
from scipy.ndimage import label, maximum_filter, uniform_filter


@pytest.mark.parametrize("diagonal", [True, False])
def test_component_order_matches_first_raster_pixel(diagonal):
    rng = np.random.default_rng(160)
    for density in (0, 0.08, 0.5, 0.9, 1):
        mask = rng.random((31, 43)) < density
        ref, n = label(mask, np.ones((3, 3)) if diagonal else None)
        group = Components(mask, diagonal)
        expected = np.where(mask, ref - 1, n)
        np.testing.assert_array_equal(group.ids, expected)
        np.testing.assert_array_equal(group.count, np.bincount(expected.ravel()))
        first = np.full(n + 1, mask.size)
        np.minimum.at(first, expected.ravel(), np.arange(mask.size))
        first[n] = mask.size
        np.testing.assert_array_equal(roots(mask, diagonal), first[expected])


def test_sparse_tiled_features_match_dense_reference():
    h, w = 103, 177
    rng = np.random.default_rng(590)
    rough = np.zeros((h, w, 4), np.uint8)
    rough[16:85, 20:161, 3] = rng.integers(1, 255, (69, 141), dtype=np.uint8)
    mark = np.zeros_like(rough)
    mark[49:55, 78:90, 3] = 178
    edit = PixelEdit(rough.tobytes(), mark.tobytes(), h, w, 0.4)
    proposal = np.roll(edit.rough, 3, axis=1)
    peaks = [maximum_filter(edit.rough, k, mode="constant") for k in (3, 7, 15)]
    ctx = [maximum_filter(proposal, 3), uniform_filter(proposal, 7, mode="constant")]
    ctx.extend(
        maximum_filter(x, k, mode="constant")
        for x, k in ((edit.coverage, 9), (edit.coverage, 31), (edit.alpha, 31))
    )
    wide = np.pad(maximum_filter(edit.coverage, 65, mode="constant"), 32)
    ctx.extend(wide[y : y + h, x : x + w] for y, x in ((0, 32), (64, 32), (32, 0), (32, 64)))
    channels = np.concatenate(
        (
            edit.features(proposal),
            np.stack(peaks, -1),
            loop_features(edit.rough, edit.coverage, tails=True),
            np.stack(ctx, -1),
        ),
        -1,
    )
    valid = (ctx[0] > 1 / 65535) | (edit.rough > 0)

    # Check every column and original pixel position, including regions outside sparse ink.
    class PickChannel:
        def __init__(self, index):
            self.index = index

        def run(self, outputs, feeds):
            return [feeds["x"][:, self.index : self.index + 1]]

    for i in range(channels.shape[-1]):
        result = compact_merge(edit, proposal, PickChannel(i), tile=37)
        np.testing.assert_allclose(result, np.where(valid, channels[..., i], 0), atol=2e-7)
