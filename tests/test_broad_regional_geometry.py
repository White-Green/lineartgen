import jax
import numpy as np
import subpixel_field_study as study
from cue_preserving_data import MODES
from rough_family_data import BASE
from test_focused_geometry import data


def test_full_campaign_covers_each_geometry_at_each_depth():
    counts = np.zeros((512, 6), np.int32)
    modes = np.zeros(len(MODES), np.int32)
    base = 0
    for step in range(study.STEPS):
        selected = study.select(step)
        for i in selected["indices"]:
            counts[i, selected["depth"] - 1] += 1
        for m in selected["modes"]:
            modes[m] += 1
        base += sum(x == BASE for x in selected["augmentation"])
    assert counts.min() > 0
    assert counts.sum() == modes.sum() == study.STEPS * 4
    assert 0.23 < base / counts.sum() < 0.27
    assert modes.min() > counts.sum() * 0.3


def test_full_distribution_batch_matches_across_objectives(monkeypatch):
    arrays = data()
    options = [BASE, BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)] * 2
    # The tiny fixture has 32px inputs; scale only the mask fixture with it.
    original = study.strength_map
    monkeypatch.setattr(study, "strength_map", lambda mode: original(mode)[::8, ::8])
    for depth in range(1, 7):
        monkeypatch.setattr(
            study,
            "select",
            lambda step, depth=depth: dict(
                indices=[0] * 4, modes=[0, 1, 2, 0], depth=depth, augmentation=options
            ),
        )
        a, b = (study.batch(arrays, depth, m) for m in study.METHODS)
        jax.tree.map(np.testing.assert_array_equal, a, b)
        assert len(a[0].guides) == depth
        assert np.any(np.asarray(a[0].noise_levels[0]) == 0)
