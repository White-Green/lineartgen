import focused_fit_study as study
import jax
import numpy as np
from rough_family_data import BASE, rough_source
from test_path_profile import annotation


def small_data():
    clean = np.ones((1, 32, 32, 1), np.float32)
    clean[:, :, 12:20] = -1
    styled = np.ones_like(clean)
    styled[:, :, 15:17] = -1
    return dict(
        source=np.stack([rough_source(clean[0], BASE)]),
        variants=np.repeat(clean[:, None], 27, 1),
        allstroke=np.repeat(styled[:, None], 27, 1),
        **annotation(),
    )


def test_only_source_varies_and_clean_style_denominator_is_retained(monkeypatch):
    data = small_data()
    options = BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)
    monkeypatch.setattr(
        study, "select", lambda step: dict(indices=[0] * 4, depth=3, augmentation=[options] * 4)
    )
    fixed = study.batch(data, 0, "fixed")
    augmented = study.batch(data, 0, "augmented")
    jax.tree.map(np.testing.assert_array_equal, fixed[1:], augmented[1:])
    jax.tree.map(np.testing.assert_array_equal, fixed[0].noise_levels, augmented[0].noise_levels)
    assert len(fixed[0].guides) == 3
    np.testing.assert_array_equal(fixed[0].guides[0], np.repeat(data["source"], 4, 0))
    assert np.max(abs(np.asarray(fixed[0].guides[0] - augmented[0].guides[0]))) > 0.05
    np.testing.assert_array_equal(fixed[2]["normalizers"], fixed[1])


def test_base_augmentation_is_exact_fixed_replay(monkeypatch):
    monkeypatch.setattr(
        study, "select", lambda step: dict(indices=[0] * 4, depth=6, augmentation=[BASE] * 4)
    )
    data = small_data()
    jax.tree.map(
        np.testing.assert_array_equal,
        study.batch(data, 0, "fixed"),
        study.batch(data, 0, "augmented"),
    )


def test_all_cases_receive_all_depths_without_source_condition_selection_bias():
    counts = np.zeros((16, 6), np.int32)
    for step in range(study.STEPS):
        s = study.select(step)
        for i in s["indices"]:
            counts[i, s["depth"] - 1] += 1
    assert counts.min() > 70
    assert counts.sum() == 12000
