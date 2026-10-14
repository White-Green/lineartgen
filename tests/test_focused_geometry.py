import focused_geometry_study as study
import jax
import numpy as np
from rough_family_data import BASE
from subpixel_field_data import central_field
from subpixel_field_model import coverage
from test_focused_fit import small_data
from test_subpixel_field import fixture


def data():
    arrays = small_data()
    _, field = fixture()
    return dict(
        source=arrays["source"],
        variants=arrays["variants"],
        fields=np.asarray(field),
        directions=np.array([[0, 1]], np.float32),
        clean=np.asarray(coverage(central_field(field)[..., None])),
    )


def test_fixed_augmented_only_change_input_not_geometry(monkeypatch):
    options = BASE | dict(opacity=0.2, sigma=2.4, amplitude=1.5)
    monkeypatch.setattr(
        study, "select", lambda step: dict(indices=[0] * 4, depth=3, augmentation=[options] * 4)
    )
    arrays = data()
    fixed = study.batch(arrays, 0, "fixed")
    augmented = study.batch(arrays, 0, "augmented")
    jax.tree.map(np.testing.assert_array_equal, fixed[1:], augmented[1:])
    jax.tree.map(np.testing.assert_array_equal, fixed[0].targets, augmented[0].targets)
    jax.tree.map(np.testing.assert_array_equal, fixed[0].noise_levels, augmented[0].noise_levels)
    np.testing.assert_array_equal(fixed[0].guides[0], np.repeat(arrays["source"], 4, 0))
    assert np.max(abs(np.asarray(fixed[0].guides[0] - augmented[0].guides[0]))) > 0.05
    assert fixed[1].shape == (4, 160, 160)


def test_base_sources_are_exactly_shared(monkeypatch):
    monkeypatch.setattr(
        study, "select", lambda step: dict(indices=[0] * 4, depth=6, augmentation=[BASE] * 4)
    )
    arrays = data()
    jax.tree.map(
        np.testing.assert_array_equal,
        study.batch(arrays, 0, "fixed"),
        study.batch(arrays, 0, "augmented"),
    )


def test_all_six_depths_cover_each_of_the_sixteen_geometries():
    counts = np.zeros((16, 6), np.int32)
    for step in range(study.STEPS):
        selected = study.select(step)
        for index in selected["indices"]:
            counts[index, selected["depth"] - 1] += 1
    assert counts.min() > 70
    assert counts.sum() == 12000
