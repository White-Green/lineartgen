import continuous_width_data as width_data
import geometry_coverage_study as control
import jax
import numpy as np
import width_coverage_study as study
from expanded_family_data import make_case
from rough_family_data import BASE


def fixture():
    data, record = make_case("expanded", 58, 2)
    arrays = {
        k: np.broadcast_to(data[k][None], (512,) + data[k].shape) for k in ("source", "variants")
    }
    arrays["fields"] = np.broadcast_to(np.zeros((1, 160, 160), np.float16), (512, 160, 160))
    arrays["directions"] = np.broadcast_to(data["direction"][None], (512, 2))
    arrays["records"] = [record] * 512
    return arrays, record


def test_original_width_reproduces_source_and_data_stays_immutable():
    arrays, record = fixture()
    original = arrays["source"][464].copy()
    import copy

    before = copy.deepcopy(record)
    peak = max(record["metadata"]["path_geometry"][0]["widths"])
    np.testing.assert_array_equal(
        width_data.width_source(arrays, record, 464, peak, BASE), original
    )
    changed = width_data.width_source(arrays, record, 464, 8.0, BASE)
    assert np.max(abs(changed - original)) > 0.1
    np.testing.assert_array_equal(arrays["source"][464], original)
    assert record == before


def test_only_source_changes_with_identical_geometry_masks_and_rng(monkeypatch):
    arrays, _ = fixture()

    def selected(step):
        return dict(
            indices=[464, 0, 465, 1],
            companions=[0] * 4,
            compose=[False] * 4,
            modes=[0, 1, 2, 0],
            depth=step + 1,
            transforms=[0, 1, 4, 7],
            augmentation=[BASE] * 4,
        )

    monkeypatch.setattr(study, "select", selected)
    monkeypatch.setattr(control, "select", selected)
    monkeypatch.setattr(
        study, "width_choice", lambda seed, step, item, i: 8.0 if i >= 464 else None
    )
    for step in range(6):
        a = study.batch(arrays, step, "width")
        b = control.batch(arrays, step, "control")
        jax.tree.map(np.testing.assert_array_equal, a[1:], b[1:])
        for ta, tb, mask, ga, gb in zip(
            a[0].targets, b[0].targets, a[0].noise_levels, a[0].guides, b[0].guides, strict=True
        ):
            active = np.asarray(mask) > 0
            np.testing.assert_array_equal(np.asarray(ta)[active], np.asarray(tb)[active])
            np.testing.assert_array_equal(np.asarray(ta)[~active], np.asarray(ga)[~active])
            np.testing.assert_array_equal(np.asarray(tb)[~active], np.asarray(gb)[~active])
        jax.tree.map(np.testing.assert_array_equal, a[0].noise_levels, b[0].noise_levels)
        jax.tree.map(
            lambda x, y: np.testing.assert_array_equal(
                np.asarray(x)[[1, 3]], np.asarray(y)[[1, 3]]
            ),
            a,
            b,
        )
        assert np.any(np.asarray(a[0].guides[0]) != np.asarray(b[0].guides[0]))
        for pa, pb, ga, gb in zip(a[0].noisy, b[0].noisy, a[0].guides, b[0].guides, strict=True):
            np.testing.assert_allclose(pa - ga, pb - gb, atol=1.2e-7, rtol=0)


def test_width_randomness_does_not_change_original_sampling_and_fills_gap():
    selected = []
    kept = 0
    for step in range(study.STEPS):
        assert study.select(step) == control.select(step)
        for item, i in enumerate(study.select(step)["indices"]):
            width = width_data.width_choice(study.SEED, step, item, i)
            if i < 464:
                assert width is None
            elif width is None:
                kept += 1
            else:
                assert 4 <= width <= 32
                selected.append(width)
    assert 0.22 < kept / (kept + len(selected)) < 0.28
    assert sum(5 < w < 16 for w in selected) > len(selected) * 0.45
