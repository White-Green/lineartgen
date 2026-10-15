import numpy as np
from continuous_width_data import width_choice
from contrast_composition_data import SINGLES, overlay_choice, overlay_source
from rough_family_data import augmentation


def test_translucent_composition_preserves_strokes_and_darkens_overlap():
    first = np.ones((8, 8, 1), np.float32)
    second = first.copy()
    first[3:5, :, :] = 0.2
    second[:, 3:5, :] = 0.6
    result = overlay_source(first, second)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert result.min() >= -1 and result.max() <= 1
    np.testing.assert_allclose(result[3, 1], first[3, 1], atol=1e-7)
    np.testing.assert_allclose(result[1, 3], second[1, 3], atol=1e-7)
    np.testing.assert_allclose(result[3, 3], -0.04, atol=1e-7)
    assert result[3, 3, 0] < min(first[3, 3, 0], second[3, 3, 0])
    np.testing.assert_array_equal(result[0, 0], [1])
    np.testing.assert_allclose(result, overlay_source(second, first), atol=1e-7)


def test_overlay_choice_is_independent_and_guards_multiline_and_fill():
    seed = 2026101502
    plans = [overlay_choice(seed, k, 0, 10) for k in range(1000)]
    assert 430 < sum(p["enabled"] for p in plans) < 570
    assert len(SINGLES) == 288
    assert len({p["augmentation"]["opacity"] for p in plans}) > 500
    for k, plan in enumerate(plans):
        assert plan == overlay_choice(seed, k, 0, 10)
        assert plan["index"] in SINGLES
        assert plan["augmentation"] == augmentation(seed, k, 4)
        assert plan["width"] == width_choice(seed, k, 4, plan["index"])
    for i in range(192, 384):
        assert not overlay_choice(seed, i, 0, i)["enabled"]
    for i in range(432, 464):
        assert not overlay_choice(seed, i, 0, i)["enabled"]
