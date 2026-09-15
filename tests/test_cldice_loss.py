import jax
import numpy as np
import pytest
from scipy.ndimage import maximum_filter, minimum_filter

from lineartgen.config import load_config
from lineartgen.training.cldice import centerline_loss, soft_skeleton
from lineartgen.training.comparison import comparison_configs
from lineartgen.training.losses import image_loss


def line():
    ink = np.zeros((1, 40, 40, 1), np.float32)
    ink[:, 4:36, 17:20] = 1
    ink[:, 4:36, 16:17] = 0.25
    ink[:, 4:36, 20:21] = 0.4
    return 1 - 2 * ink


def test_binary_skeleton_and_score_match_morphological_definition():
    ink = (line()[0, ..., 0] < 0).astype(np.float32)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    current, skeleton = ink.copy(), np.zeros_like(ink)
    for _ in range(17):
        eroded = minimum_filter(current, footprint=cross, mode="constant", cval=np.inf)
        opened = maximum_filter(eroded, size=3, mode="constant", cval=-np.inf)
        skeleton = np.maximum(skeleton, np.maximum(current - opened, 0))
        current = eroded
    actual = np.asarray(soft_skeleton(ink[None, ..., None], 16))[0, ..., 0]
    np.testing.assert_array_equal(actual, skeleton)
    assert skeleton.sum() > 0 and np.all(skeleton <= ink)
    predicted = ink.copy()
    predicted[18:22] = 0
    sp = np.asarray(soft_skeleton(predicted[None, ..., None], 16))[0, ..., 0]
    precision = ((sp * ink).sum() + 1) / (sp.sum() + 1)
    recall = ((skeleton * predicted).sum() + 1) / (skeleton.sum() + 1)
    expected = 1 - 2 * precision * recall / (precision + recall)
    actual = centerline_loss(
        (1 - 2 * predicted)[None, ..., None],
        (1 - 2 * ink)[None, ..., None],
        np.ones((1, 40, 40, 1), bool),
        16,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("iterations", [1, 4, 16])
def test_soft_gray_truth_has_zero_loss_and_gradient(iterations):
    # Include weak subpixel lines and arbitrary grayscale occupancies.
    rng = np.random.default_rng(47)
    truth = np.concatenate([line(), 0.8 + 0.2 * line(), rng.uniform(-1, 1, line().shape)], axis=0)
    truth = truth.astype(np.float32)
    active = np.ones_like(truth, bool)
    fn = jax.jit(jax.value_and_grad(lambda x: centerline_loss(x, truth, active, iterations)))
    value, gradient = fn(truth)
    assert float(value) == 0 and np.all(np.asarray(gradient) == 0)
    ink = (1 - truth) / 2
    skel = np.asarray(soft_skeleton(ink, iterations))
    assert np.all(skel >= 0) and np.all(skel <= ink)


def test_cut_bridge_and_faint_line_have_useful_gradients():
    truth = line()
    cut = truth.copy()
    cut[:, 18:22] = 1
    weak = 0.7 + 0.3 * truth
    shifted = np.roll(truth, 3, axis=2)
    missing = np.ones_like(truth)
    gap = truth.copy()
    gap[:, 18:22] = 1
    for predicted, target in [
        (cut, truth),
        (weak, truth),
        (shifted, truth),
        (missing, truth),
        (truth, gap),
    ]:
        fn = jax.jit(
            jax.value_and_grad(
                lambda x, target=target: centerline_loss(x, target, np.ones_like(target, bool), 16)
            )
        )
        value, gradient = fn(predicted)
        assert float(value) > 0 and np.isfinite(np.asarray(gradient)).all()
        assert np.any(np.asarray(gradient) != 0)
        assert float(fn(predicted - 0.1 * gradient)[0]) < float(value)


def test_protected_context_empty_targets_and_per_image_normalization():
    truth, predicted = line(), np.roll(line(), 2, axis=2)
    active = np.ones_like(truth, bool)
    active[:, :20] = False
    fn = jax.jit(jax.value_and_grad(lambda x: centerline_loss(x, truth, active, 16)))
    value, gradient = fn(predicted)
    assert float(value) > 0 and np.all(np.asarray(gradient)[~active] == 0)
    other = predicted.copy()
    other[~active] = -0.9
    assert float(fn(other)[0]) == float(value)
    for target, mask in [
        (truth, np.zeros_like(active)),
        (np.ones_like(truth), active),
        (-np.ones_like(truth), active),
    ]:
        v, g = jax.value_and_grad(
            lambda x, target=target, mask=mask: centerline_loss(x, target, mask, 16)
        )(predicted)
        assert float(v) == 0 and np.all(np.asarray(g) == 0)
    # Adding a blank image halves the contribution rather than diluting its denominator.
    both = centerline_loss(
        np.concatenate([predicted, predicted]),
        np.concatenate([truth, np.ones_like(truth)]),
        np.concatenate([active, active]),
        16,
    )
    np.testing.assert_allclose(both, value / 2, atol=1e-7)


def test_loss_variant_changes_only_auxiliary_and_disabled_path_is_unchanged():
    config = load_config("configs/cldice-loss.toml")
    control, candidate = comparison_configs(config, study="cldice_loss").values()
    assert candidate.loss.boundary_weight == control.loss.boundary_weight == 0
    assert control == candidate.model_copy(update={"loss": control.loss})
    assert control.loss == candidate.loss.model_copy(update={"cldice_weight": 0})
    truth, predicted = line(), 0.5 + 0.5 * line()
    mask = np.full_like(truth, 0.5)
    baseline, parts0 = image_loss(predicted, truth, mask, control.loss, 0.001)
    total, parts1 = image_loss(predicted, truth, mask, candidate.loss, 0.001)
    assert "centerline" not in parts0 and "boundary" not in parts1
    np.testing.assert_allclose(
        total - baseline, candidate.loss.cldice_weight * parts1["centerline"], rtol=1e-5, atol=1e-6
    )
