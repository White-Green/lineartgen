import jax
import jax.numpy as jnp
import numpy as np

from lineartgen.config import GeometryDataConfig
from lineartgen.data.geometry import draw_geometry
from lineartgen.training.lineart import Profiles, line_region_loss, make_profiles, profile_loss


def sample():
    truth = np.ones((1, 48, 48, 1), np.float32)
    truth[:, 5:43, 23:25] = -1
    truth[:, 5:43, 22] = 0.5
    truth[:, 5:43, 25] = 0.5
    coordinates = np.array([[[[24, x] for x in np.linspace(14, 33, 49)]]], np.float32)
    return truth, Profiles(coordinates, np.ones((1, 1), np.float32))


def test_geometry_metadata_does_not_change_generated_pixels_or_random_stream():
    config = GeometryDataConfig()
    for kind in ("straight", "bezier", "parallel", "taper", "gap", "junction"):
        a, metadata = draw_geometry(np.random.default_rng(12), 64, config, kind=kind)
        b, paths = draw_geometry(
            np.random.default_rng(12), 64, config, kind=kind, include_paths=True
        )
        np.testing.assert_array_equal(a, b)
        assert metadata == {k: v for k, v in paths.items() if k != "path_geometry"}
        profiles = make_profiles([dict(paths, item=0)], 1, 64, seed=6, step=3)
        assert profiles.valid.sum() > 0 if kind != "junction" else profiles.valid.sum() == 0


def test_exact_soft_target_and_protected_pixels_have_zero_auxiliary_gradient():
    truth, profiles = sample()
    active = np.ones_like(truth, bool)
    active[:, :25] = False
    for loss in (
        lambda x: line_region_loss(x, truth, active),
        lambda x: profile_loss(x, truth, active, profiles),
    ):
        value, gradient = jax.value_and_grad(loss)(truth)
        assert float(value) == 0
        np.testing.assert_array_equal(gradient, 0)
        prediction = truth.copy()
        prediction[:, 20:30, 20:28] = -0.1
        value, gradient = jax.value_and_grad(loss)(prediction)
        assert np.isfinite(float(value))
        assert np.all(np.asarray(gradient)[~active] == 0)
        np.testing.assert_array_equal(loss(np.where(active, prediction, -0.9)), value)


def test_halo_and_line_erasure_both_cost_loss_with_improving_gradients():
    truth, profiles = sample()
    halo = truth.copy()
    halo[:, 5:43, 18:22] = 0.5
    halo[:, 5:43, 26:30] = 0.5
    for prediction in (halo, np.ones_like(truth), np.roll(truth, 3, axis=2)):
        for loss in (
            lambda x: line_region_loss(x, truth, jnp.ones_like(x, bool)),
            lambda x: profile_loss(x, truth, jnp.ones_like(x, bool), profiles),
        ):
            value, gradient = jax.value_and_grad(loss)(prediction)
            assert float(value) > 0 and np.isfinite(np.asarray(gradient)).all()
            assert float(loss(prediction - 0.01 * gradient)) < float(value)


def test_blank_and_absent_profiles_are_finite_and_fully_protected_loss_is_zero():
    truth, profiles = sample()
    empty = profiles._replace(valid=np.zeros_like(profiles.valid))
    assert float(profile_loss(-truth, truth, np.ones_like(truth, bool), empty)) == 0
    for value in (-1.0, 1.0):
        target = np.full_like(truth, value)
        loss = line_region_loss(-target, target, np.zeros_like(truth, bool))
        assert float(loss) == 0
        assert np.isfinite(float(line_region_loss(-target, target, np.ones_like(truth, bool))))
