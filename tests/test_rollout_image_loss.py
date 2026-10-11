import jax
import jax.numpy as jnp
import numpy as np
from diagnose_learnability import update
from flax import nnx
from rollout_endpoint import rollout_gradient
from rollout_image_fit import split_arms
from rollout_image_loss import gradient, objective
from test_progress_modulation import modulated
from test_rollout_endpoint import TwoTimes

from lineartgen.config import LossConfig
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm


def image_config():
    return LossConfig(
        balance_tones=True,
        tone_balance_power=0.5,
        pixel_loss="interval_l1",
        noise_weight_floor=0.05,
        opacity_deficit_weight=0.3,
        opacity_deficit_preclip=True,
        gap_clearance_weight=0.1,
        paper_band_weight=0.1,
        identity_weight=0.1,
        edge_weight=0.1,
        contrast_weight=0.01,
        contrast_active_only=True,
    )


def test_coefficient_zero_preserves_actual_loss_and_all_gradients(tmp_path):
    _, model, _, p = modulated(tmp_path)
    actual = gradient(model, p, image_config(), 0.001, 0)
    expected = rollout_gradient(model, p)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


def test_added_image_loss_reaches_first_prediction_and_second_prediction():
    x = jnp.zeros((1, 16, 16, 1))
    p = dict(images=(x,), levels=(jnp.ones_like(x),), guides=(x,), target=jnp.full_like(x, 0.5))
    values = []
    for coefficient in (0.0, 0.1):
        values.append(
            nnx.value_and_grad(
                lambda m, c=coefficient: objective(m, p, image_config(), 0.001, c), has_aux=True
            )(TwoTimes())
        )
    assert float(values[1][0][0]) > float(values[0][0][0])
    for key in ("early", "middle"):
        delta = float(values[1][1][key] - values[0][1][key])
        assert delta < -0.001 and np.isfinite(delta)
    parts = values[1][0][1]
    np.testing.assert_allclose(
        parts["loss"], parts["base_loss"] + 0.1 * parts["image_loss"], atol=1e-7
    )


def test_fully_protected_images_have_no_added_error_or_gradient():
    x = jnp.zeros((1, 16, 16, 1))
    p = dict(images=(x,), levels=(x,), guides=(x,), target=x)
    (loss, parts), g = nnx.value_and_grad(
        lambda m: objective(m, p, image_config(), 0.001, 0.1), has_aux=True
    )(TwoTimes())
    assert float(loss) == 0 and float(parts["image_loss"]) == 0
    jax.tree.map(lambda a: np.testing.assert_array_equal(a, 0), g)


def test_same_parent_state_independent_updates_restore_and_next_update(tmp_path):
    _, model, c, p = modulated(tmp_path)
    c = c.model_copy(update={"loss": image_config()})
    original = create_arm(c, model)
    original["gradient"] = rollout_gradient
    update(original, p)
    configs = {m: c for m in ("control", "image")}
    arms = split_arms(original, configs)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), lineart_state.payload({"x": original}, 1))
    for arm in arms.values():
        jax.tree.map(np.testing.assert_array_equal, before, lineart_state.payload({"x": arm}, 1))
    assert update(arms["control"], p) == update(original, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"x": original}, 2),
        lineart_state.payload({"x": arms["control"]}, 2),
    )
    jax.tree.map(
        np.testing.assert_array_equal, before, lineart_state.payload({"x": arms["image"]}, 1)
    )
    stats = update(arms["image"], p)
    assert stats["gradient_norm"] > 0 and np.isfinite(stats["loss"])
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, arms, 2)
        restored = split_arms(original, configs)
        assert lineart_state.restore(saved, restored, 2) == 2
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
    for name in arms:
        assert update(arms[name], p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 3),
        lineart_state.payload(restored, 3),
    )
