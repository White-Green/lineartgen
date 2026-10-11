import jax
import jax.numpy as jnp
import numpy as np
from diagnose_learnability import update
from endpoint_image_velocity_fit import make_arms
from endpoint_image_velocity_objective import endpoint_image_objective, start_image_loss
from flax import nnx
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import objective

from lineartgen.config import LossConfig, load_config, save_config
from lineartgen.training import checkpoints, lineart_state


def loss_config():
    return LossConfig(
        pixel_loss="interval_l1",
        balance_tones=True,
        tone_balance_power=0.5,
        opacity_deficit_weight=0.3,
        opacity_deficit_preclip=True,
        gap_clearance_weight=0.1,
        paper_band_weight=0.1,
        identity_weight=0.1,
        edge_weight=0.1,
        contrast_weight=0.01,
        contrast_active_only=True,
        noise_weight_floor=0.05,
    )


def test_start_only_gating_batch_normalization_and_inward_image_gradients():
    y = jnp.ones((2, 32, 32, 1)).at[1].set(-1)
    prediction = y * 0.6
    p = dict(target=y, levels=(jnp.full_like(y, 0.5),), progress=(jnp.ones_like(y).at[1].set(0.5),))

    def fn(x):
        return start_image_loss(x, p, loss_config(), 0.001)[0]

    value, g = jax.value_and_grad(fn)(prediction)
    q = dict(p, target=y[:1], levels=(p["levels"][0][:1],), progress=(p["progress"][0][:1],))
    one = start_image_loss(prediction[:1], q, loss_config(), 0.001)[0]
    np.testing.assert_allclose(value, one / 2)
    assert np.all(np.asarray(g[0]) < 0)
    np.testing.assert_array_equal(g[1], 0)
    both = dict(p, progress=(jnp.ones_like(y),))
    _, g = jax.value_and_grad(lambda x: start_image_loss(x, both, loss_config(), 0.001)[0])(
        prediction
    )
    assert np.all(np.asarray(g[1]) > 0)


def test_coefficient_zero_and_non_start_preserve_original_loss_and_gradients(tmp_path):
    model, p, _ = fixture(tmp_path)

    def original(m):
        return objective(m, p)[0]

    a, ga = nnx.value_and_grad(original)(model)
    b, gb = nnx.value_and_grad(
        lambda m: endpoint_image_objective(m, p, loss_config(), 0.001, 0)[0]
    )(model)
    np.testing.assert_array_equal(a, b)
    jax.tree.map(np.testing.assert_array_equal, ga, gb)
    p = dict(p, progress=tuple(jnp.full_like(x, 0.5) for x in p["progress"]))
    a, ga = nnx.value_and_grad(lambda m: objective(m, p)[0])(model)
    b, gb = nnx.value_and_grad(lambda m: endpoint_image_objective(m, p, loss_config(), 0.001)[0])(
        model
    )
    np.testing.assert_array_equal(a, b)
    jax.tree.map(np.testing.assert_array_equal, ga, gb)


def test_actual_endpoint_image_update_and_full_restore(tmp_path):
    _, _, direct = fixture(tmp_path)
    p = dict(direct, velocity_target=direct["target"] - direct["images"][0])
    c = load_config(tmp_path / "configs/velocity.json").model_copy(update={"loss": loss_config()})
    save_config(c, tmp_path / "configs/endpoint_image.json")
    arms = make_arms(tmp_path)
    m = update(arms["endpoint_image"], p)
    assert m["gradient_norm"] > 0 and m["weighted_start_image"] > 0 and m["start_fraction"] == 1
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["endpoint_image"], p) == update(restored["endpoint_image"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
