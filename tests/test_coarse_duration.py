import jax
import jax.numpy as jnp
import numpy as np
from coarse_duration_study import HI, START, STEPS, cycle_index
from cue_capacity import make_arm
from flax import nnx
from learned_coarse_axis_model import load_field
from plausible_study import exact

from lineartgen.training import checkpoints


def test_cycles_preserve_order_depth_and_boundaries():
    for step in range(START, STEPS):
        assert cycle_index(step) == step - (step // START) * START
        assert cycle_index(step) % 6 == step % 6
    assert [cycle_index(s) for s in (6000, 11999, 12000, 17999)] == [0, 5999, 0, 5999]


def test_original_schedule_stays_at_floor_without_restart():
    model, cfg = load_field(HI / "snapshots/step-6000/consistent/raw")
    arm = make_arm(cfg, model)
    assert cfg.train.peak_learning_rate == 1e-4 * np.sqrt(cfg.train.batch_size)
    actual = np.asarray(arm["schedule"](jnp.arange(6000, 18001)))
    np.testing.assert_allclose(actual, 2e-5, atol=1e-11, rtol=0)
    assert float(arm["schedule"](0)) > 9 * float(actual[0])


def test_full_optimizer_restore_preserves_next_update_not_only_parameters():
    model, cfg = load_field(HI / "snapshots/step-6000/consistent/raw")
    arm = make_arm(cfg, model)
    for _ in range(3):
        grad = jax.tree.map(lambda v: jnp.ones_like(v) * 0.01, nnx.state(model, nnx.Param))
        arm["optimizer"].update(model, grad)
    saved = checkpoints.payload(model, arm["optimizer"], arm["ema"], 3)
    other, _ = load_field(HI / "snapshots/step-6000/consistent/raw")
    resumed = make_arm(cfg, other)
    checkpoints.restore_into(other, saved["model"])
    checkpoints.restore_into(resumed["optimizer"], saved["optimizer"])
    assert int(resumed["optimizer"].step) == 3
    for a in (arm, resumed):
        grad = jax.tree.map(lambda v: jnp.ones_like(v) * 0.02, nnx.state(a["model"], nnx.Param))
        a["optimizer"].update(a["model"], grad)
    exact(nnx.state(model), nnx.state(other))
    exact(nnx.state(arm["optimizer"]), nnx.state(resumed["optimizer"]))
