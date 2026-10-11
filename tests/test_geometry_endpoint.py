import jax
import numpy as np
import pytest
from anchored_endpoint import velocity as endpoint_velocity
from diagnose_learnability import update
from geometry_endpoint_fit import make_arms, run_sample
from region_velocity_fit import run_sample as old_sample
from test_progress_modulation import modulated
from velocity_bridge_toy import sample

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import save_inference
from lineartgen.training import checkpoints, lineart_state


def fixture(root):
    old, m, c, p = modulated(root / "seed")
    old_config = load_config(root / "seed/configs/velocity.json")
    (root / "configs").mkdir()
    for name, model, config in (("plain", old, old_config), ("modulated", m, c)):
        save_config(config, root / f"configs/{name}.json")
        save_inference(model, config, root / "initial" / name, provenance="test")
    return p


def test_actual_updates_are_independent_and_restore_next_update(tmp_path):
    p = fixture(tmp_path)
    arms = make_arms(tmp_path)
    before = jax.tree.map(
        lambda x: np.asarray(x).copy(), lineart_state.payload({"x": arms["modulated"]}, 0)
    )
    a = update(arms["plain"], p)
    jax.tree.map(
        np.testing.assert_array_equal, before, lineart_state.payload({"x": arms["modulated"]}, 0)
    )
    b = update(arms["modulated"], p)
    assert a["gradient_norm"] > 0 and b["gradient_norm"] > 0
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    for name in arms:
        assert update(arms[name], p) == update(restored[name], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )


@pytest.mark.parametrize("method", ["plain", "modulated"])
def test_endpoint_sampler_and_exact_BC_control_replay(tmp_path, method):
    p = fixture(tmp_path)
    model = make_arms(tmp_path)[method]["model"]
    s, t = p["guides"][0], p["levels"][0]
    key = jax.random.PRNGKey(33)
    expected = sample(
        lambda x, t, s, u: endpoint_velocity(model, x, t, s, u, False), s, t, key, depth=2, steps=32
    )
    actual = run_sample(model, s, t, key, 2, method)
    np.testing.assert_array_equal(actual, expected)
    mask = np.asarray(t) == 0
    np.testing.assert_array_equal(actual[mask], np.asarray(s)[mask])
    np.testing.assert_array_equal(
        run_sample(model, s, t, key, 2, "control"), old_sample(model, s, t, key, 2)
    )
