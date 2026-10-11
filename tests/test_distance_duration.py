import jax
import numpy as np
from diagnose_learnability import update
from distance_duration import (
    SWITCH,
    TRAINED,
    copy_numerical_state,
    make_arms,
    phase_config,
)
from test_distance_supervision import fixture

from lineartgen.config import save_config
from lineartgen.training import checkpoints, lineart_state


def setup(root):
    _, original, prepared = fixture(root)
    config = original.model_copy(
        update={"train": original.train.model_copy(update={"batch_size": 4})}
    )
    prepared = jax.tree.map(lambda x: jax.numpy.concatenate((x, x)), prepared)
    for method in TRAINED:
        for settled in (False, True):
            phase = "settle" if settled else "constant"
            save_config(phase_config(config, settled), root / f"configs/{method}-{phase}.json")
    batches = {"point": prepared, "margin": dict(prepared)}
    batches["margin"]["margin"] = batches["margin"].pop("distance")
    return batches


def test_phase_change_preserves_nonzero_full_state_and_restored_next_update(tmp_path):
    batches = setup(tmp_path)
    before = make_arms(tmp_path)
    for method, arm in before.items():
        for _ in range(3):
            assert update(arm, batches[method])["gradient_norm"] > 0
        assert int(arm["optimizer"].step[...]) == 3
        np.testing.assert_allclose([arm["schedule"](s) for s in (1200, 5999)], 1e-4)
    after = make_arms(tmp_path, SWITCH + 1)
    copy_numerical_state(before, after, 3)
    for arm in after.values():
        np.testing.assert_allclose([arm["schedule"](s) for s in (6000, 7199)], 1e-5)
        assert int(arm["optimizer"].step[...]) == 3
    with checkpoints.manager(tmp_path / "full-state") as saved:
        lineart_state.save(saved, after, 3)
        restored = make_arms(tmp_path, SWITCH + 1)
        assert lineart_state.restore(saved, restored, 3) == 3
    for method in TRAINED:
        assert update(after[method], batches[method]) == update(restored[method], batches[method])
        assert int(after[method]["optimizer"].step[...]) == 4
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(after, 4),
        lineart_state.payload(restored, 4),
    )
