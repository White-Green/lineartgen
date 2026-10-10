import jax
import jax.numpy as jnp
import numpy as np
import pytest
from boundary_time_bridge_toy import batch, redistribute_time
from transport_bridge_toy import batch as old_batch

from lineartgen.diffusion import pyramid


def audit_batch(step, depth):
    p, labels, tau = batch(step, depth)
    q, old_labels, old_tau = old_batch(step, depth)
    np.testing.assert_array_equal(labels, old_labels)
    jax.tree.map(np.testing.assert_array_equal, p["direct"], q["direct"])
    for field in ("target", "levels", "guides"):
        jax.tree.map(np.testing.assert_array_equal, p["bridge"][field], q["bridge"][field])
    np.testing.assert_allclose(
        tau, np.sin(np.pi * np.asarray(old_tau, dtype=np.float64) / 2) ** 2, atol=1.5e-7, rtol=0
    )
    assert np.all(np.asarray(tau)[np.asarray(old_tau) == 0] == 0)
    assert np.all((np.asarray(tau) >= 0) & (np.asarray(tau) <= 1))
    clean = pyramid(p["direct"]["target"], depth)
    for z, y, t, s, x, u in zip(
        p["direct"]["images"],
        clean,
        p["direct"]["levels"],
        p["direct"]["guides"],
        p["bridge"]["images"],
        p["bridge"]["progress"],
        strict=True,
    ):
        np.testing.assert_allclose(
            x, jnp.where(t > 0, (1 - tau) * z + tau * y, s), atol=2.5e-7, rtol=0
        )
        np.testing.assert_allclose(u, jnp.broadcast_to(1 - tau, x.shape), atol=1.2e-7, rtol=0)
        np.testing.assert_array_equal(
            np.asarray(x)[np.asarray(t) == 0], np.asarray(s)[np.asarray(t) == 0]
        )
    return dict(
        step=step,
        depth=depth,
        old_time=list(map(float, np.asarray(old_tau).ravel())),
        new_time=list(map(float, np.asarray(tau).ravel())),
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_only_time_and_interpolated_state_change(depth):
    for step in (0, 1, 1999, 5999):
        audit_batch(step, depth)


def test_time_density_emphasizes_both_boundaries():
    t = np.linspace(0, 1, 10001)
    transformed = np.sin(np.pi * t / 2) ** 2
    assert np.all(np.diff(transformed) >= 0)
    assert abs(np.mean(transformed < 0.01) - 2 * np.arcsin(0.1) / np.pi) < 0.0002
    assert abs(np.mean(transformed > 0.99) - 2 * np.arcsin(0.1) / np.pi) < 0.0002
    prepared, labels, _ = old_batch(0, 1, count=2)
    _, _, new = redistribute_time(prepared, labels, jnp.array([0.0, 1.0]).reshape(2, 1, 1, 1))
    np.testing.assert_array_equal(np.asarray(new).ravel(), [0.0, 1.0])


def test_actual_update_matches_parent_and_full_state_restores(tmp_path):
    import shutil

    from boundary_time_bridge_toy import make_arms, velocity_inputs
    from diagnose_learnability import update
    from test_velocity_bridge_toy import fixture
    from velocity_bridge_toy import make_arms as parent_arms

    from lineartgen.training import checkpoints, lineart_state

    fixture(tmp_path)
    shutil.copyfile(tmp_path / "configs/velocity.json", tmp_path / "configs/boundary.json")
    arms = make_arms(tmp_path)
    control = parent_arms(tmp_path)
    prepared = velocity_inputs(batch(1, 2, count=2)[0])
    result = update(arms["boundary"], prepared)
    assert result["gradient_norm"] > 0
    assert result == update(control["velocity"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload({"boundary": control["velocity"]}, 1),
    )
    with checkpoints.manager(tmp_path / "boundary-checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["boundary"], prepared) == update(restored["boundary"], prepared)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
