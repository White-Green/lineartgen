import itertools

import jax
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from transport_bridge_toy import batch, targets, transport_labels
from velocity_bridge_toy import batch as independent_batch


def audit_batch(step, depth):
    p, labels, tau = batch(step, depth)
    q, old_labels, old_tau = independent_batch(step, depth)
    labels, old_labels = np.asarray(labels), np.asarray(old_labels)
    np.testing.assert_array_equal(tau, old_tau)
    for arm, fields in (
        ("direct", ("images", "levels", "guides", "progress")),
        ("bridge", ("levels", "guides", "progress")),
    ):
        for field in fields:
            jax.tree.map(np.testing.assert_array_equal, p[arm][field], q[arm][field])
    z, levels, source = [np.asarray(p["direct"][f][0]) for f in ("images", "levels", "guides")]
    active = levels > 0
    group = (levels.max((1, 2, 3)) > 0.3).astype(int) + 2 * (levels[:, 0, 0, 0] == 0)
    a, b, _ = targets()
    templates = np.stack((a, b)).astype(np.float64)
    before, after = 0.0, 0.0
    for g in range(4):
        ii = np.where(group == g)[0]
        np.testing.assert_array_equal(np.sort(labels[ii]), np.sort(old_labels[ii]))
        if not len(ii):
            continue
        costs = np.sum(
            ((z[ii, None].astype(np.float64) - templates[old_labels[ii]][None]) * active[ii, None])
            ** 2,
            axis=(2, 3, 4),
        )
        rows, columns = linear_sum_assignment(costs)
        best = costs[rows, columns].sum()
        actual = np.sum(((z[ii].astype(np.float64) - templates[labels[ii]]) * active[ii]) ** 2)
        independent = np.trace(costs)
        np.testing.assert_allclose(actual, best, atol=1e-7, rtol=1e-12)
        assert actual <= independent + 1e-7
        before += independent
        after += actual
    np.testing.assert_array_equal(
        p["bridge"]["target"], np.where(active, templates[labels], source)
    )
    np.testing.assert_array_equal(np.asarray(p["bridge"]["target"])[~active], source[~active])
    return dict(
        step=step,
        depth=depth,
        independent_cost=before,
        transport_cost=after,
        changed_labels=int(np.sum(labels != old_labels)),
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_conditional_transport_is_optimal_and_preserves_all_other_observations(depth):
    for step in (0, 1, 1999, 5999):
        audit_batch(step, depth)


def test_stable_ties_preserve_conditional_label_counts():
    import jax.numpy as jnp

    z = jnp.ones((4, 64, 64, 1))
    level = jnp.ones_like(z) * 0.5
    labels = jnp.array([1, 0, 1, 0])
    result = transport_labels(z, z, level, labels)
    np.testing.assert_array_equal(result, [0, 0, 1, 1])


def test_all_label_assignments_of_tiny_group_reach_exact_optimum():
    import jax.numpy as jnp

    a, b, s = targets()
    z = np.broadcast_to(s, (4, *s.shape)).copy()
    for i in range(4):
        z[i] += (i - 1.5) * (a - b) * 0.02
    levels = np.full_like(z, 0.5)
    labels = np.array([0, 1, 0, 1])
    actual = np.asarray(
        transport_labels(
            jnp.asarray(z), jnp.broadcast_to(s, z.shape), jnp.asarray(levels), jnp.asarray(labels)
        )
    )
    costs = []
    for perm in set(itertools.permutations(labels)):
        y = np.stack([a if i == 0 else b for i in perm])
        costs.append(np.sum((z.astype(np.float64) - y) ** 2))
    y = np.stack([a if i == 0 else b for i in actual])
    np.testing.assert_allclose(np.sum((z.astype(np.float64) - y) ** 2), min(costs), rtol=1e-12)


def test_transport_arm_uses_same_update_and_restores_full_state(tmp_path):
    import shutil

    from diagnose_learnability import update
    from test_velocity_bridge_toy import fixture
    from transport_bridge_toy import make_arms, velocity_inputs
    from velocity_bridge_toy import make_arms as old_make_arms

    from lineartgen.training import checkpoints, lineart_state

    fixture(tmp_path)
    shutil.copyfile(tmp_path / "configs/velocity.json", tmp_path / "configs/transport.json")
    arms = make_arms(tmp_path)
    baseline = old_make_arms(tmp_path)
    p = velocity_inputs(batch(1, 2, count=2)[0])
    before = lineart_state.payload(arms, 0)
    result = update(arms["transport"], p)
    assert result == update(baseline["velocity"], p)
    assert result["gradient_norm"] > 0
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload({"transport": baseline["velocity"]}, 1),
    )
    assert any(
        np.any(a != b)
        for a, b in zip(
            jax.tree.leaves(before), jax.tree.leaves(lineart_state.payload(arms, 1)), strict=True
        )
    )
    with checkpoints.manager(tmp_path / "transport-checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["transport"], p) == update(restored["transport"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
