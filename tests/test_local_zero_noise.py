import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diagnose_learnability import update
from local_edge_energy import prepared_batch as control_batch
from local_recursive_fit import fresh_model, make_arm
from local_zero_noise import prepared_batch, split_local
from test_local_random_bridge import fixture
from train_procedural_structure import prepare_batch
from zero_noise_rollout import integrate, without_noise

from lineartgen.config import load_config
from lineartgen.diffusion import pyramid
from lineartgen.training import checkpoints, lineart_state


def test_same_data_only_gaussian_removed():
    c = load_config(
        "../jax-local-edge-energy-20261012/runs/local-edge-energy-20261012/configs/control.json"
    )
    preparation = prepare_batch(
        load_config(
            "../jax-local-edge-energy-20261012/runs/local-edge-energy-20261012/configs/source.json"
        )
    )
    for index in (14000, 14001, 17999):
        p, row = prepared_batch(c, preparation, index, size=64)
        old, old_row = control_batch(c, preparation, index, size=64)
        assert row == {
            **old_row,
            "control_inputs_sha256": hashlib.sha256(
                b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(old))
            ).hexdigest(),
        }
        for k in old:
            jax.tree.map(
                np.testing.assert_array_equal, p[k], old["guides"] if k == "images" else old[k]
            )
        assert any(
            np.any(np.asarray(x) != np.asarray(s))
            for x, s in zip(old["images"], p["images"], strict=True)
        )


def test_actual_update_and_restore(tmp_path):
    c, pair = fixture(tmp_path)
    parent = make_arm(c, fresh_model(c))
    update(parent, pair["direct"])
    arm = split_local(parent, c)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": parent}, 1),
        lineart_state.payload({"local": arm}, 1),
    )
    p = without_noise(pair["direct"])
    expected = make_arm(c, parent["model"])
    a, b = arm["gradient"](arm["model"], p), expected["gradient"](expected["model"], p)
    jax.tree.map(np.testing.assert_array_equal, a, b)
    stats = update(arm, p)
    assert stats["gradient_norm"] > 0 and stats["loss"] > 0
    assert "midpoint_mse" in stats and "flow_loss" not in stats
    with checkpoints.manager(tmp_path / "saved") as manager:
        lineart_state.save(manager, {"local": arm}, 2)
        restored = split_local(parent, c)
        assert lineart_state.restore(manager, {"local": restored}, 2) == 2
    assert update(arm, p) == update(restored, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": arm}, 3),
        lineart_state.payload({"local": restored}, 3),
    )


@pytest.mark.parametrize("steps", [2, 32])
def test_teacher_free_sampler_pyramid_and_protection(steps):
    source = jnp.arange(1024, dtype=jnp.float32).reshape(1, 32, 32, 1) / 512 - 1
    strength = jnp.full_like(source, 0.5).at[:, :, :11].set(0)
    guides, levels = pyramid(source, 4), pyramid(strength, 4)
    calls = []

    def predict(x, t, s, u):
        if not calls:
            jax.tree.map(np.testing.assert_array_equal, x, guides)
        jax.tree.map(np.testing.assert_array_equal, t, levels)
        jax.tree.map(np.testing.assert_array_equal, s, guides)
        for value in u:
            np.testing.assert_array_equal(value, jnp.full_like(value, 1 - len(calls) / steps))
        calls.append(1)
        return tuple((0.25 - a) / jnp.maximum(b, 1 / 32) for a, b in zip(x, u, strict=True))

    actual = np.asarray(integrate(predict, source, strength, depth=4, steps=steps))
    assert len(calls) == steps
    np.testing.assert_array_equal(actual[:, :, :11], np.asarray(source)[:, :, :11])
    np.testing.assert_allclose(actual[:, :, 11:], 0.25, atol=1e-7)
