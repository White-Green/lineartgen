import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import endpoints
from anchored_endpoint import objective as endpoint_objective
from diagnose_learnability import update
from flax import nnx
from geometry_velocity_fit import prepared_batch as bridge_batch
from local_edge_energy import prepared_batch as control_batch
from local_edge_energy import sibling_configs
from local_random_bridge import prepared_batch, split_local
from local_recursive_fit import fresh_model, make_arm
from random_bridge_image_loss import objective
from test_local_recursive import config_pair
from train_procedural_structure import prepare_batch

from lineartgen.config import load_config
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.losses import image_loss


def fixture(tmp_path):
    _, c = config_pair(tmp_path)
    c = c.model_copy(
        update={
            "train": c.train.model_copy(
                update={
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "warmup_steps": 0,
                }
            )
        }
    )
    c = sibling_configs(c)["control"]
    y = jnp.ones((2, 32, 32, 1)).at[:, 8:24, 12:15].set(-1)
    source = 1 - (1 - y) * 0.3
    strength = jnp.full_like(y, 0.5).at[:, :, :5].set(0)
    p, _ = bridge_batch(y, source, strength, 14000, 2)
    return c, p


def test_bridge_risk_and_image_coefficient(tmp_path):
    c, p = fixture(tmp_path)
    model = fresh_model(c)
    bridge = p["bridge"]
    value, parts = objective(model, bridge, c.loss, c.diffusion.min_noise, 0.1)
    expected, _ = endpoint_objective(model, bridge)
    np.testing.assert_array_equal(parts["flow_loss"], expected)
    raw = endpoints(
        model, bridge["images"], bridge["levels"], bridge["guides"], bridge["progress"]
    )[0][0]
    auxiliary = jnp.mean(
        jnp.stack(
            [
                image_loss(
                    jnp.clip(raw[i : i + 1], -1, 1),
                    bridge["target"][i : i + 1],
                    bridge["levels"][0][i : i + 1],
                    c.loss,
                    c.diffusion.min_noise,
                    preclip=raw[i : i + 1],
                )[0]
                for i in range(2)
            ]
        )
    )
    np.testing.assert_allclose(value, expected + 0.1 * auxiliary, rtol=1e-6)
    assert np.isfinite(float(value))
    a = nnx.value_and_grad(lambda m: objective(m, bridge, c.loss, c.diffusion.min_noise, 0)[0])(
        model
    )
    b = nnx.value_and_grad(lambda m: endpoint_objective(m, bridge)[0])(model)
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7), a, b)


def test_real_state_continuation_and_next_update_restore(tmp_path):
    c, p = fixture(tmp_path)
    parent = make_arm(c, fresh_model(c))
    update(parent, p["direct"])
    arm = split_local(parent, c)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": parent}, 1),
        lineart_state.payload({"local": arm}, 1),
    )
    stats = update(arm, p["bridge"])
    assert stats["flow_loss"] > 0 and stats["gradient_norm"] > 0
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, {"local": arm}, 2)
        restored = split_local(parent, c)
        assert lineart_state.restore(saved, {"local": restored}, 2) == 2
    assert update(arm, p["bridge"]) == update(restored, p["bridge"])
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": arm}, 3),
        lineart_state.payload({"local": restored}, 3),
    )


def test_same_clean_source_noise_as_control_and_sampled_teacher_path():
    c = load_config(
        "../jax-local-edge-energy-20261012/runs/local-edge-energy-20261012/configs/control.json"
    )
    source_config = load_config(
        "../jax-local-edge-energy-20261012/runs/local-edge-energy-20261012/configs/source.json"
    )
    preparation = prepare_batch(source_config)
    import hashlib

    from lineartgen.diffusion import pyramid

    for index in (14000, 14001, 17999):
        p, row = prepared_batch(c, preparation, index, size=64)
        direct, old_row = control_batch(c, preparation, index, size=64)
        assert row["clean_sha256"] == old_row["clean_sha256"]
        assert row["teacher_counts"] == old_row["teacher_counts"]
        digest = hashlib.sha256()
        for x in jax.tree.leaves(direct):
            digest.update(np.asarray(x).tobytes())
        assert row["control_inputs_sha256"] == digest.hexdigest()
        assert 0 <= row["tau_min"] <= row["tau_mean"] <= row["tau_max"] <= 1
        clean = pyramid(p["target"], len(p["images"]))
        for x, z, y, s, t, u in zip(
            p["images"],
            direct["images"],
            clean,
            p["guides"],
            p["levels"],
            p["progress"],
            strict=True,
        ):
            np.testing.assert_allclose(x, jnp.where(t > 0, u * z + (1 - u) * y, s), atol=2e-7)
        jax.tree.map(np.testing.assert_array_equal, p["levels"], direct["levels"])
        jax.tree.map(np.testing.assert_array_equal, p["guides"], direct["guides"])
