import jax
import jax.numpy as jnp
import numpy as np
import pytest
from anchored_endpoint import endpoints
from diagnose_learnability import update
from flax import nnx
from local_recursive_fit import fresh_model, local_config, make_arm
from rollout_image_loss import gradient
from test_progress_modulation import modulated
from test_rollout_image_loss import image_config
from velocity_bridge_toy import batch, velocity_inputs

from lineartgen.config import ModelConfig
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state


def config_pair(tmp_path):
    _, _, c, _ = modulated(tmp_path)
    # Same full U-Net as the research baseline, not the tiny toy fixture.
    mc = ModelConfig(
        channels=(4, 10, 20, 40),
        source_conditioning=True,
        progress_conditioning=True,
        progress_modulation="fourier_film",
        detail_channels=16,
        output_mode="residual",
    )
    c = c.model_copy(update={"model": mc, "loss": image_config()})
    return c, c.model_copy(update={"model": local_config()})


@pytest.mark.parametrize(
    "changes",
    [
        {"channels": (4, 8)},
        {"source_conditioning": False},
        {"progress_modulation": "none"},
        {"detail_channels": 2},
        {"convolutions_per_block": 3},
        {"skip_mode": "pre_pool"},
        {"nested_channels": ((32,),)},
        {"residual_scale_floor": 0.1},
    ],
)
def test_rejects_unimplemented_local_options(changes):
    with pytest.raises(ValueError):
        ModelConfig.model_validate(local_config().model_dump() | changes)


def test_parameter_counts_initial_function_and_protection(tmp_path):
    unet, local = config_pair(tmp_path)
    models = [fresh_model(c) for c in (unet, local)]
    assert [sum(x.size for x in jax.tree.leaves(nnx.state(m, nnx.Param))) for m in models] == [
        134688,
        15809,
    ]
    p = velocity_inputs(batch(2, 3, count=2)[0])
    values = [endpoints(m, p["images"], p["levels"], p["guides"], p["progress"]) for m in models]
    jax.tree.map(np.testing.assert_array_equal, *values)
    for raw, x, t, source in zip(values[0][0], p["images"], p["levels"], p["guides"], strict=True):
        np.testing.assert_array_equal(raw, jnp.where(t > 0, x, source))


def test_local_gradient_restore_next_update_and_strict_inference(tmp_path):
    _, c = config_pair(tmp_path)
    model = fresh_model(c)
    arm = make_arm(c, model)
    p = velocity_inputs(batch(2, 2, count=2)[0])
    stats = update(arm, p)
    assert stats["gradient_norm"] > 0
    # Zero head blocks hidden-layer gradient at update zero; after it changes,
    # every hidden block and progress projection must receive a finite gradient.
    _, g = gradient(arm["model"], p, c.loss, c.diffusion.min_noise, 0.1)
    for key in ("stem", "stem_modulation", "head"):
        assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(g[key])) > 0
    for block in g["blocks"].values():
        for component in ("first", "second", "modulation"):
            assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(block[component])) > 0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(g))
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, {"local": arm}, 1)
        restored = make_arm(c, model)
        assert lineart_state.restore(saved, {"local": restored}, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": arm}, 1),
        lineart_state.payload({"local": restored}, 1),
    )
    assert update(arm, p) == update(restored, p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"local": arm}, 2),
        lineart_state.payload({"local": restored}, 2),
    )
    save_inference(arm["model"], c, tmp_path / "inference", provenance="test local roundtrip")
    loaded, cfg = load_inference(tmp_path / "inference")
    assert c == cfg
    jax.tree.map(
        np.testing.assert_array_equal,
        nnx.state(loaded, nnx.Param),
        nnx.state(arm["model"], nnx.Param),
    )
    args = [p[k] for k in ("images", "levels", "guides", "progress")]
    jax.tree.map(
        np.testing.assert_array_equal, endpoints(loaded, *args), endpoints(arm["model"], *args)
    )


def test_local_stage_translation_and_no_fine_to_coarse_feedback():
    m = create_model(local_config(), rngs=nnx.Rngs(4))
    # Nonzero head makes this a functional check, not zero-output tautology.
    m.head.kernel[...] = jnp.ones_like(m.head.kernel[...]) * 0.02
    x = jax.random.normal(jax.random.PRNGKey(22), (1, 64, 64, 1))
    t, s, u = jnp.full_like(x, 0.5), x * 0.2, jnp.full_like(x, 0.5)
    insert = x[:, ::2, ::2]
    a = m(x, t, insert, s, u, return_network_output=True)
    b = m(
        jnp.roll(x, 2, 2),
        t,
        jnp.roll(insert, 1, 2),
        jnp.roll(s, 2, 2),
        u,
        return_network_output=True,
    )
    np.testing.assert_allclose(
        b[:, 9:-9, 9:-9], jnp.roll(a, 2, 2)[:, 9:-9, 9:-9], atol=2e-7, rtol=2e-6
    )
    p = velocity_inputs(batch(2, 3, count=2)[0])
    before = endpoints(m, *[p[k] for k in ("images", "levels", "guides", "progress")])
    changed = (p["images"][0] + 0.2, *p["images"][1:])
    after = endpoints(m, changed, *[p[k] for k in ("levels", "guides", "progress")])
    for a, b in zip(before, after, strict=True):
        jax.tree.map(np.testing.assert_array_equal, a[1:], b[1:])
    assert not np.array_equal(before[0][0], after[0][0])
