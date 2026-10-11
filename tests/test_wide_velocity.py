import jax
import numpy as np
from diagnose_learnability import update
from flax import nnx
from test_velocity_bridge_toy import fixture
from velocity_bridge_toy import batch, objective, recursive_velocity, velocity_inputs
from wide_velocity_fit import make_arms

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import save_inference
from lineartgen.models.widen import widen_unet
from lineartgen.training import checkpoints, lineart_state


def test_widen_learned_progress_velocity_and_new_paths_then_restore(tmp_path):
    narrow, _, _ = fixture(tmp_path)
    c = load_config(tmp_path / "configs/velocity.json")
    for i, name in enumerate(
        ("guide_encoder", "guide_head", "progress_encoder", "progress_head", "progress_detail")
    ):
        layer = getattr(narrow, name)
        layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(40 + i), layer.kernel.shape) * 0.1
    d = c.model.model_copy(
        update=dict(
            channels=(4, 8), detail_channels=8, final_decoder_channels=4, head_hidden_channels=6
        )
    )
    wide = widen_unet(narrow, c.model, d, seed=2026101113)
    for depth in (1, 2, 3):
        p = velocity_inputs(batch(4, depth, count=2)[0])
        args = [p[k] for k in ("images", "levels", "guides", "progress")]
        a, b = recursive_velocity(narrow, *args), recursive_velocity(wide, *args)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5, rtol=0), a, b)
        for image, target, level in zip(b[1], p["guides"], p["levels"], strict=True):
            mask = np.asarray(level) == 0
            np.testing.assert_array_equal(np.asarray(image)[mask], np.asarray(target)[mask])
    p = velocity_inputs(batch(4, 2, count=2)[0])
    g = nnx.grad(lambda m: objective(m, p)[0])(wide)
    index = len(wide.detail_head.layers) - 1
    new = np.asarray(g["detail_head"]["layers"][index]["kernel"])[
        :, :, c.model.detail_channels :, :
    ]
    assert np.isfinite(new).all() and np.any(new != 0)
    np.testing.assert_array_equal(
        wide.detail_head.layers[-1].kernel[:, :, c.model.detail_channels :, :], 0
    )
    c = c.model_copy(update={"model": d})
    (tmp_path / "initial").rename(tmp_path / "narrow-initial")
    save_inference(wide, c, tmp_path / "initial", provenance="widening test")
    save_config(c, tmp_path / "configs/wide.json")
    arms = make_arms(tmp_path)
    m = update(arms["wide"], p)
    assert m["gradient_norm"] > 0 and m["velocity_mse"] > 0
    with checkpoints.manager(tmp_path / "checkpoints") as saved:
        lineart_state.save(saved, arms, 1)
        restored = make_arms(tmp_path)
        assert lineart_state.restore(saved, restored) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 1),
        lineart_state.payload(restored, 1),
    )
    assert update(arms["wide"], p) == update(restored["wide"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 2),
        lineart_state.payload(restored, 2),
    )
