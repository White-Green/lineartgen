import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import forward, make_arm, prepare_inputs
from flax import nnx
from lineartgen_runtime.sampling import denoise
from style_family_objective import style_gradient
from style_family_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture
from test_source_family import source_config

from lineartgen.config import ModelConfig
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state


def configs():
    c, p, _ = fixture()
    c = source_config(c)
    result = []
    for ds in ((1, 1, 1), (1, 2, 4)):
        mc = ModelConfig.model_validate(c.model.model_dump() | dict(local_dilations=ds))
        result.append(c.model_copy(update={"model": mc}))
    return result, p


def test_same_capacity_initial_function_and_existing_default():
    cs, p = configs()
    plain, dilated = [create_model(c.model, rngs=nnx.Rngs(31)) for c in cs]
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(plain, nnx.Param), nnx.state(dilated, nnx.Param)
    )
    assert all(
        sum(x.size for x in jax.tree.leaves(nnx.state(m, nnx.Param))) == 15809
        for m in (plain, dilated)
    )
    jax.tree.map(np.testing.assert_array_equal, forward(plain, p), forward(dilated, p))
    default = create_model(
        ModelConfig.model_validate(cs[0].model.model_dump() | dict(local_dilations=())),
        rngs=nnx.Rngs(31),
    )
    for m in (plain, default):
        m.head.kernel[...] = jnp.full_like(m.head.kernel[...], 0.01)
    jax.tree.map(np.testing.assert_array_equal, forward(plain, p), forward(default, p))


def test_dilation_increases_actual_source_receptive_field():
    cs, _ = configs()
    gradients = []
    guide = jnp.ones((1, 64, 64, 1))
    for c in cs:
        m = create_model(c.model, rngs=nnx.Rngs(31))
        for layer in (m.stem, *[x for b in m.blocks for x in (b.first, b.second)], m.head):
            layer.kernel[...] = jnp.full_like(layer.kernel[...], 0.05)
            layer.bias[...] = jnp.full_like(layer.bias[...], 0.1)
        g = jax.grad(
            lambda s, m=m: m(
                s,
                jnp.full_like(s, 0.5),
                jnp.ones((1, 32, 32, 1)),
                s,
                jnp.ones_like(s),
                return_network_output=True,
            )[0, 32, 32, 0]
        )(guide)
        gradients.append(np.asarray(g)[0, :, :, 0])
    assert gradients[0][32, 45] == 0
    assert abs(gradients[1][32, 45]) > 1e-8
    assert gradients[1][32, 48] == 0
    assert np.isfinite(gradients).all()


def test_dilated_update_restore_serialization_and_sampler(tmp_path):
    cs, p = configs()
    c = cs[1]
    initial = create_model(c.model, rngs=nnx.Rngs(31))
    arm = make_arm(c, initial)
    arm["gradient"] = style_gradient()
    _, _, v = candidates()
    ann = annotation() | dict(normalizers=v)
    stats = update(arm, p, v, ann)
    assert stats["gradient_norm"] > 0
    assert np.any(np.asarray(arm["model"].head.kernel[...]) != 0)
    _, g = arm["gradient"](arm["model"], p, v, ann)
    for block in g["blocks"].values():
        for key in ("first", "second"):
            assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(block[key])) > 0
    with checkpoints.manager(tmp_path / "state") as saved:
        lineart_state.save(saved, {"dilated": arm}, 1)
        restored = make_arm(c, initial)
        restored["gradient"] = style_gradient()
        assert lineart_state.restore(saved, {"dilated": restored}, 1) == 1
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"dilated": arm}, 1),
        lineart_state.payload({"dilated": restored}, 1),
    )
    assert update(arm, p, v, ann) == update(restored, p, v, ann)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"dilated": arm}, 2),
        lineart_state.payload({"dilated": restored}, 2),
    )
    save_inference(arm["model"], c, tmp_path / "inference", provenance="dilation test")
    loaded, cfg = load_inference(tmp_path / "inference")
    assert cfg == c and loaded.local_dilations == (1, 2, 4)
    jax.tree.map(np.testing.assert_array_equal, forward(loaded, p), forward(arm["model"], p))
    t = p.noise_levels[0].at[:, :, :16].set(0)
    prep = prepare_inputs(p.targets[0], p.guides[0], t, jax.random.PRNGKey(9), depth=2)
    expected = forward(loaded, prep)[0][0]
    np.testing.assert_array_equal(expected[t == 0], p.guides[0][t == 0])

    def stage(image, strength, insert, guide, progress):
        return np.asarray(loaded(*map(jnp.asarray, (image, strength, insert, guide, progress))))

    actual = denoise(
        stage,
        np.asarray(p.guides[0]),
        np.asarray(t),
        seed=7,
        steps=3,
        levels=2,
        source_conditioning=True,
        progress_conditioning=True,
        preserve_zero_noise=True,
        sampling_mode="one_pass",
    )
    np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_invalid_options_and_unvalidated_export(tmp_path):
    cs, _ = configs()
    for ds in ((1, 2), (0, 1, 1), (1, 2, 5)):
        with pytest.raises(ValueError, match="local_dilations"):
            ModelConfig.model_validate(cs[0].model.model_dump() | dict(local_dilations=ds))
    with pytest.raises(ValueError, match="local_dilations"):
        ModelConfig(local_dilations=(1, 2, 4))
    with pytest.raises(ValueError, match="Dilated"):
        export_model(None, cs[1], tmp_path / "unused.onnx")
