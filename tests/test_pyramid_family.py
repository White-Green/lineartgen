import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cue_capacity import forward, make_arm, prepare_inputs
from flax import nnx
from lineartgen_runtime.sampling import denoise
from local_recursive_fit import fresh_model
from style_family_objective import style_gradient
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture
from test_source_family import source_config

from lineartgen.config import ModelConfig
from lineartgen.diffusion import recursive_forward, recursive_forward_with_preclip, sample_prepared
from lineartgen.export import export_model
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference


def configure(config, mode):
    return config.model_copy(
        update={
            "model": ModelConfig.model_validate(
                config.model.model_dump() | dict(pyramid_conditioning=mode)
            )
        }
    )


def test_zero_initialization_level_dependency_and_saved_sampler(tmp_path):
    config, p, _ = fixture()
    config = source_config(config)
    parent = fresh_model(config)
    parent.head.kernel[...] = jnp.full_like(parent.head.kernel[...], 0.01)
    models = {}
    for mode in ("constant", "level"):
        cfg = configure(config, mode)
        model = create_model(cfg.model, rngs=nnx.Rngs(3))
        nnx.update(model, nnx.state(parent, nnx.Param))
        assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 15937
        jax.tree.map(np.testing.assert_array_equal, forward(parent, p), forward(model, p))
        model.stem_pyramid_modulation.kernel[...] = jnp.full_like(
            model.stem_pyramid_modulation.kernel[...], 0.05
        )
        models[mode] = model
        args = (
            p.noisy[0],
            p.noise_levels[0],
            jnp.zeros((1, 16, 16, 1)),
            p.guides[0],
            p.progress[0],
        )
        a, b = [model(*args, pyramid_level=k) for k in (0, 5)]
        if mode == "constant":
            np.testing.assert_array_equal(a, b)
        else:
            assert np.max(np.abs(np.asarray(a - b))) > 1e-5
        save_inference(model, cfg, tmp_path / mode, provenance="pyramid test")
        loaded, restored = load_inference(tmp_path / mode)
        assert restored == cfg and loaded.pyramid_conditioning == mode
        jax.tree.map(np.testing.assert_array_equal, forward(model, p), forward(loaded, p))
        with pytest.raises(ValueError, match="explicit ONNX level input"):
            export_model(model, cfg, tmp_path / "unsupported.onnx")
    model = models["level"]
    strength = p.noise_levels[0].at[:, :, :16].set(0)
    prepared = prepare_inputs(p.targets[0], p.guides[0], strength, jax.random.PRNGKey(4), depth=3)
    expected = forward(model, prepared)[0][0]
    out = sample_prepared(
        model, prepared.noisy, prepared.noise_levels, steps=8, guides=prepared.guides
    )
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out[strength == 0], p.guides[0][strength == 0])
    levels = []

    def stage(image, noise, insert, guide, progress, *, pyramid_level):
        levels.append(pyramid_level)
        assert np.all(progress == 1)
        return np.asarray(
            model(
                *map(jnp.asarray, (image, noise, insert, guide, progress)),
                pyramid_level=pyramid_level,
            )
        )

    actual = denoise(
        stage,
        np.asarray(p.guides[0]),
        np.asarray(strength),
        seed=91,
        steps=7,
        levels=3,
        source_conditioning=True,
        progress_conditioning=True,
        preserve_zero_noise=True,
        sampling_mode="one_pass",
        pyramid_conditioning="level",
    )
    assert levels == [2, 1, 0]
    np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_finest_level_zero_independent_of_recursion_depth():
    class ObserveLevel:
        pyramid_conditioning = "level"

        def __call__(
            self, image, noise, insert, guide, progress, *, pyramid_level, return_preclip=False
        ):
            value = jnp.full_like(image, pyramid_level)
            return (value, value) if return_preclip else value

    for depth in range(1, 7):
        images = tuple(jnp.zeros((1, 128 // 2**k, 128 // 2**k, 1)) for k in range(depth))
        kwargs = dict(guides=images, progress=images)
        a = recursive_forward(ObserveLevel(), images, images, **kwargs)
        b, raw = recursive_forward_with_preclip(ObserveLevel(), images, images, **kwargs)
        for k in range(depth):
            np.testing.assert_array_equal(a[k], np.full(a[k].shape, k))
            np.testing.assert_array_equal(a[k], b[k])
            np.testing.assert_array_equal(a[k], raw[k])


def test_pyramid_modulation_has_real_gradients_and_updates():
    config, p, _ = fixture()
    cfg = configure(source_config(config), "level")
    parent = fresh_model(source_config(config))
    parent.head.kernel[...] = jnp.full_like(parent.head.kernel[...], 0.01)
    model = create_model(cfg.model, rngs=nnx.Rngs(0))
    nnx.update(model, nnx.state(parent, nnx.Param))
    _, _, v = candidates()
    ann = annotation() | dict(normalizers=v)
    arm = make_arm(cfg, model)
    arm["gradient"] = style_gradient()
    _, grads = arm["gradient"](model, p, v, ann)
    leaves = [grads["stem_pyramid_modulation"]["kernel"][...]] + [
        grads["blocks"][k]["pyramid_modulation"]["kernel"][...] for k in range(3)
    ]
    assert all(float(jnp.linalg.norm(x)) > 0 for x in leaves)
    arm["update"](model, arm["optimizer"], grads, arm["ema"])
    assert np.any(np.asarray(model.stem_pyramid_modulation.kernel[...]) != 0)


@pytest.mark.parametrize("mode", ["constant", "level"])
def test_pyramid_conditioning_rejects_unsupported_architecture(mode):
    with pytest.raises(ValueError, match="one-pass local"):
        ModelConfig(pyramid_conditioning=mode)
