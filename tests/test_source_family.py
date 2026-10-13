import jax
import jax.numpy as jnp
import numpy as np
import pytest
from augmented_family_objective import augmented_gradient
from cue_capacity import forward, make_arm, prepare_inputs
from flax import nnx
from lineartgen_runtime.sampling import denoise
from local_recursive_fit import fresh_model, local_config
from source_family_study import update
from test_family_objective import candidates
from test_path_profile import annotation
from test_plausible_lineart import fixture

from lineartgen.config import ModelConfig
from lineartgen.diffusion import sample_prepared
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference


def source_config(config):
    mc = ModelConfig.model_validate(
        config.model.model_dump()
        | dict(residual_base="source", feature_input="source", sampling_mode="one_pass")
    )
    return config.model_copy(update={"model": mc})


def test_source_local_invariance_protection_serialization_and_sampler(tmp_path):
    config, p, _ = fixture()
    noisy = fresh_model(config)
    noisy.head.kernel[...] = jnp.full_like(noisy.head.kernel[...], 0.01)
    cfg = source_config(config)
    source = create_model(cfg.model, rngs=nnx.Rngs(0))
    nnx.update(source, nnx.state(noisy, nnx.Param))
    assert sum(x.size for x in jax.tree.leaves(nnx.state(source, nnx.Param))) == 15809
    strength = p.noise_levels[0].at[:, :, :16].set(0)
    prepared = [
        prepare_inputs(p.targets[0], p.guides[0], strength, jax.random.PRNGKey(k), depth=2)
        for k in (1, 2)
    ]
    a, b = [forward(source, x) for x in prepared]
    jax.tree.map(np.testing.assert_array_equal, a, b)
    jax.tree.map(
        np.testing.assert_array_equal,
        a,
        forward(noisy, prepared[0]._replace(noisy=prepared[0].guides)),
    )
    np.testing.assert_array_equal(a[0][0][strength == 0], p.guides[0][strength == 0])
    save_inference(source, cfg, tmp_path / "model", provenance="source local test")
    loaded, restored = load_inference(tmp_path / "model")
    assert restored == cfg and loaded.sampling_mode == "one_pass"
    jax.tree.map(np.testing.assert_array_equal, a, forward(loaded, prepared[0]))
    for steps in (1, 8):
        np.testing.assert_array_equal(
            sample_prepared(
                loaded,
                prepared[0].noisy,
                prepared[0].noise_levels,
                steps=steps,
                guides=prepared[0].guides,
            ),
            a[0][0],
        )
    # Runtime orchestration accepts progress-conditioned one-pass models and uses u=1.
    calls = []

    def stage(image, level, insert, guide, progress):
        assert np.all(progress == 1)
        calls.append(image.shape)
        return np.asarray(
            loaded(
                jnp.asarray(image),
                jnp.asarray(level),
                jnp.asarray(insert),
                jnp.asarray(guide),
                jnp.asarray(progress),
            )
        )

    out = denoise(
        stage,
        np.asarray(p.guides[0]),
        np.asarray(strength),
        seed=8,
        steps=7,
        levels=2,
        source_conditioning=True,
        progress_conditioning=True,
        preserve_zero_noise=True,
        sampling_mode="one_pass",
    )
    assert len(calls) == 2
    np.testing.assert_allclose(out, np.asarray(a[0][0]), atol=2e-7)


def test_source_local_gradient_updates_hidden_parameters():
    config, p, _ = fixture()
    cfg = source_config(config)
    model = fresh_model(cfg)
    _, _, v = candidates()
    ann = annotation() | dict(opacity=jnp.asarray([0.4]))
    arm = make_arm(cfg, model)
    arm["gradient"] = augmented_gradient()
    assert update(arm, p, v, ann)["gradient_norm"] > 0
    _, grads = arm["gradient"](arm["model"], p, v, ann)
    for component in ("stem", "head"):
        assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(grads[component])) > 0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))


@pytest.mark.parametrize(
    "changes",
    [
        dict(feature_input="source"),
        dict(residual_base="source"),
        dict(feature_input="source", residual_base="source"),
        dict(sampling_mode="one_pass"),
    ],
)
def test_inconsistent_local_contract_rejected(changes):
    with pytest.raises(ValueError):
        ModelConfig.model_validate(local_config().model_dump() | changes)
