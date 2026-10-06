import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lineartgen.config import ExperimentConfig
from lineartgen.diffusion import TrainingInputs
from lineartgen.models import create_model
from lineartgen.training.lineart import Profiles
from lineartgen.training.lineart_adversarial import (
    PatchCritic,
    critic_input,
    extract_patches,
    patch_coordinates,
)
from lineartgen.training.lineart_steps import accumulate_generator, make_lineart_steps
from lineartgen.training.steps import accumulate_gradients, make_steps


def test_patch_coordinates_and_extraction_pair_real_and_fake_without_borders():
    image = np.arange(2 * 32 * 32, dtype=np.float32).reshape(2, 32, 32, 1)
    coordinates = patch_coordinates(image / image.max(), seed=6, step=9, size=16)
    np.testing.assert_array_equal(
        coordinates, patch_coordinates(image / image.max(), seed=6, step=9, size=16)
    )
    patches = np.asarray(extract_patches(image, coordinates, 16))
    for i, (y, x) in enumerate(coordinates.reshape(-1, 2)):
        np.testing.assert_array_equal(patches[i], image[i // 4, y : y + 16, x : x + 16])


def fixture():
    config = ExperimentConfig.model_validate(
        dict(
            model=dict(channels=[2, 4], convolutions_per_block=1, output_mode="residual"),
            train=dict(batch_size=1, accumulate_steps=2),
        )
    )
    model = create_model(config.model, rngs=nnx.Rngs(6))
    images = (jnp.zeros((2, 32, 32, 1)), jnp.zeros((2, 16, 16, 1)))
    targets = tuple(jnp.ones_like(x) * 0.2 for x in images)
    prepared = TrainingInputs(targets, tuple(jnp.ones_like(x) * 0.5 for x in images), images)
    profiles = Profiles(jnp.zeros((2, 1, 3, 2)), jnp.zeros((2, 1)))
    coordinates = jnp.zeros((2, 4, 2), jnp.int32)
    return config, model, prepared, profiles, coordinates


def test_no_auxiliary_objective_matches_existing_recursive_accumulated_gradient():
    config, model, prepared, profiles, coordinates = fixture()
    ordinary = accumulate_gradients(make_steps(config)[0], model, prepared, config.train)
    study = accumulate_generator(
        make_lineart_steps(config)[0],
        model,
        prepared,
        profiles,
        coordinates,
        None,
        0.0,
        config.train,
    )
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b), ordinary, study)


def test_prior_changes_generator_gradient_without_mutating_critic_or_protected_pixels():
    config, model, prepared, profiles, coordinates = fixture()
    critic = PatchCritic(rngs=nnx.Rngs(8))
    before = jax.tree.map(np.asarray, nnx.state(critic))
    gradient = make_lineart_steps(config, adversarial=True, patch_size=16)[0]
    args = (model, prepared, profiles, coordinates, critic)
    metrics0, grads0 = gradient(*args, jnp.asarray(0.0))
    metrics1, grads1 = gradient(*args, jnp.asarray(0.1))
    assert float(metrics1["loss"]) > float(metrics0["loss"])
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(grads0), jax.tree.leaves(grads1), strict=True)
    )
    jax.tree.map(np.testing.assert_array_equal, before, nnx.state(critic))
    inactive = prepared._replace(
        noise_levels=tuple(jnp.zeros_like(x) for x in prepared.noise_levels)
    )

    def fn(x):
        return critic_input(x, inactive, coordinates, 16, 0.001)[0].sum()

    np.testing.assert_array_equal(jax.grad(fn)(prepared.targets[0]), 0)
