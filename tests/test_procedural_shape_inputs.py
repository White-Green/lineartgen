import jax
import jax.numpy as jnp
import numpy as np
from cue_capacity import prepare_inputs
from plausible_study import exact
from procedural_shape_study import procedural_inputs


def example():
    rng = np.random.default_rng(2026101704)
    source = jnp.asarray(rng.uniform(-0.4, 1, (4, 64, 64, 1)).astype(np.float32))
    clean = jnp.asarray(np.where(np.indices((64, 64))[1] == 32, -1, 1)[None, ..., None])
    clean = jnp.repeat(clean.astype(jnp.float32), 4, axis=0)
    strength = jnp.where(jnp.indices((64, 64))[1][None, ..., None] > 32, 0.1, 0.0)
    strength = jnp.repeat(strength, 4, axis=0)
    return clean, source, strength, jax.random.PRNGKey(2026101704)


def test_matches_established_prepare_after_downsampling_at_all_six_depths():
    clean, source, strength, key = example()
    for depth in range(1, 7):
        reference = prepare_inputs(clean, source, strength, key, depth)
        actual = procedural_inputs(clean, source, reference.noise_levels, reference.progress, key)
        exact(actual, reference)


def test_supplied_coarse_masks_control_noise_and_protection_independently():
    clean, source, strength, key = example()
    reference = prepare_inputs(clean, source, strength, key, 6)
    levels = (reference.noise_levels[0], *[jnp.zeros_like(x) for x in reference.noise_levels[1:]])
    actual = procedural_inputs(clean, source, levels, reference.progress, key)
    exact(actual.noise_levels, levels)
    for guide, noise, target in zip(
        actual.guides[1:], actual.noisy[1:], actual.targets[1:], strict=True
    ):
        exact(guide, noise)
        exact(guide, target)
    protected = np.asarray(strength) == 0
    np.testing.assert_array_equal(
        np.asarray(actual.targets[0])[protected], np.asarray(source)[protected]
    )
