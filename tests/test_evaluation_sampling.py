import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lineartgen_runtime.sampling import denoise

from lineartgen.config import ExperimentConfig
from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation_sampling import background_metrics, runtime_inputs, runtime_levels
from lineartgen.sketch import augment_sketch
from lineartgen.training.comparison import comparison_configs


@pytest.mark.parametrize("levels", [1, 3, 6])
def test_fixed_noise_recursion_matches_existing_runtime(levels):
    rng = np.random.default_rng(5)
    clean = rng.uniform(-1, 1, (1, 64, 128, 1)).astype(np.float32)
    strength = rng.uniform(0, 0.7, clean.shape).astype(np.float32)
    strength[:, :, :12] = 0
    full = runtime_inputs(clean, strength, 47, 6)
    prepared = runtime_inputs(clean, strength, 47, levels)
    for actual, reference in zip(prepared, full, strict=True):
        for a, b in zip(actual, reference[:levels], strict=True):
            np.testing.assert_array_equal(a, b)

    def forward(xp):
        return lambda image, noise_level, insert: (
            0.4 * image + 0.1 * noise_level + 0.3 * xp.repeat(xp.repeat(insert, 2, 1), 2, 2)
        )

    expected = denoise(forward(np), clean, strength, seed=47, steps=5, levels=levels)
    actual = jax.jit(lambda x, t: sample_prepared(forward(jnp), x, t, steps=5))(*prepared)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert runtime_levels(512, 512, 8) == 3
    assert runtime_levels(128, 128, 8) == 1


def test_rate_comparison_changes_only_probability_and_keeps_common_corruptions():
    base = ExperimentConfig.model_validate(
        {
            "model": {
                "architecture": "nested",
                "channels": [4, 6, 8],
                "nested_channels": [[6, 8, 12], [8, 12, 16]],
                "output_mode": "residual",
            },
            "matching": {"method": "independent"},
            "diffusion": {"pattern_weights": [1, 1, 1, 0]},
        }
    )
    variants = comparison_configs(base, study="sketch_rates")
    clean = jnp.ones((64, 16, 16, 1)).at[:, :, 7:9].set(-1)
    changed, sources = [], []
    for name, probability in (("p75", 0.75), ("p50", 0.50), ("p25", 0.25)):
        config = variants[name]
        assert config.sketch.probability == probability
        assert config.model_copy(update={"sketch": base.sketch}) == base
        source = np.asarray(augment_sketch(clean, jax.random.PRNGKey(42), config.sketch))
        changed.append(np.any(source != np.asarray(clean), axis=(1, 2, 3)))
        sources.append(source)
    assert 0 < changed[2].sum() < changed[1].sum() < changed[0].sum() < 64
    for i in (1, 2):
        assert np.all(~changed[i] | changed[i - 1])
        np.testing.assert_array_equal(sources[i][changed[i]], sources[i - 1][changed[i]])


def test_background_metrics_exclude_protected_and_target_ink():
    target = np.array([1, 1, -1, 1], np.float32)
    result = background_metrics(np.array([0, 1, 1, -1]), target, np.array([0, 0.5, 0.5, 0.5]))
    assert result == dict(
        active_background_pixels=2, active_background_mae=1.0, active_background_dark_fraction=0.5
    )
