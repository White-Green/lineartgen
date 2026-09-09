import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lineartgen_runtime.sampling import denoise

from lineartgen.diffusion import sample


def run_sampler(backend, forward, clean, noise_level, *, steps, levels):
    if backend == "numpy":
        return denoise(forward, clean, noise_level, seed=7, steps=steps, levels=levels)
    return sample(
        forward,
        jnp.asarray(clean),
        jax.random.PRNGKey(7),
        strength=jnp.asarray(noise_level),
        steps=steps,
        levels=levels,
    )


def reference_scales(image, levels):
    """Block averages taken directly from the clean input, before any corruption."""
    batch, height, width, channels = image.shape
    return [
        image.reshape(batch, height // factor, factor, width // factor, factor, channels).mean(
            (2, 4)
        )
        for factor in (2**level for level in range(levels))
    ]


def install_noise(monkeypatch, backend, noises):
    draws = []

    def draw(shape):
        draws.append(shape)
        return noises[shape].copy()

    if backend == "numpy":

        class FixedGenerator:
            def standard_normal(self, shape, dtype):
                return draw(shape).astype(dtype)

        monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedGenerator())
    else:
        monkeypatch.setattr(jax.random, "normal", lambda key, shape: jnp.asarray(draw(shape)))
    return draws


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("zero_noise", [False, True])
def test_resize_before_noise_and_keep_each_scales_trajectory(monkeypatch, backend, zero_noise):
    rng = np.random.default_rng(12)
    clean = rng.uniform(-1, 1, (1, 64, 96, 1)).astype(np.float32)
    level = np.full_like(clean, 0.8)
    # Include fully protected blocks and a boundary inside a 2x2 pooling block.
    level[:, :, :9] = 0
    level[:, ::2, 32:] = 0
    if zero_noise:
        level.fill(0)
    targets = reference_scales(clean, 3)
    maps = reference_scales(level, 3)
    noises = {x.shape: rng.standard_normal(x.shape).astype(np.float32) for x in targets}
    draws = install_noise(monkeypatch, backend, noises)
    calls = []

    def oracle(image, noise_level, insert):
        index = next(i for i, target in enumerate(targets) if target.shape == image.shape)
        calls.append((index, np.asarray(image), np.asarray(noise_level), np.asarray(insert)))
        return targets[index].copy()

    result = run_sampler(backend, oracle, clean, level, steps=3, levels=3)
    assert draws == [target.shape for target in targets]  # One independent draw per scale.
    assert [call[0] for call in calls] == [2, 1, 0] * 3
    for call_index, (index, image, noise_level, insert) in enumerate(calls):
        t = (3 - call_index // 3) / 3
        target, noise = targets[index], noises[targets[index].shape]
        expected = target * np.sqrt(1 - maps[index] * t) + noise * np.sqrt(maps[index] * t)
        np.testing.assert_allclose(image, expected, atol=1e-6)
        np.testing.assert_allclose(noise_level, maps[index] * t, atol=1e-7)
        # A higher resolution receives the smaller clean prediction, not its noisy state.
        if index == 2:
            np.testing.assert_array_equal(insert, np.zeros_like(insert))
        else:
            np.testing.assert_array_equal(insert, targets[index + 1])
    np.testing.assert_allclose(result, clean, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_coarse_noise_keeps_unit_variance_and_is_independent(backend):
    clean = np.zeros((1, 256, 256, 1), np.float32)
    level = np.full_like(clean, 0.4)
    noises = {}

    def observe(image, noise_level, insert):
        noises[image.shape[1]] = np.asarray(image) / np.sqrt(np.float32(0.4))
        return np.zeros(image.shape, np.float32)

    run_sampler(backend, observe, clean, level, steps=1, levels=3)
    for size in (256, 128, 64):
        noise = noises[size]
        assert abs(noise.mean()) < 0.06
        assert 0.9 < noise.var() < 1.1  # Downsampling fine-scale noise would give 1/4 or 1/16.
        if size < 256:
            downsampled = reference_scales(noises[size * 2], 2)[1]
            correlation = np.corrcoef(noise.ravel(), downsampled.ravel())[0, 1]
            assert abs(correlation) < 0.08


@pytest.mark.parametrize("levels", [1, 3])
def test_jitted_jax_and_runtime_agree_given_same_noise(monkeypatch, levels):
    rng = np.random.default_rng(24)
    clean = rng.uniform(-1, 1, (1, 64, 96, 1)).astype(np.float32)
    level = rng.uniform(0, 0.8, clean.shape).astype(np.float32)
    level[:, :, :12] = 0
    noises = {
        x.shape: rng.standard_normal(x.shape).astype(np.float32)
        for x in reference_scales(clean, levels)
    }

    def forward(xp):
        return lambda image, noise_level, insert: (
            0.4 * image + 0.1 * noise_level + 0.3 * xp.repeat(xp.repeat(insert, 2, 1), 2, 2)
        )

    install_noise(monkeypatch, "numpy", noises)
    expected = run_sampler("numpy", forward(np), clean, level, steps=3, levels=levels)
    install_noise(monkeypatch, "jax", noises)
    compiled = jax.jit(
        lambda image, noise_level: sample(
            forward(jnp),
            image,
            jax.random.PRNGKey(7),
            strength=noise_level,
            steps=3,
            levels=levels,
        )
    )
    np.testing.assert_allclose(compiled(clean, level), expected, atol=1e-6)
