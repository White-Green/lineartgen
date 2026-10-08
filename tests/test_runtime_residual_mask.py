import numpy as np
import pytest
from lineartgen_runtime.sampling import denoise, pyramid


@pytest.mark.parametrize("steps", [2, 5, 20])
@pytest.mark.parametrize("identity", [False, True])
def test_residual_zero_noise_is_exact_at_every_scale_and_iteration(steps, identity):
    random = np.random.default_rng(712)
    source = random.uniform(-1, 1, (1, 32, 48, 1)).astype(np.float32)
    strength = np.full_like(source, 0.3)
    strength[:, :, :7] = 0  # Includes boundaries inside a pooling block.
    strength[:, 16:, 24:] = 0
    if identity:
        strength.fill(0)
    sources, masks = pyramid(source, 3), pyramid(strength, 3)
    calls = []

    def forward(image, noise, insert):
        index = next(i for i, s in enumerate(sources) if s.shape == image.shape)
        np.testing.assert_array_equal(image[masks[index] == 0], sources[index][masks[index] == 0])
        calls.append(index)
        return np.clip(image + noise * np.float32(0.31), -1, 1)

    actual = denoise(
        forward, source, strength, seed=9, steps=steps, levels=3, preserve_zero_noise=True
    )
    np.testing.assert_array_equal(actual[strength == 0], source[strength == 0])
    assert calls == [2, 1, 0] * steps
    # Pointwise oracle: the guard cannot change active-pixel updates or noise draws.
    unguarded = denoise(
        lambda image, noise, insert: np.clip(image + noise * np.float32(0.31), -1, 1),
        source,
        strength,
        seed=9,
        steps=steps,
        levels=3,
    )
    np.testing.assert_array_equal(actual[strength > 0], unguarded[strength > 0])


def test_direct_output_default_still_permits_changes_at_zero_noise():
    source = np.zeros((1, 16, 16, 1), np.float32)
    result = denoise(
        lambda image, noise, insert: np.ones_like(image),
        source,
        source,
        seed=1,
        steps=2,
        levels=1,
    )
    np.testing.assert_array_equal(result, np.ones_like(source))
