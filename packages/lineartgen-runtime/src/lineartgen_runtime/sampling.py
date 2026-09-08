"""NumPy orchestration around a single-resolution ONNX forward function."""

import numpy as np


def pyramid(image, count):
    result = [image]
    for _ in range(count - 1):
        batch, height, width, channels = result[-1].shape
        result.append(
            result[-1].reshape(batch, height // 2, 2, width // 2, 2, channels).mean((2, 4))
        )
    return result


def denoise(forward, clean, noise_level, *, seed, steps, levels):
    noise = np.random.default_rng(seed).standard_normal(clean.shape, dtype=np.float32)
    image = clean * np.sqrt(1 - noise_level) + noise * np.sqrt(noise_level)
    # Keep the same number of calls for strength=0: the model's identity behavior
    # is learned and zero strength is still a valid forward request.
    for step in range(steps):
        current = np.float32((steps - step) / steps)
        following = np.float32((steps - step - 1) / steps)
        images = pyramid(image, levels)
        noises = pyramid(noise_level * current, levels)
        smallest = images[-1]
        insert = np.zeros((1, smallest.shape[1] // 2, smallest.shape[2] // 2, 1), np.float32)
        for x, level in zip(reversed(images), reversed(noises), strict=True):
            insert = forward(
                np.ascontiguousarray(x), np.ascontiguousarray(level), np.ascontiguousarray(insert)
            )
        ratio = np.sqrt(following / current)
        image = image * ratio + insert * (
            np.sqrt(1 - noise_level * following) - np.sqrt(1 - noise_level * current) * ratio
        )
    return image
