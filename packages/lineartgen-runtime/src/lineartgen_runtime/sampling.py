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


def denoise(forward, clean, noise_level, *, seed, steps, levels, source_conditioning=False):
    rng = np.random.default_rng(seed)
    clean_scales = pyramid(clean, levels)
    noise_levels = pyramid(noise_level, levels)
    # Match training: resize the clean image and mask before adding independent
    # noise at each resolution. Draw once; later steps evolve each scale's state.
    images = [
        target * np.sqrt(1 - level)
        + rng.standard_normal(target.shape, dtype=np.float32) * np.sqrt(level)
        for target, level in zip(clean_scales, noise_levels, strict=True)
    ]
    for step in range(steps):
        current = np.float32((steps - step) / steps)
        following = np.float32((steps - step - 1) / steps)
        smallest = images[-1]
        insert = np.zeros(
            (smallest.shape[0], smallest.shape[1] // 2, smallest.shape[2] // 2, 1), np.float32
        )
        predictions = []
        for x, level, guide in zip(
            reversed(images), reversed(noise_levels), reversed(clean_scales), strict=True
        ):
            arguments = (
                np.ascontiguousarray(x),
                np.ascontiguousarray(level * current),
                np.ascontiguousarray(insert),
            )
            insert = (
                forward(*arguments, np.ascontiguousarray(guide))
                if source_conditioning
                else forward(*arguments)
            )
            predictions.append(insert)
        ratio = np.sqrt(following / current)
        images = [
            image * ratio
            + prediction * (np.sqrt(1 - level * following) - np.sqrt(1 - level * current) * ratio)
            for image, prediction, level in zip(
                images, reversed(predictions), noise_levels, strict=True
            )
        ]
    return images[0]
