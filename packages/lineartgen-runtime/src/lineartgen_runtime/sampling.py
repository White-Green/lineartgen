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


def local_contrast_source(guide):
    """NumPy counterpart of full-resolution local-peak normalization."""
    ink = np.clip((1 - guide) / 2, 0, 1)
    peak = ink
    for axis in (1, 2):
        padding = [(0, 0)] * 4
        padding[axis] = (16, 16)
        padded = np.pad(peak, padding, mode="constant")
        peak = np.zeros_like(ink)
        for offset in range(33):
            region = [slice(None)] * 4
            region[axis] = slice(offset, offset + ink.shape[axis])
            np.maximum(peak, padded[tuple(region)], out=peak)
    return 1 - np.float32(0.8) * ink / np.maximum(peak, np.float32(0.02))


def denoise(
    forward,
    clean,
    noise_level,
    *,
    seed,
    steps,
    levels,
    source_conditioning=False,
    progress_conditioning=False,
    preserve_zero_noise=False,
    sampling_mode="diffusion",
    pyramid_conditioning="none",
    source_preprocess="raw",
):
    if source_preprocess not in ("raw", "local_peak_full"):
        raise ValueError("Unsupported source preprocessing")
    if source_preprocess != "raw" and not (source_conditioning and sampling_mode == "one_pass"):
        raise ValueError("Full source normalization requires one-pass source conditioning")
    if pyramid_conditioning not in ("none", "constant", "level"):
        raise ValueError("Unsupported pyramid conditioning")
    if pyramid_conditioning != "none" and not (source_conditioning and sampling_mode == "one_pass"):
        raise ValueError("Pyramid conditioning requires one-pass source conditioning")
    if sampling_mode not in ("diffusion", "one_pass"):
        raise ValueError("Unsupported sampling mode")
    if sampling_mode == "one_pass":
        if not source_conditioning or not preserve_zero_noise:
            raise ValueError("One-pass restoration requires a protected source-conditioned model")
        if steps < 1:
            raise ValueError("steps must be positive")
        steps = 1
    rng = np.random.default_rng(seed)
    clean_scales = pyramid(clean, levels)
    guides = (
        pyramid(local_contrast_source(clean), levels)
        if source_preprocess == "local_peak_full"
        else clean_scales
    )
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
        for index, (x, level, guide) in enumerate(
            zip(reversed(images), reversed(noise_levels), reversed(guides), strict=True)
        ):
            extra = {} if pyramid_conditioning == "none" else dict(pyramid_level=levels - 1 - index)
            arguments = (
                np.ascontiguousarray(x),
                np.ascontiguousarray(level * current),
                np.ascontiguousarray(insert),
            )
            if progress_conditioning:
                if not source_conditioning:
                    raise ValueError("Progress conditioning requires the original rough guide")
                insert = forward(
                    *arguments, np.ascontiguousarray(guide), np.full_like(x, current), **extra
                )
            else:
                insert = (
                    forward(*arguments, np.ascontiguousarray(guide), **extra)
                    if source_conditioning
                    else forward(*arguments)
                )
            predictions.append(insert)
        ratio = np.sqrt(following / current)
        updated = [
            image * ratio
            + prediction * (np.sqrt(1 - level * following) - np.sqrt(1 - level * current) * ratio)
            for image, prediction, level in zip(
                images, reversed(predictions), noise_levels, strict=True
            )
        ]
        # Residual models are exactly identity at zero noise. Match JAX's
        # protection against roundoff in the sampler's cancelling terms.
        images = (
            [
                np.where(level == 0, image, value)
                for image, value, level in zip(images, updated, noise_levels, strict=True)
            ]
            if preserve_zero_noise
            else updated
        )
    return images[0]
