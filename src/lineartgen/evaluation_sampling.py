"""Runtime-compatible fixed noise draws for isolated recursion comparisons."""

import numpy as np
from lineartgen_runtime.images import pyramid_plan
from lineartgen_runtime.sampling import pyramid


def runtime_levels(height, width, minimum_size):
    plan = pyramid_plan(height, width, minimum_size, max(128, minimum_size))
    if plan.sizes[0] != (height, width):
        raise ValueError("Fixed evaluation inputs must already have runtime-compatible dimensions")
    return len(plan.sizes)


def runtime_inputs(source, strength, seed, levels):
    """Draw in fine-to-coarse order so adding levels leaves all shared draws unchanged."""
    sources, masks = pyramid(source, levels), tuple(pyramid(strength, levels))
    rng = np.random.default_rng(seed)
    images = tuple(
        x * np.sqrt(1 - t) + rng.standard_normal(x.shape, dtype=np.float32) * np.sqrt(t)
        for x, t in zip(sources, masks, strict=True)
    )
    return images, masks


def inference_with_source(model, images, noise_levels, source, *, steps):
    """Use the actual supplied rough image as a fixed guide, never a clean teacher."""
    from lineartgen.diffusion import sample_prepared

    guides = (
        tuple(pyramid(source, len(images)))
        if getattr(model, "source_conditioning", False)
        else None
    )
    return sample_prepared(model, images, noise_levels, steps=steps, guides=guides)


def background_metrics(prediction, target, strength):
    mask = (strength > 0) & (target >= 0.99)
    count = int(mask.sum())
    return dict(
        active_background_pixels=count,
        active_background_mae=float(np.abs(prediction[mask] - target[mask]).mean())
        if count
        else None,
        active_background_dark_fraction=float((prediction[mask] < 0).mean()) if count else None,
    )
