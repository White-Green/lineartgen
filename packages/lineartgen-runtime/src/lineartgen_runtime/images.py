"""The stable Krita BGRA contract. No model framework is imported here."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PyramidPlan:
    original: tuple[int, int]
    padded: tuple[int, int]
    sizes: tuple[tuple[int, int], ...]


def pyramid_plan(height, width, minimum=16, stop=128):
    if min(height, width) < minimum:
        raise ValueError(f"Both image dimensions must be at least {minimum}")
    if minimum < 2 or stop < minimum:
        raise ValueError("Invalid model pyramid metadata")
    h, w, reductions = height, width, 0
    while h > stop or w > stop:
        following = ((h + 1) // 2, (w + 1) // 2)
        if min(following) < minimum:
            break
        h, w = following
        reductions += 1
    alignment = minimum * 2**reductions
    padded = tuple((size + alignment - 1) // alignment * alignment for size in (height, width))
    sizes = tuple((padded[0] // 2**level, padded[1] // 2**level) for level in range(reductions + 1))
    return PyramidPlan((height, width), padded, sizes)


def prepare_input(scribble, lineart, plan, strength, image_scale=1.1):
    height, width = plan.original
    expected = height * width * 4
    for name, value in (("scribble", scribble), ("lineart", lineart)):
        if len(value) != expected:
            raise ValueError(f"{name} must contain exactly {expected} BGRA bytes")
    scribble = np.frombuffer(scribble, np.uint8).reshape(height, width, 4)
    lineart = np.frombuffer(lineart, np.uint8).reshape(height, width, 4)
    scribble_luma = 1 - scribble[..., 3].astype(np.float32) / 255
    alpha = lineart[..., 3].astype(np.float32) / 255
    luma = (
        lineart[..., :3].astype(np.float32) @ np.array([0.0722, 0.7152, 0.2126], np.float32) / 255
    )
    combined = luma * alpha + scribble_luma * (1 - alpha)
    clean = np.ones((1, *plan.padded, 1), np.float32)
    noise = np.zeros_like(clean)
    clean[0, :height, :width, 0] = np.clip((combined * 2 - 1) * image_scale, -1, 1)
    noise[0, :height, :width, 0] = strength * (1 - alpha)
    return clean, noise


def to_bgra(prediction, plan, alpha_threshold=0.1):
    height, width = plan.original
    gray = np.clip(prediction[0, :height, :width, 0], -1, 1)
    luma = np.floor((gray + 1) * 127.5 + 0.5).astype(np.uint8)
    alpha = 255 - luma
    output = np.zeros((height, width, 4), np.uint8)
    output[..., 3] = np.where(alpha.astype(np.float32) / 255 < alpha_threshold, 0, alpha)
    return output.tobytes()
