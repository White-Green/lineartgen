"""NumPy feedback and whole-image recursive ONNX stage execution."""

import numpy as np


def resize_linear(value, height, width):
    def axis_resize(image, axis, count):
        source = image.shape[axis]
        coordinate = (np.arange(count, dtype=np.float32) + np.float32(0.5)) * np.float32(
            source / count
        ) - np.float32(0.5)
        low = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - low.astype(np.float32)
        shape = [1] * image.ndim
        shape[axis] = count
        fraction = fraction.reshape(shape)
        left = np.take(image, np.clip(low, 0, source - 1), axis=axis)
        right = np.take(image, np.clip(low + 1, 0, source - 1), axis=axis)
        return left * (1 - fraction) + right * fraction

    return axis_resize(axis_resize(value, 1, height), 2, width)


def sigmoid(value):
    exponent = np.exp(-np.abs(value))
    return np.where(value >= 0, 1 / (1 + exponent), exponent / (1 + exponent))


def features(image, previous, level):
    image = np.asarray(image, np.float32)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 5 or min(image.shape[1:3]) < 8:
        raise ValueError("Expected one HxWx5 image with dimensions at least eight")
    if not 0 <= level < 6 or not np.isfinite(image).all():
        raise ValueError("Invalid level or input")
    _, height, width, _ = image.shape
    if previous is None:
        context = np.zeros((1, height, width, 3), np.float32)
    else:
        previous = np.asarray(previous, np.float32)
        if previous.shape != (1, height // 2, width // 2, 9) or not np.isfinite(previous).all():
            raise ValueError("Previous graph must be finite at half resolution")
        occupied, fill = sigmoid(previous[..., :1]), sigmoid(previous[..., 8:9])
        union = 1 - (1 - occupied) * (1 - fill)
        log_width = np.clip(previous[..., 3:4] + np.float32(np.log(2.0)), -8, 8) / 4
        context = resize_linear(
            np.concatenate((union, occupied, occupied * log_width), -1), height, width
        )
    condition = np.broadcast_to(
        np.asarray([level / 5, float(previous is None)], np.float32), (1, height, width, 2)
    )
    return np.concatenate((image, context, condition), -1)


def forward(session, levels):
    if not 1 <= len(levels) <= 6:
        raise ValueError("Expected one to six levels")
    heads, previous = [None] * len(levels), None
    for level in reversed(range(len(levels))):
        value = features(levels[level][..., :5], previous, level)
        previous = session.run(["graph"], {"features": value})[0]
        if previous.shape != (*value.shape[:3], 9) or not np.isfinite(previous).all():
            raise RuntimeError("Invalid stage graph")
        heads[level] = previous
    return tuple(heads)
