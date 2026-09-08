import subprocess
import sys

import numpy as np
import pytest
from lineartgen_runtime import LineartModel
from lineartgen_runtime.images import prepare_input, pyramid_plan, to_bgra


@pytest.fixture(scope="module")
def model():
    return LineartModel(provider="cpu")


def inputs(width=33, height=17):
    scribble = np.zeros((height, width, 4), np.uint8)
    scribble[:, width // 2, 3] = 255
    return dict(
        scribble=scribble.tobytes(),
        lineart=bytes(width * height * 4),
        width=width,
        height=height,
        strength=0.5,
        seed=42,
        denoise_steps=2,
    )


@pytest.mark.parametrize("size", [(33, 17), (145, 129), (16, 300)])
def test_packaged_model_contract(model, size):
    options = inputs(*size)
    result = model.infer(**options)
    assert result == model.infer(**options)
    bgra = np.frombuffer(result, np.uint8).reshape(size[1], size[0], 4)
    assert not bgra[..., :3].any()
    assert bgra[..., 3].any()
    recolored = np.frombuffer(options["scribble"], np.uint8).copy().reshape(-1, 4)
    recolored[:, :3] = [255, 80, 190]
    assert model.infer(**(options | {"scribble": recolored.tobytes()})) == result


@pytest.mark.parametrize(
    "invalid",
    [
        dict(strength=float("nan")),
        dict(strength=1.1),
        dict(denoise_steps=0),
        dict(seed=-1),
        dict(scribble=b""),
        dict(width=15),
    ],
)
def test_invalid_input(model, invalid):
    with pytest.raises(ValueError):
        model.infer(**(inputs() | invalid))


def test_bgra_compositing_padding_and_output():
    plan = pyramid_plan(17, 33)
    scribble = np.zeros((17, 33, 4), np.uint8)
    scribble[0, :3, 3] = [255, 128, 255]
    lineart = np.zeros_like(scribble)
    lineart[0, 2] = [255, 255, 255, 255]
    clean, noise = prepare_input(scribble.tobytes(), lineart.tobytes(), plan, 0.5)
    np.testing.assert_allclose(clean[0, 0, :3, 0], [-1, (1 - 256 / 255) * 1.1, 1], atol=1e-7)
    assert noise[0, 0, 2, 0] == 0
    assert np.all(clean[:, 17:] == 1) and np.all(noise[:, 17:] == 0)
    result = np.frombuffer(to_bgra(clean, plan), np.uint8).reshape(17, 33, 4)
    assert result[0, 0, 3] == 255 and result[0, 2, 3] == 0


def test_runtime_import_does_not_load_training_framework():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from lineartgen_runtime import LineartModel; "
            "m=LineartModel(provider='cpu'); import sys; "
            "assert not {'jax', 'flax', 'torch', 'orbax'} & sys.modules.keys()",
        ],
        check=True,
    )
