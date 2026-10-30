"""BGRA and width contracts; these fixtures make no learned-quality claim."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import onnx
import pytest
from lineartgen_runtime.graph import render
from lineartgen_runtime.spatial import SpatialLineartModel


def head():
    value = np.full((1, 256, 256, 9), -20, np.float32)
    value[..., 1:3] = 0
    value[..., 3] = np.log(12)
    value[0, 128, :, 0] = 20
    value[0, 128, :, 4] = 20
    return value


def options(width=256, height=256, depth=3):
    image = np.zeros((height, width, 4), np.uint8)
    image[..., 3] = np.arange(width, dtype=np.uint32)[None] % 256
    return dict(
        scribble=image.tobytes(),
        lineart=bytes(width * height * 4),
        width=width,
        height=height,
        strength=0.6,
        seed=2**40 + 123,
        denoise_steps=depth,
    )


@pytest.fixture
def manifest(tmp_path):
    entries = {}
    for depth in (1, 3, 6):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "Constant", [], ["graph"], value=onnx.numpy_helper.from_array(head())
                    )
                ],
                "spatial_runtime_contract_fixture",
                [
                    onnx.helper.make_tensor_value_info(
                        f"image_{i}", onnx.TensorProto.FLOAT, [1, 256 // 2**i, 256 // 2**i, 5]
                    )
                    for i in range(depth)
                ],
                [
                    onnx.helper.make_tensor_value_info(
                        "graph", onnx.TensorProto.FLOAT, [1, 256, 256, 9]
                    )
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 20)],
        )
        model.ir_version = 10
        path = tmp_path / f"depth-{depth}.onnx"
        onnx.save(model, path)
        entries[str(depth)] = dict(
            file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    spec = dict(
        format_version=1,
        representation="spatial_intent_graph",
        tile_size=256,
        core_size=128,
        input_channels=5,
        output_channels=9,
        normalization="separate_layers_sum",
        noise_rng="numpy_seedsequence_pcg64_v1",
        noise_after_downsample=True,
        renderer="attraction_graph_native_width_v1",
        models=entries,
    )
    path = tmp_path / "spatial-model.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize(
    "width,height,depth", [(256, 193, 1), (385, 193, 3), (385, 385, 6), (1, 1, 6)]
)
def test_native_shapes_width_and_tile_seams(manifest, width, height, depth):
    model = SpatialLineartModel(manifest, provider="cpu")
    args = options(width, height, depth)
    result = model.infer(**args)
    assert result == model.infer(**args)
    image = np.frombuffer(result, np.uint8).reshape(height, width, 4)
    assert not image[..., :3].any()
    for y in [128] if height <= 256 else [64, 192, 320]:
        if y < height:
            assert np.all(image[y, :, 3] == 255)
            # A former 8px width limit would fail this native 12px stroke check.
            assert np.all(image[y - 5 : y + 6, :, 3] == 255)
    assert not image[0, :, 3].any()


def test_pressure_width_profile_is_not_clamped():
    value = head()[0]
    value[128, :, 3] = np.log(np.linspace(2, 18, 256))
    alpha = (1 - render(value, max_width=None)[..., 0]) / 2
    for x in (32, 96, 160, 224):
        assert abs(alpha[:, x].sum() - (2 + 16 * x / 255)) < 0.25
        assert alpha[128, x] == 1


def test_strength_zero_and_black_white_protection(manifest):
    model = SpatialLineartModel(manifest, provider="cpu")
    args = options(385, 193)
    assert model.infer(**(args | dict(strength=0))) == args["scribble"]
    paint = np.zeros((193, 385, 4), np.uint8)
    paint[:80, :190] = [0, 0, 0, 255]
    paint[80:, :190] = [255, 255, 255, 255]
    output = np.frombuffer(model.infer(**(args | dict(lineart=paint.tobytes()))), np.uint8).reshape(
        193, 385, 4
    )
    assert np.all(output[:80, :190, 3] == 255)
    assert not output[80:, :190, 3].any()


def test_validation_and_no_training_framework_import(manifest):
    model = SpatialLineartModel(manifest, provider="cpu")
    for change in (
        dict(width=0),
        dict(strength=float("nan")),
        dict(seed=-1),
        dict(seed=2**64),
        dict(denoise_steps=2),
        dict(scribble=b""),
    ):
        with pytest.raises(ValueError):
            model.infer(**(options() | change))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lineartgen_runtime.spatial import SpatialLineartModel; "
            "assert 'jax' not in sys.modules and 'flax' not in sys.modules",
        ],
        check=True,
    )
    spec = json.loads(manifest.read_text())
    spec["models"]["1"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="digest mismatch"):
        SpatialLineartModel(manifest, provider="cpu")
