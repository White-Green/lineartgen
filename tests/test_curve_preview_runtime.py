"""Image/geometry contracts; these tests do not establish learned cleanup quality."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import onnx
import pytest
from lineartgen_runtime import LineartModel
from lineartgen_runtime.curve_inputs import levels_from_bgra, tile_inputs
from lineartgen_runtime.curve_render import render_curves
from lineartgen_runtime.curves import CurveLineartModel


def options(width=256, height=256, depth=3):
    image = np.zeros((height, width, 4), np.uint8)
    image[..., 3] = np.arange(width, dtype=np.uint32)[None] % 256
    return dict(
        scribble=image.tobytes(),
        lineart=bytes(width * height * 4),
        width=width,
        height=height,
        strength=0.6,
        seed=123,
        denoise_steps=depth,
    )


def inputs(args):
    return levels_from_bgra(
        **{k: v for k, v in args.items() if k != "denoise_steps"}, depth=args["denoise_steps"]
    )


def curves():
    shape = np.zeros((1, 16, 256, 2), np.float32)
    shape[..., 0] = np.linspace(0, 1, 256)
    shape[..., 1] = 0.5
    widths = np.full((1, 16, 128), 2 / 256, np.float32)
    positions = np.broadcast_to(np.linspace(0, 1, 128, dtype=np.float32), widths.shape).copy()
    flags = np.full((1, 16, 2), -2, np.float32)
    flags[0, 0, 0] = 2
    return shape, widths, positions, flags


@pytest.fixture
def manifest(tmp_path):
    names = ("shape", "widths", "positions", "flags")
    output = curves()
    entries = {}
    for depth in (1, 3):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "Constant", [], [k], value=onnx.numpy_helper.from_array(v)
                    )
                    for k, v in zip(names, output, strict=True)
                ],
                "runtime_contract_fixture",
                [
                    onnx.helper.make_tensor_value_info(
                        f"image_{i}", onnx.TensorProto.FLOAT, [1, 256 // 2**i, 256 // 2**i, 11]
                    )
                    for i in range(depth)
                ],
                [
                    onnx.helper.make_tensor_value_info(k, onnx.TensorProto.FLOAT, v.shape)
                    for k, v in zip(names, output, strict=True)
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
        representation="independent_shape_width_curves",
        tile_size=256,
        core_size=128,
        input_channels=11,
        curves=16,
        shape_knots=256,
        width_knots=128,
        normalization="separate_layers_sum",
        noise_rng="numpy_seedsequence_pcg64_v1",
        noise_after_downsample=True,
        renderer="swept_discs_v1",
        models=entries,
    )
    path = tmp_path / "curve-model.json"
    path.write_text(json.dumps(spec))
    return path


def test_noise_added_after_downsampling_and_reproducible():
    args = options(depth=4)
    levels, _, _ = inputs(args)
    repeat, _, _ = inputs(args)
    short, _, _ = inputs(args | dict(denoise_steps=3))
    for i, image in enumerate(levels):
        np.testing.assert_array_equal(image, repeat[i])
        if i < 3:
            np.testing.assert_array_equal(image, short[i])
        epsilon = (image[..., 4] - image[..., 3]) / (0.125 * image[..., 2])
        assert abs(epsilon.mean()) < 0.10
        assert 0.90 < epsilon.std() < 1.10
    # Noise added before averaging would have standard deviation 1/8 at level3.
    seed_changed, _, _ = inputs(args | dict(seed=124))
    assert not np.array_equal(levels[0][..., 3], seed_changed[0][..., 3])
    np.testing.assert_array_equal(levels[0][..., :3], seed_changed[0][..., :3])
    np.testing.assert_array_equal(levels[0][..., 4], seed_changed[0][..., 4])


def test_overlapping_tiles_share_input_and_noise_at_every_depth():
    levels, rows, cols = inputs(options(600, 193, depth=6))
    assert rows.pad == 0 and cols.pad == 64
    a, b = tile_inputs(levels, 0, 0), tile_inputs(levels, 0, 128)
    for level, (x, y) in enumerate(zip(a, b, strict=True)):
        overlap = 128 // 2**level
        np.testing.assert_array_equal(x[:, :, overlap:], y[:, :, :overlap])
        assert x.shape == (1, 256 // 2**level, 256 // 2**level, 11)
    tile_inputs(levels, 0, 512)


def test_renderer_preserves_variable_width_and_opaque_center():
    output = curves()
    output[1][0, 0] = np.linspace(2, 18, 128) / 256
    alpha = (1 - render_curves(output)[..., 0]) / 2
    for x in (32, 96, 160, 224):
        expected = 2 + 16 * (x + 0.5) / 256
        assert abs(alpha[:, x].sum() - expected) < 0.25
        assert alpha[128, x] == 1
        assert not alpha[:115, x].any()
    output[3][0, 0, 0] = -2
    assert np.all(render_curves(output) == 1)


@pytest.mark.parametrize("width,height", [(256, 193), (385, 193), (385, 385), (1, 1)])
def test_bgra_shapes_and_crop_seams_with_analytic_line_fixture(manifest, width, height):
    model = CurveLineartModel(manifest, provider="cpu")
    args = options(width, height)
    result = model.infer(**args)
    assert result == model.infer(**args)
    image = np.frombuffer(result, np.uint8).reshape(height, width, 4)
    assert not image[..., :3].any()
    for y in [128] if height <= 256 else [64, 192, 320]:
        if y < height:
            assert np.all(image[y, :, 3] == 255)
    assert not image[0, :, 3].any()


def test_strength_zero_and_protected_black_white_pixels(manifest):
    model = CurveLineartModel(manifest, provider="cpu")
    args = options(385, 193)
    assert model.infer(**(args | dict(strength=0))) == args["scribble"]
    lineart = np.zeros((193, 385, 4), np.uint8)
    lineart[:80, :190] = [0, 0, 0, 255]
    lineart[80:, :190] = [255, 255, 255, 255]
    image = np.frombuffer(
        model.infer(**(args | dict(lineart=lineart.tobytes()))), np.uint8
    ).reshape(193, 385, 4)
    assert np.all(image[:80, :190, 3] == 255)
    assert not image[80:, :190, 3].any()


def test_dispatch_validation_and_no_model_framework_import(manifest):
    model = LineartModel(manifest, provider="cpu")
    assert model.backend == "onnxruntime curves"
    assert model.infer(**options()) == CurveLineartModel(manifest, provider="cpu").infer(
        **options()
    )
    for change in (
        dict(width=0),
        dict(strength=float("nan")),
        dict(seed=-1),
        dict(seed=2**64),
        dict(denoise_steps=6),
        dict(scribble=b""),
    ):
        with pytest.raises(ValueError):
            model.infer(**(options() | change))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lineartgen_runtime import LineartModel; "
            "assert 'jax' not in sys.modules and 'flax' not in sys.modules",
        ],
        check=True,
    )
    spec = json.loads(manifest.read_text())
    spec["models"]["1"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="digest mismatch"):
        CurveLineartModel(manifest, provider="cpu")
