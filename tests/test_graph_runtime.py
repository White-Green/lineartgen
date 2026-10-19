"""Runtime invariants independent of the research checkout and training dataset."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from lineartgen_runtime import LineartModel
from lineartgen_runtime.graph import GraphLineartModel, normalized_guide, render


@pytest.fixture(scope="module")
def graph_model():
    assets = Path(
        os.environ.get(
            "LINEART_GRAPH_TEST_ASSETS", "packages/lineartgen-runtime/src/lineartgen_runtime/assets"
        )
    )
    return GraphLineartModel(assets / "graph-model.json", provider="cpu")


def options(width=33, height=17):
    scribble = np.zeros((height, width, 4), np.uint8)
    scribble[:, width // 2, 3] = 128
    return dict(
        scribble=scribble.tobytes(),
        lineart=bytes(width * height * 4),
        width=width,
        height=height,
        strength=0.5,
        seed=42,
        denoise_steps=3,
    )


def test_normalized_guide_blank_and_opacity_invariance():
    source = np.ones((1, 35, 49, 1), np.float32)
    np.testing.assert_array_equal(normalized_guide(source), source)
    source[:, 4:31, 17:19] = -1
    faint = 1 - 0.25 * (1 - source)
    np.testing.assert_array_equal(normalized_guide(source), normalized_guide(faint))


def test_opaque_renderer_tile_boundaries_and_isolated_points():
    head = np.full((21, 35, 9), -20, np.float32)
    head[10, 5:30, 0] = 2
    head[10, 5:29, 4] = 2
    head[..., 1:3] = 0
    head[..., 3] = 0
    head[2, 2, 0] = 2
    head[3:5, 30:33, 8] = 2
    full = render(head, tile_size=128)
    np.testing.assert_array_equal(render(head, tile_size=3), full)
    assert np.all(full[10, 5:30, 0] < 0)
    assert full[2, 2, 0] < 0
    assert np.all(full[3:5, 30:33] == -1)
    assert np.all(full[0] == 1)
    head[..., 0] = -20
    head[..., 8] = -20
    np.testing.assert_array_equal(render(head), np.ones((21, 35, 1), np.float32))


@pytest.mark.parametrize("size,depth", [((33, 17), 1), ((145, 129), 3), ((16, 300), 6)])
def test_graph_bgra_rectangles_depths_and_repeatability(graph_model, size, depth):
    args = options(*size) | dict(denoise_steps=depth)
    actual = graph_model.infer(**args)
    assert actual == graph_model.infer(**args)
    assert len(actual) == size[0] * size[1] * 4
    bgra = np.frombuffer(actual, np.uint8).reshape(size[1], size[0], 4)
    assert not bgra[..., :3].any()
    assert graph_model.infer(**(args | dict(seed=999))) == actual
    recolored = np.frombuffer(args["scribble"], np.uint8).copy().reshape(-1, 4)
    recolored[:, :3] = [255, 80, 190]
    assert graph_model.infer(**(args | dict(scribble=recolored.tobytes()))) == actual


def test_graph_blank_identity_and_protected_pixels(graph_model):
    args = options()
    blank = graph_model.infer(**(args | dict(scribble=bytes(33 * 17 * 4))))
    assert not any(blank)
    identity = graph_model.infer(**(args | dict(strength=0)))
    assert identity == args["scribble"]
    lineart = np.zeros((17, 33, 4), np.uint8)
    lineart[:8, :16] = [255, 255, 255, 255]
    lineart[8:, :16] = [0, 0, 0, 255]
    result = np.frombuffer(
        graph_model.infer(**(args | dict(lineart=lineart.tobytes()))), np.uint8
    ).reshape(17, 33, 4)
    assert not result[:8, :16].any()
    assert np.all(result[8:, :16, 3] == 255)


@pytest.mark.parametrize(
    "invalid",
    [
        dict(denoise_steps=0),
        dict(denoise_steps=7),
        dict(strength=float("nan")),
        dict(strength=1.1),
        dict(seed=-1),
        dict(width=15),
        dict(scribble=b""),
    ],
)
def test_graph_invalid_inputs(graph_model, invalid):
    with pytest.raises(ValueError):
        graph_model.infer(**(options() | invalid))


def test_graph_manifest_dispatch(graph_model):
    assets = Path(
        os.environ.get(
            "LINEART_GRAPH_TEST_ASSETS", "packages/lineartgen-runtime/src/lineartgen_runtime/assets"
        )
    )
    model = LineartModel(assets / "graph-model.json", provider="cpu")
    assert model.infer(**options()) == graph_model.infer(**options())
    assert model.backend == "onnxruntime graph"


def test_graph_stage_tiling_preserves_non_square_predictions(graph_model):
    rng = np.random.default_rng(17)
    source = rng.uniform(0.2, 1, (1, 64, 320, 1)).astype(np.float32)
    noise = rng.uniform(0, 0.75, source.shape).astype(np.float32)
    insert = rng.uniform(-1, 1, (1, 32, 160, 1)).astype(np.float32)
    actual = graph_model.stage(source, noise, insert, 0, tile_size=64)
    expected = graph_model.stage(source, noise, insert, 0, tile_size=None)
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)


def test_packaged_graph_matches_independent_jax_reference(graph_model):
    assets = Path(
        os.environ.get(
            "LINEART_GRAPH_TEST_ASSETS", "packages/lineartgen-runtime/src/lineartgen_runtime/assets"
        )
    )
    reference = json.loads((assets / "graph-reference.json").read_text())
    assert reference["checkpoint_sha256"] == graph_model.metadata["checkpoint_sha256"]
    path = assets / "graph-reference.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["arrays_sha256"]
    with np.load(path, allow_pickle=False) as z:
        heads = graph_model.heads(z["source"], z["strength"], 6)
        for level, actual in enumerate(heads):
            np.testing.assert_allclose(actual, z[f"head{level}"], **reference["tolerance"])
        assert np.any(heads[0][..., 0] >= 0)
        np.testing.assert_allclose(render(heads[0][0]), z["raster"], atol=2 / 255, rtol=0)
