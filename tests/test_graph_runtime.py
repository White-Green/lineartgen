import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from attraction_graph import decode  # noqa: E402
from lineartgen_runtime.graph import normalized_guide, render  # noqa: E402
from projected_edge_gate import gate  # noqa: E402

from lineartgen.models.local import local_contrast_cue  # noqa: E402


def test_graph_local_normalization_matches_training_at_borders():
    rng = np.random.default_rng(42)
    source = rng.uniform(-1, 1, (1, 35, 49, 1)).astype(np.float32)
    source[:, 9:20, 12:40] = 1
    np.testing.assert_allclose(
        normalized_guide(source),
        np.asarray(local_contrast_cue(jnp.asarray(source))),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_graph_opaque_renderer_matches_research_including_gate_and_isolates(seed):
    rng = np.random.default_rng(seed)
    head = rng.normal(size=(16, 16, 9)).astype(np.float32)
    head[..., 0] -= 1
    head[..., 8] -= 2
    head[..., 3] = rng.uniform(-8, np.log(8) + 1, (16, 16))
    expected, _ = decode(gate(head, base_radius=1.0)[0], base_radius=1.0)
    np.testing.assert_array_equal(render(head), expected)
    np.testing.assert_array_equal(render(head, tile_size=3), expected)


def test_empty_and_filled_rectangular_graphs():
    head = np.full((9, 21, 9), -20, np.float32)
    np.testing.assert_array_equal(render(head), np.ones((9, 21, 1), np.float32))
    head[2:5, 3:7, 8] = 0
    expected = np.ones((9, 21, 1), np.float32)
    expected[2:5, 3:7] = -1
    np.testing.assert_array_equal(render(head), expected)
