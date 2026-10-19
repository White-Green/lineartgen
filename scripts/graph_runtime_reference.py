"""Save independent JAX predictions for deployment CI without training dependencies."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from attraction_graph import decode
from cue_capacity import prepare_inputs
from learned_centerline_model import infer_heads, load_weights
from learned_centerline_study import make_model
from projected_edge_gate import gate


def build(root, checkpoint, output):
    output.mkdir(parents=True, exist_ok=True)
    model, _ = make_model(root)
    load_weights(model, checkpoint)
    source = np.ones((1, 128, 128, 1), np.float32)
    source[:, 24:31, 14:112] = 0.4
    source[:, 12:49, 76:83] = 0.2
    noise = np.full_like(source, 0.5)
    noise[:, :, :32] = 0
    p = prepare_inputs(np.ones_like(source), source, noise, jax.random.PRNGKey(2026101913), 6)
    heads = [np.asarray(h) for h in infer_heads(model, p)]
    raster, _ = decode(gate(heads[0][0], base_radius=1.0)[0], base_radius=1.0)
    assert np.any(heads[0][..., 0] >= 0)
    path = output / "graph-reference.npz"
    np.savez_compressed(
        path,
        source=source,
        strength=noise,
        raster=raster,
        **{f"head{i}": head for i, head in enumerate(heads)},
    )
    (output / "graph-reference.json").write_text(
        json.dumps(
            dict(
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                depth=6,
                reference="JAX CPU research graph, no ONNX outputs used",
                tolerance=dict(atol=2e-4, rtol=2e-4),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    build(**vars(p.parse_args()))
