"""Compare exported heads and final raster against the JAX research implementation."""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from attraction_graph import decode
from attraction_stress import prepared
from learned_centerline_model import infer_heads, load_weights
from learned_centerline_study import crop, make_model
from lineartgen_runtime.graph import GraphLineartModel, normalized_guide, render
from PIL import Image, ImageDraw
from projected_edge_gate import gate

from lineartgen.diffusion import source_guides


def validate(root, checkpoint, exported, inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    model, _ = make_model(root)
    load_weights(model, checkpoint)
    runtime = GraphLineartModel(exported / "graph-model.json", provider="cpu")
    rows = []
    for name in ("probe-00", "probe-36", "probe-54", "probe-56", "scene-3"):
        with np.load(inputs / "inputs" / (name + ".npz")) as z:
            source = z["source"]
        panels = [source]
        labels = [name + " source"]
        for depth, mode in ((1, "whole"), (3, "whole"), (6, "whole"), (3, "partial")):
            before = time.monotonic()
            p, mask = prepared(source, depth, mode)
            expected = [np.asarray(h) for h in infer_heads(model, p)]
            guides = source_guides(model, p.guides)
            np.testing.assert_allclose(
                normalized_guide(np.asarray(p.guides[0])),
                np.asarray(guides[0]),
                atol=1e-7,
                rtol=1e-6,
            )
            actual = runtime.heads(np.asarray(p.guides[0]), np.asarray(p.noise_levels[0]), depth)
            for a, e in zip(actual, expected, strict=True):
                np.testing.assert_allclose(a, e, atol=2e-4, rtol=2e-4)
            ref, _ = decode(gate(expected[0][0], base_radius=1.0)[0], base_radius=1.0)
            got = render(actual[0][0])
            ref = np.where(mask > 0, crop(ref), source)
            got = np.where(mask > 0, crop(got), source)
            np.testing.assert_array_equal(got[mask == 0], source[mask == 0])
            delta = np.abs(ref - got)
            # Float roundoff can move a near-zero discrete logit across zero.
            # Record this explicitly, and keep the final raster review visible.
            row = dict(
                name=name,
                depth=depth,
                mode=mode,
                head_max_abs_error=max(
                    float(np.abs(a - e).max()) for a, e in zip(actual, expected, strict=True)
                ),
                raster_max_abs_error=float(delta.max()),
                raster_pixels_over_one_byte=int((delta > 2 / 255).sum()),
                categorical_sign_changes=int(
                    (
                        (actual[0][..., [0, 4, 5, 6, 7, 8]] >= 0)
                        != (expected[0][..., [0, 4, 5, 6, 7, 8]] >= 0)
                    ).sum()
                ),
                protected_exact=True,
                seconds=time.monotonic() - before,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            panels.extend([ref, got])
            labels.extend([f"JAX d{depth} {mode}", f"ONNX d{depth} {mode}"])
        n = source.shape[0]
        canvas = Image.new("RGB", (n * len(panels), n + 28), "#ddd")
        d = ImageDraw.Draw(canvas)
        for i, (a, label) in enumerate(zip(panels, labels, strict=True)):
            d.text((i * n + 3, 3), label, fill="black")
            canvas.paste(
                Image.fromarray(np.rint((a[..., 0] + 1) * 127.5).astype(np.uint8)).convert("RGB"),
                (i * n, 28),
            )
        canvas.save(output / (name + ".png"))
        jax.clear_caches()
    # Verify overlap/cropping on a spatially varying, non-square stage input.
    rng = np.random.default_rng(5)
    g = rng.uniform(0.2, 1, (1, 192, 320, 1)).astype(np.float32)
    noise = rng.uniform(0, 0.75, g.shape).astype(np.float32)
    insert = rng.uniform(-1, 1, (1, 96, 160, 1)).astype(np.float32)
    full = runtime.stage(g, noise, insert, 0, tile_size=None)
    tiled = runtime.stage(g, noise, insert, 0, tile_size=64)
    np.testing.assert_allclose(tiled, full, atol=2e-4, rtol=2e-4)
    report = dict(
        rows=rows,
        tile_max_abs_error=float(np.abs(tiled - full).max()),
        heads_passed=True,
        protected_exact=True,
        visually_reviewed=False,
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "checkpoint", "exported", "inputs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    validate(**vars(args))
