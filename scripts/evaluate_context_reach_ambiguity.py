"""Revisit FU's fixed ambiguity pair after the matched architecture comparison."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from context_reach_model import infer, load_field
from context_reach_study import AMBIGUITY, METHODS, STEPS, check
from cue_capacity import prepare_inputs
from flax import nnx
from PIL import Image, ImageDraw
from plausible_study import digest

from lineartgen.diffusion import source_guides

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
out = root / "diagnostics/ambiguity"
out.mkdir(parents=True, exist_ok=False)
ip = AMBIGUITY / "inputs.npz"
with np.load(ip) as z:
    source, clean = z["source"], z["clean"]
models = {m: load_field(root / f"snapshots/step-{STEPS}/{m}/raw")[0] for m in METHODS}
states = {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
bindings = {str(ip.resolve()): sha(ip)}
rows, pages = [], []
for depth in (1, 3, 6):
    parts = [
        prepare_inputs(
            clean[i : i + 1],
            source[i : i + 1],
            np.full_like(source[i : i + 1], 0.5),
            jax.random.PRNGKey(2026101615),
            depth,
        )
        for i in range(2)
    ]
    prepared = jax.tree.map(lambda a, b: np.concatenate([a, b]), *parts)
    if depth == 1:
        guide = np.asarray(source_guides(models["local"], prepared.guides)[0])
        # Local control radius45: negative x padding is identical in both images.
        np.testing.assert_array_equal(guide[0, 83:174, :79], guide[1, 83:174, :79])
    predictions = {m: np.asarray(infer(model, prepared)) for m, model in models.items()}
    for value in predictions.values():
        assert np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1
    if depth == 1:
        np.testing.assert_array_equal(
            predictions["local"][0, 128, 33], predictions["local"][1, 128, 33]
        )
    file = out / f"depth-{depth}.npz"
    np.savez_compressed(file, **predictions)
    bindings[str(file)] = sha(file)
    rows.append(
        dict(
            depth=depth,
            point_yx=[128, 33],
            target=clean[:, 128, 33, 0].tolist(),
            predictions={m: value[:, 128, 33, 0].tolist() for m, value in predictions.items()},
        )
    )
    canvas = Image.new("RGB", (1024, 584), "white")
    pen = ImageDraw.Draw(canvas)
    for col, (label, value) in enumerate(dict(source=source, target=clean, **predictions).items()):
        pen.text((256 * col + 4, 6), label, fill="black")
        for row in range(2):
            pixels = np.uint8(np.rint(np.clip((value[row, ..., 0] + 1) * 127.5, 0, 255)))
            canvas.paste(Image.fromarray(pixels).convert("RGB"), (256 * col, 48 + 280 * row))
            pen.text(
                (256 * col + 4, 30 + 280 * row),
                f"depth{depth} / " + ("bar" if row == 0 else "fill"),
                fill="black",
            )
    page = out / f"native-depth-{depth}.png"
    canvas.save(page)
    pages.append(dict(file=page.name, sha256=sha(page)))
assert states == {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
write(out / "manifest.json", dict(records=pages, native_scale=True, thresholding=False))
write(
    out / "metrics.json",
    dict(
        outputs=12,
        states_unchanged=True,
        rows=rows,
        bindings=bindings,
        constructed_pair_only=True,
        training=False,
    ),
)
print(rows, flush=True)
