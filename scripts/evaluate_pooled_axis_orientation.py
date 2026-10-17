"""Inspect every final readout on four fixed cases; do not select a component."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from learned_coarse_axis_model import infer, load_field
from PIL import Image, ImageDraw
from plausible_study import digest
from pooled_axis_study import AMBIGUITY, STEPS, check

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
out = root / "diagnostics/orientation"
model, _ = load_field(root / f"snapshots/step-{STEPS}/consistent/raw")
before = digest(nnx.state(model, nnx.Param))
bindings, cases = {}, []
with np.load(out / "square48-input.npz") as z:
    cases.append(("square48", z["source"], 1, None))
with np.load(AMBIGUITY / "inputs.npz") as z:
    cases.append(("bar62", z["source"][0], 3, "diagnostics/ambiguity/depth-3.npz"))
    bindings[str((AMBIGUITY / "inputs.npz").resolve())] = sha(AMBIGUITY / "inputs.npz")
with np.load(root / "width-sweep/wide-62-factor-2.0.npz") as z:
    cases.append(("wide62x2", z["source"], 3, "width-sweep/wide-62-factor-2.0.npz"))
with np.load(root / "scene512/input-7.npz") as z:
    cases.append(("scene7uint8", z["quantized"], 6, "scene512/predictions-7-whole-0.npz"))
pages, rows = [], []
for name, source, depth, expected_file in cases:
    prepared = prepare_inputs(
        source[None],
        source[None],
        np.full_like(source[None], 0.5),
        jax.random.PRNGKey(2026101621),
        depth,
    )
    predictions = []
    for g in range(8):
        model.head.component = g
        pred = np.asarray(infer(model, prepared))[0]
        assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
        predictions.append(pred)
    if expected_file:
        with np.load(root / expected_file) as z:
            expected = z["consistent_uint8"] if name == "scene7uint8" else z["consistent"]
            if name == "bar62":
                expected = expected[0]
            np.testing.assert_array_equal(predictions[0], expected)
        bindings[str(root / expected_file)] = sha(root / expected_file)
    file = out / f"{name}-predictions.npz"
    np.savez_compressed(file, source=source, predictions=np.stack(predictions))
    bindings[str(file)] = sha(file)
    size = source.shape[0]
    canvas = Image.new("RGB", (3 * size, 3 * (size + 26)), "white")
    pen = ImageDraw.Draw(canvas)
    for i, value in enumerate([source, *predictions]):
        x, y = (i % 3) * size, (i // 3) * (size + 26)
        px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
        canvas.paste(Image.fromarray(px).convert("RGB"), (x, y + 26))
        pen.text(
            (x + 3, y + 4),
            name + " / " + ("source" if i == 0 else f"component{i - 1}"),
            fill="black",
        )
    page = out / f"native-{name}.png"
    canvas.save(page)
    pages.append(dict(file=page.name, sha256=sha(page)))
    rows.append(dict(case=name, depth=depth, component0_reference_exact=expected_file is not None))
assert before == digest(nnx.state(model, nnx.Param))
write(out / "manifest.json", dict(records=pages, native_scale=True, component_selection=False))
write(out / "metrics.json", dict(outputs=32, states_unchanged=True, rows=rows, bindings=bindings))
