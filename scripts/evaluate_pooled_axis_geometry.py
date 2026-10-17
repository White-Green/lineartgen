"""Review every predeclared GK variation at three depths for the full-data models."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from learned_coarse_axis_model import infer, load_field
from PIL import Image, ImageDraw
from plausible_study import digest
from pooled_axis_study import GK, LABELS, MODELREF, STEPS, check
from pooled_axis_study import REPORT_METHODS as METHODS

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
out = root / "diagnostics/geometry"
out.mkdir(parents=True, exist_ok=False)
ip = GK / "inputs.npz"
with np.load(ip) as z:
    source, target = z["source"], z["target"]
names = [r["name"] for r in read(GK / "plan.json")["cases"]]
paths = dict(initial=root / "reference") | {
    m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS
}
models = {m: load_field(v)[0] for m, v in paths.items()}
states = {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
bindings = {str(ip.resolve()): sha(ip)}
for path in paths.values():
    bindings.update({str(f): sha(f) for f in path.iterdir() if f.is_file()})
pages = []
for depth in (1, 3, 6):
    reference = root / f"diagnostics/ambiguity/depth-{depth}.npz"
    bindings[str(reference)] = sha(reference)
    with np.load(reference) as z:
        expected = {m: z[m][1] for m in METHODS}
    initial_ref = MODELREF / f"diagnostics/ambiguity/depth-{depth}.npz"
    bindings[str(initial_ref.resolve())] = sha(initial_ref)
    with np.load(initial_ref) as z:
        expected["initial"] = z["mixed"][1]
    values = []
    for i in range(8):
        prepared = prepare_inputs(
            target[i],
            source[i],
            np.full_like(source[i], 0.5),
            jax.random.PRNGKey(2026101701),
            depth,
        )
        pred = {m: np.asarray(infer(v, prepared)) for m, v in models.items()}
        for m, v in pred.items():
            assert np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1
            np.testing.assert_array_equal(v[1], expected[m])
        file = out / f"predictions-{depth}-{i}.npz"
        np.savez_compressed(file, **pred)
        bindings[str(file)] = sha(file)
        values.append(pred)
    for start in (0, 3, 6):
        ids = list(range(start, min(start + 3, 8)))
        canvas = Image.new("RGB", (1280, 292 * len(ids)), "white")
        pen = ImageDraw.Draw(canvas)
        for row, i in enumerate(ids):
            arrays = [source[i, 0], target[i, 0], *[values[i][m][0] for m in models]]
            for col, (label, value) in enumerate(
                zip(("source", "target", *models), arrays, strict=True)
            ):
                pen.text(
                    (col * 256 + 3, row * 292 + 5),
                    f"d{depth} {names[i]} / {LABELS.get(label, label)}",
                    fill="black",
                )
                px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(px).convert("RGB"), (col * 256, row * 292 + 36))
        file = out / f"native-depth-{depth}-{start}.png"
        canvas.save(file)
        pages.append(dict(file=file.name, sha256=sha(file)))
assert states == {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
write(
    out / "metrics.json",
    dict(
        outputs=144,
        primary_outputs=72,
        states_unchanged=True,
        repeated_anchors_exact=True,
        trained_anchors_match_FU_exact=True,
        bindings=bindings,
    ),
)
write(out / "manifest.json", dict(records=pages, native_scale=True))
print(dict(outputs=144, primary_outputs=72, pages=9), flush=True)
