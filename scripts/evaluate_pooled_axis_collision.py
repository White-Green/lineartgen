"""Reuse GO's fixed constructed pair as a descriptive mechanism diagnostic."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from PIL import Image, ImageDraw
from plausible_study import digest
from pooled_axis_model import infer, load_field
from pooled_axis_study import GO, STEPS, check
from pooled_axis_study import REPORT_METHODS as METHODS

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
check(root)
assert jax.devices()[0].platform != "cpu"
out = root / "diagnostics/collision"
out.mkdir(parents=True, exist_ok=False)
ip = GO / "inputs.npz"
with np.load(ip) as data:
    source, clean = data["source"], data["clean"]
paths = dict(initial=root / "reference") | {
    m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS
}
models = {m: load_field(p)[0] for m, p in paths.items()}
states = {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
bindings = {str(ip): sha(ip)}
for folder in paths.values():
    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
records, differences, queries = [], [], {}
for depth in (1, 3, 6):
    prepared = prepare_inputs(
        clean, source, np.full_like(source, 0.5), jax.random.PRNGKey(2026101704), depth
    )
    predictions = {m: np.asarray(infer(v, prepared)) for m, v in models.items()}
    for m, value in predictions.items():
        assert np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1
        if m == "initial":
            original = GO / f"predictions-{m}-{depth}.npz"
            bindings[str(original)] = sha(original)
            with np.load(original) as old:
                np.testing.assert_allclose(value, old["prediction"], atol=1e-4, rtol=0)
                differences.append(
                    dict(method=m, depth=depth, max_abs=float(abs(value - old["prediction"]).max()))
                )
    queries[str(depth)] = {m: v[:, 128, 85, 0].tolist() for m, v in predictions.items()}
    file = out / f"predictions-{depth}.npz"
    np.savez_compressed(file, **predictions)
    bindings[str(file)] = sha(file)
    canvas = Image.new("RGB", (1280, 584), "white")
    pen = ImageDraw.Draw(canvas)
    for i in range(2):
        values = dict(
            source=source[i], target=clean[i], **{m: v[i] for m, v in predictions.items()}
        )
        for col, (label, value) in enumerate(values.items()):
            pen.text(
                (col * 256 + 3, i * 292 + 6),
                f"d{depth} {'fill' if i else 'bar'} / {label}",
                fill="black",
            )
            px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
            canvas.paste(Image.fromarray(px).convert("RGB"), (col * 256, i * 292 + 36))
    file = out / f"native-depth-{depth}.png"
    canvas.save(file)
    records.append(dict(file=file.name, sha256=sha(file)))
assert states == {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
write(out / "manifest.json", dict(records=records, native_scale=True))
write(
    out / "metrics.json",
    dict(
        outputs=18,
        states_unchanged=True,
        reference_cpu_gpu_atol=1e-4,
        reference_differences=differences,
        query_pixels=queries,
        bindings=bindings,
        descriptive_only=True,
    ),
)
print(dict(outputs=18, pages=3, query_pixels=queries), flush=True)
