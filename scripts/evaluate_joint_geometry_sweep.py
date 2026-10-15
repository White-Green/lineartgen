"""Re-evaluate all frozen EL width perturbations with the matched models."""

import argparse
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from joint_geometry_study import DIAGNOSTIC, REPORT_METHODS, STEPS, check
from PIL import Image, ImageDraw
from plausible_study import digest
from subpixel_field_model import infer, load_field
from width_response_study import FACTORS, SEED


def evaluate(root):
    check(root)
    paths = {"initial": root / "reference"} | {
        m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in REPORT_METHODS
    }
    models = {m: load_field(p)[0] for m, p in paths.items()}
    states = {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    out = root / "width-sweep"
    out.mkdir(exist_ok=False)
    bindings = {}
    pages = []
    maximum = 0.0
    for i in range(58, 64):
        canvas = Image.new("RGB", (1536, 4 * 280 + 30), "white")
        pen = ImageDraw.Draw(canvas)
        for col, factor in enumerate(FACTORS):
            ip = DIAGNOSTIC / f"evaluation/wide-{i}-factor-{factor}.npz"
            with np.load(ip) as z:
                source = z["source"]
                old = z["prediction"]
            bindings[str(ip)] = sha(ip)
            prepared = prepare_inputs(
                source[None],
                source[None],
                np.full_like(source[None], 0.5),
                jax.random.PRNGKey(SEED),
                3,
            )
            predictions = {m: np.asarray(infer(model, prepared))[0] for m, model in models.items()}
            for value in predictions.values():
                assert np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1
            np.testing.assert_allclose(predictions["initial"], old, atol=1e-4, rtol=1e-4)
            maximum = max(maximum, float(abs(predictions["initial"] - old).max()))
            pp = out / ip.name
            np.savez_compressed(pp, source=source, **predictions)
            bindings[str(pp)] = sha(pp)
            pen.text((col * 256 + 4, 5), f"probe{i} x{factor}", fill="black")
            for row, (label, value) in enumerate([("source", source)] + list(predictions.items())):
                pen.text((col * 256 + 4, 30 + row * 280), label, fill="black")
                px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(px).convert("RGB"), (col * 256, 50 + row * 280))
        dest = out / f"native-wide-{i}.png"
        canvas.save(dest)
        pages.append(dict(file=dest.name, sha256=sha(dest)))
    assert states == {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    for p in paths.values():
        for f in p.iterdir():
            if f.is_file():
                bindings[str(f)] = sha(f)
    write(out / "manifest.json", dict(records=pages))
    write(
        out / "metrics.json",
        dict(
            outputs=108,
            states_unchanged=True,
            initial_cpu_gpu_max_abs_difference=maximum,
            bindings=bindings,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    evaluate(p.parse_args().root.resolve())
