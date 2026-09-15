"""Render selected width comparisons and check them against the completed fixed evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from PIL import Image, ImageDraw

from lineartgen.diffusion import sample, sample_prepared
from lineartgen.evaluation import protected_metrics
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.models.serialization import load_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", default="strokes")
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--mode", choices=["raw", "ema"], default="raw")
    parser.add_argument("--cases", type=int, nargs="+", default=[256, 479, 480, 1376])
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "snapshots" / f"step-{args.step}" / args.method / args.mode
    model, config = load_inference(source)
    manifest_path = root / "evaluation/cases.json"
    manifest = json.loads(manifest_path.read_text())
    report_path = root / "evaluation" / f"step-{args.step}-{args.mode}" / "metrics.json"
    report = json.loads(report_path.read_text())
    expected = {
        (r["case"], r["subnet"]): r
        for r in report["results"]
        if r["method"] == args.method and r["denoise_steps"] == 1
    }
    output = root / "evaluation" / f"widths-{args.step}-{args.method}-{args.mode}"
    output.mkdir(exist_ok=True)
    infer = nnx.jit(
        lambda m, x, t, key, levels: sample(m, x, key, strength=t, steps=1, levels=levels),
        static_argnums=(4,),
    )
    infer_prepared = nnx.jit(lambda m, x, t: sample_prepared(m, x, t, steps=1))
    records = []
    for index in args.cases:
        case = manifest["cases"][index]
        with np.load(root / "evaluation" / case["file"], allow_pickle=False) as arrays:
            clean, strength = arrays["image"], arrays["strength"]
            rough = arrays["source"] if "source" in arrays else clean
        target = np.where(strength > 0, clean, rough)
        levels = expected[index, 0].get(
            "recursion_levels",
            config.diffusion.levels_for_scale(512, config.model.minimum_size, case["scale_level"]),
        )
        x, t = jax.device_put((rough[None], strength[None]))
        key = jax.random.PRNGKey(case["seed"])
        noise_key = key if levels == 1 else jax.random.split(key, levels)[0]
        noisy = np.asarray(
            x * jax.numpy.sqrt(1 - t) + jax.random.normal(noise_key, x.shape) * jax.numpy.sqrt(t)
        )[0]
        prepared = None
        if report.get("noise_generator") == "numpy":
            host = runtime_inputs(rough[None], strength[None], case["seed"], levels)
            noisy = host[0][0][0]
            prepared = jax.device_put(host)
        pictures, measurements = [target, rough, noisy], []
        for subnet in range(3):
            prediction = np.asarray(
                infer_prepared(model.extract(subnet), *prepared)
                if prepared is not None
                else infer(model.extract(subnet), x, t, key, levels)
            )[0]
            measured = protected_metrics(prediction, target, strength)
            for metric in ("active_ink_f1", "active_mae", "protected_max_error"):
                reference = expected[index, subnet][metric]
                if reference is None:
                    assert measured[metric] is None
                else:
                    np.testing.assert_allclose(measured[metric], reference, atol=1e-7, rtol=1e-6)
            pictures.append(prediction)
            measurements.append(dict(subnet=subnet, **measured))
        pixels = np.rint(
            np.clip((np.concatenate(pictures, axis=1)[..., 0] + 1) * 127.5, 0, 255)
        ).astype(np.uint8)
        canvas = Image.new("L", (pixels.shape[1], 536), 255)
        canvas.paste(Image.fromarray(pixels), (0, 24))
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(
            ("target (preserve mask)", "source", "noisy input", "S 5933", "M 10206", "L 17893")
        ):
            draw.text((col * 512 + 8, 5), label, fill=0)
        path = output / f"case-{index:03d}.png"
        canvas.save(path)
        records.append(
            dict(
                case=index,
                metadata=case,
                recursion_levels=levels,
                metrics=measurements,
                image=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    (output / "verification.json").write_text(
        json.dumps(
            dict(
                method=args.method,
                mode=args.mode,
                step=args.step,
                denoise_steps=1,
                records=records,
                weights_sha256=hashlib.sha256((source / "weights.npz").read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                metrics_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                selection="Explicit case indices; numerical metrics reproduce the fixed suite. "
                "See the experiment report for the selection timing.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
