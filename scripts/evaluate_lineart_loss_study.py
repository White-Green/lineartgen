"""Fixed development evaluation with continuous halo metrics and white-composited outputs."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from lineartgen_runtime.images import pyramid_plan, to_bgra
from PIL import Image, ImageDraw
from train_lineart_loss_study import verify

from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation import edit_boundary_metrics, probe_metrics, protected_metrics
from lineartgen.evaluation_sampling import background_metrics, runtime_inputs
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart_study import study_methods


def summaries(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        group = tuple(row[k] for k in ("method", "domain", "family", "severity", "denoise_steps"))
        groups[group].append(row)
    return [
        dict(zip(("method", "domain", "family", "severity", "denoise_steps"), group, strict=True))
        | dict(cases=len(items))
        | {
            key: float(np.mean([r[key] for r in items if r.get(key) is not None]))
            if any(r.get(key) is not None for r in items)
            else None
            for key in sorted(keys)
        }
        for group, items in sorted(groups.items())
    ]


def render(folder, index, iterations, target, source, noisy, predictions):
    images = [target, source, noisy, *predictions.values()]
    labels = ["target (preserve mask)", "source", "noisy", *predictions]
    size = target.shape[0]
    for display in ("continuous", "krita-white"):
        canvas = Image.new("L", (size * len(images), size + 24), 255)
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            pixels = np.rint(np.clip((image[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
            if display == "krita-white" and column >= 3:
                bgra = np.frombuffer(
                    to_bgra(image[None], pyramid_plan(size, size)), np.uint8
                ).reshape(size, size, 4)
                pixels = 255 - bgra[..., 3]
            canvas.paste(Image.fromarray(pixels), (column * size, 24))
            ImageDraw.Draw(canvas).text((column * size + 5, 5), label, fill=0)
        canvas.save(folder / f"case-{index:04d}-steps-{iterations}-{display}.png")


def evaluate(root, step, mode, *, diagnostic=False, limit=None):
    protocol = verify(root)
    reference = read(root / "evaluation.json")
    cases = Path(reference["root"])
    assert sha(cases / "cases.json") == reference["cases_sha256"]
    manifest = read(cases / "cases.json")
    suffix = "-diagnostic" if diagnostic else ""
    if limit:
        suffix += f"-preflight-{limit}"
    folder = root / "evaluation" / f"step-{step}-{mode}{suffix}"
    if (folder / "metrics.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    names = ("initial",) if step == 0 else study_methods(protocol)
    model_paths = {
        name: Path(protocol["initial"])
        if step == 0
        else root / "snapshots" / f"step-{step}" / name / mode
        for name in names
    }
    models = {name: load_inference(path)[0] for name, path in model_paths.items()}
    inference = nnx.jit(lambda m, x, t, s: sample_prepared(m, x, t, steps=s), static_argnums=3)
    indices = protocol["selected_panels"] if diagnostic else list(range(len(manifest["cases"])))
    if limit:
        indices = indices[:limit]
    rows, metric_keys = [], set()
    for offset, index in enumerate(indices):
        case = manifest["cases"][index]
        assert sha(cases / case["file"]) == reference["files"][case["file"]]
        with np.load(cases / case["file"], allow_pickle=False) as arrays:
            clean, strength = arrays["image"], arrays["strength"]
            source = arrays["source"] if "source" in arrays else clean
            target = np.where(strength > 0, clean, source)
            probes = arrays["probes"] if "probes" in arrays else None
            connections = arrays["connections"] if "connections" in arrays else None
        prepared = runtime_inputs(source[None], strength[None], case["seed"], levels=3)
        noisy = prepared[0][0][0]
        prepared = jax.device_put(prepared)
        for iterations in (1, 2, 5) if diagnostic else (1,):
            predictions = {}
            for name, model in models.items():
                prediction = np.asarray(inference(model, *prepared, iterations))[0]
                assert np.isfinite(prediction).all(), (index, name)
                extra = (
                    probe_metrics(prediction, target, probes, connections)
                    if probes is not None and not case["partial"]
                    else {}
                )
                measured = (
                    protected_metrics(prediction, target, strength)
                    | background_metrics(prediction, target, strength)
                    | edit_boundary_metrics(prediction, target, strength)
                    | halo_metrics(prediction, target, strength)
                    | extra
                )
                assert measured["protected_max_error"] in (None, 0.0), (index, name, measured)
                assert all(v is None or np.isfinite(v) for v in measured.values())
                metric_keys.update(measured)
                rows.append(
                    dict(
                        case=index,
                        method=name,
                        domain="real" if case["kind"] == "real" else "geometry",
                        kind=case["kind"],
                        source=case.get("source"),
                        suite=case["suite"],
                        family=case.get("family", "clean"),
                        severity=case.get("severity", "clean"),
                        strength=case["strength"],
                        partial=case["partial"],
                        denoise_steps=iterations,
                        **measured,
                    )
                )
                predictions[name] = prediction
            if index in protocol["selected_panels"]:
                render(folder, index, iterations, target, source, noisy, predictions)
        if (offset + 1) % 50 == 0 or offset + 1 == len(indices):
            print(f"Evaluated {offset + 1}/{len(indices)} cases", flush=True)
    write(
        folder / "metrics.json",
        dict(
            step=step,
            mode=mode,
            diagnostic=diagnostic,
            preflight_limit=limit,
            case_indices=indices,
            model_weights_sha256={n: sha(p / "weights.npz") for n, p in model_paths.items()},
            manifest_sha256=reference["cases_sha256"],
            prediction_count=len(rows),
            results=rows,
            summary=summaries(rows, metric_keys),
            recursion_levels=3,
            noise="fixed NumPy runtime draws",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mode", choices=("raw", "ema"), default="ema")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluate(
        args.root.resolve(), args.step, args.mode, diagnostic=args.diagnostic, limit=args.limit
    )


if __name__ == "__main__":
    main()
