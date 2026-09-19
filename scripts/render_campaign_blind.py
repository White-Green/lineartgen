"""Render a predetermined, anonymized final-test cohort without prediction-based selection."""

import argparse
from pathlib import Path

import jax
import numpy as np
from analyze_completion_campaign import MODELS, check_seal
from campaign_protocol import read, report, sha, write
from flax import nnx
from PIL import Image, ImageDraw

from lineartgen.diffusion import sample_prepared
from lineartgen.evaluation import protected_metrics
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.models.serialization import load_inference


def panel_plan(protocol):
    """Four corruption/mask conditions, every fourth real and every other geometry source."""
    targets = [*range(0, 64, 4), *range(64, 96, 2)]
    conditions = (8, 11, 12, 15)
    random = np.random.default_rng(protocol["blind_seed"])
    return [
        dict(
            panel=i,
            case=target * protocol["final_cases_per_source"] + conditions[i % 4],
            model_seed=str(protocol["seeds"][i % 3]["model"]),
            mode="raw" if i % 4 == 0 else "ema",
            order=random.permutation(list(MODELS)).tolist(),
        )
        for i, target in enumerate(targets)
    ]


def render(root):
    seal = check_seal(root)
    protocol, cohorts = read(root / "protocol.json"), read(root / "cohorts.json")
    manifest = read(root / "final/cases.json")
    assert manifest["selection_seal_sha256"] == sha(root / "selection-seal.json")
    folder = root / "blind"
    folder.mkdir(exist_ok=True)
    model_cache, report_cache = {}, {}
    infer = nnx.jit(lambda m, x, t: sample_prepared(m, x, t, steps=1))
    public, key = [], []
    for selection in panel_plan(protocol):
        index, seed, mode = selection["case"], selection["model_seed"], selection["mode"]
        case = manifest["cases"][index]
        with np.load(root / "final" / case["file"], allow_pickle=False) as arrays:
            clean, rough, mask = arrays["image"], arrays["source"], arrays["strength"]
        target = np.where(mask > 0, clean, rough)
        prepared = runtime_inputs(rough[None], mask[None], case["seed"], levels=3)
        pictures = [target, rough, prepared[0][0][0]]
        measurements = []
        for label, name in zip("ABCDE", selection["order"], strict=True):
            role, subnet = MODELS[name]
            arm_name = cohorts[seed][role]
            arm = root / "arms" / arm_name
            identity = arm_name, mode
            if identity not in model_cache:
                source = arm / f"snapshots/step-{seal['step']}/model/{mode}"
                model_cache[identity] = load_inference(source)[0]
                fixed = report(arm, seal["step"], mode, root / "final", scope="final-evaluation")
                report_cache[identity] = {(r["case"], r["subnet"]): r for r in fixed["results"]}
            model = model_cache[identity]
            prefix = model.extract(subnet) if hasattr(model, "extract") else model
            prediction = np.asarray(infer(prefix, *jax.device_put(prepared)))[0]
            assert np.isfinite(prediction).all()
            measured = protected_metrics(prediction, target, mask)
            expected = report_cache[identity][index, subnet]
            for metric in ("active_ink_f1", "active_mae", "protected_max_error"):
                if expected[metric] is None:
                    assert measured[metric] is None
                else:
                    np.testing.assert_allclose(
                        measured[metric], expected[metric], atol=1e-7, rtol=1e-6
                    )
            pictures.append(prediction)
            measurements.append(dict(label=label, **measured))
        pixels = np.rint(
            np.clip((np.concatenate(pictures, axis=1)[..., 0] + 1) * 127.5, 0, 255)
        ).astype(np.uint8)
        canvas = Image.new("L", (4096, 536), 255)
        canvas.paste(Image.fromarray(pixels), (0, 24))
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(("target (preserve mask)", "source", "noisy", *"ABCDE")):
            draw.text((col * 512 + 8, 5), label, fill=0)
        path = folder / f"panel-{selection['panel']:02d}.png"
        canvas.save(path)
        # Fixed central enlargement, identical for every panel; no content-based crop search.
        zoom = Image.new("L", (4096, 536), 255)
        for col in range(8):
            tile = canvas.crop((col * 512 + 128, 152, col * 512 + 384, 408))
            zoom.paste(tile.resize((512, 512), Image.Resampling.NEAREST), (col * 512, 24))
        zoom.paste(canvas.crop((0, 0, 4096, 24)), (0, 0))
        zoom_path = folder / f"panel-{selection['panel']:02d}-center2x.png"
        zoom.save(zoom_path)
        public.append(
            dict(
                panel=selection["panel"],
                case=index,
                metadata=case,
                model_seed=seed,
                mode=mode,
                metrics=measurements,
                image=path.name,
                sha256=sha(path),
                zoom=zoom_path.name,
                zoom_sha256=sha(zoom_path),
            )
        )
        key.append(
            dict(
                panel=selection["panel"], labels=dict(zip("ABCDE", selection["order"], strict=True))
            )
        )
        print(f"Rendered panel {selection['panel'] + 1}/32", flush=True)
    write(root / "blind-key.json", dict(panels=key, source_sha256=sha(__file__)))
    write(
        folder / "manifest.json",
        dict(
            panels=public,
            selection="Predetermined source indices and corruption/mask conditions; "
            "no selection by predictions. 24 EMA and 8 raw panels across all three seeds.",
            review="Record qualitative observations before opening blind-key.json. "
            "Agent review is not an independent artist or user study.",
            source_sha256=sha(__file__),
            selection_seal_sha256=sha(root / "selection-seal.json"),
        ),
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.root.resolve()))


if __name__ == "__main__":
    main()
