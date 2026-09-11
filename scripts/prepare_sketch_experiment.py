"""Freeze paired rough inputs made exclusively from development clean line art."""

import argparse
import hashlib
import json
import shutil
from functools import partial as bind
from pathlib import Path

import jax
import numpy as np
from evaluate_training_methods import data_settings
from PIL import Image, ImageDraw

from lineartgen.config import SketchAugmentationConfig, load_config
from lineartgen.data.pipeline import dataset_description, discover, split_experiment
from lineartgen.sketch import augment_sketch


def prepare(config, output, legacy):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace frozen cases: {output}")
    old = json.loads((legacy / "cases.json").read_text())
    sources = discover(config.data)
    split = split_experiment(sources, config.data, config.train.seed)
    if old["dataset"] != dataset_description(sources, *split):
        raise ValueError("The clean controls must use the same source split")
    if len(old["cases"]) != 434:
        raise ValueError("Expected the complete previous 434-case suite")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in old["cases"]:
        shutil.copyfile(legacy / case["file"], output / case["file"])
        records.append(case.copy())
    severities = {
        "mild": dict(
            opacity_min=0.55,
            opacity_max=0.9,
            blur_sigma_min=0.5,
            blur_sigma_max=1.0,
            dilation_radius=1,
            displacement=1.0,
        ),
        "strong": dict(
            opacity_min=0.2,
            opacity_max=0.45,
            blur_sigma_min=1.5,
            blur_sigma_max=2.5,
            dilation_radius=2,
            displacement=3.0,
        ),
    }
    operations = {
        (family, severity): jax.jit(
            bind(
                augment_sketch,
                config=SketchAugmentationConfig(mode=family, probability=1, **ranges),
            )
        )
        for family in ("tone", "blur", "strokes")
        for severity, ranges in severities.items()
    }
    bases = {}
    for case in old["cases"]:
        if case["suite"] == "matched_real":
            bases.setdefault(("real", case["matched_crop"]), case)
        elif case["suite"] == "geometry" and case["variant"] in (0, 1, 6, 7):
            bases.setdefault((case["kind"], case["variant"]), case)
    assert len(bases) == 44
    for base_index, case in enumerate(bases.values()):
        with np.load(legacy / case["file"], allow_pickle=False) as archive:
            clean = archive["image"]
            probes = {k: archive[k] for k in ("probes", "connections") if k in archive}
        previews, labels = [clean], ["clean"]
        for family in ("tone", "blur", "strokes"):
            for severity in severities:
                # Same component draws across the cumulative families; independent of training.
                corruption_seed = 2026091200 + base_index
                rough = np.asarray(
                    operations[family, severity](clean[None], jax.random.PRNGKey(corruption_seed))
                )[0]
                previews.append(rough)
                labels.append(f"{family} {severity}")
                conditions = [(0.1, False), (0.5, False), (0.1, True), (0.5, True)]
                if case["kind"] == "real":
                    conditions.append((0.0, False))
                for strength, partial in conditions:
                    noise = np.full_like(clean, strength)
                    if partial:
                        noise[:102] = 0
                        noise[:, :170] = 0
                        noise[384:, 384:] = 0
                    index = len(records)
                    name = f"case-{index:03d}.npz"
                    np.savez_compressed(
                        output / name, image=clean, source=rough, strength=noise, **probes
                    )
                    records.append(
                        dict(
                            case,
                            file=name,
                            suite="sketch_real" if case["kind"] == "real" else "sketch_geometry",
                            family=family,
                            severity=severity,
                            strength=strength,
                            partial=partial,
                            corruption_seed=corruption_seed,
                            seed=2026091300 + index,
                        )
                    )
        if base_index in (0, 10, 12, 16, 20, 24, 28, 32, 36, 40):
            pixels = np.rint(
                np.clip((np.concatenate(previews, axis=1)[..., 0] + 1) * 127.5, 0, 255)
            ).astype(np.uint8)
            canvas = Image.new("L", (pixels.shape[1], 536), 255)
            canvas.paste(Image.fromarray(pixels), (0, 24))
            draw = ImageDraw.Draw(canvas)
            for column, label in enumerate(labels):
                draw.text((column * 512 + 8, 5), label, fill=0)
            canvas.save(output / f"preview-{base_index:02d}.png")
    assert len(records) == 1562
    manifest = dict(
        format_version=2,
        dataset=old["dataset"],
        data_config=data_settings(config),
        cases=records,
        corruption_ranges=severities,
        limitations="Synthetic only. Strong conditions partly exceed training ranges. "
        "Rough partial-edit topology probes are excluded; active/protected/seam metrics remain.",
    )
    (output / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    (output.parent / "evaluation-input-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        f"Frozen {len(records)} cases from 12 real clean crops and 32 geometric targets", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sketch.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, default=Path("runs/methods-20260911/evaluation"))
    args = parser.parse_args()
    prepare(load_config(args.config), args.output, args.legacy)
