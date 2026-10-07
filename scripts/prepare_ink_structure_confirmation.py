"""Generate the prospective synthetic confirmation panel after checkpoint selection is sealed."""

import argparse
import hashlib
from functools import partial
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from evaluate_training_methods import data_settings

from lineartgen.config import SketchAugmentationConfig, load_config
from lineartgen.data.images import read_crop
from lineartgen.data.pipeline import dataset_description, discover, split_experiment
from lineartgen.evaluation import KINDS, line_case
from lineartgen.sketch import augment_sketch


def check_selection(seal_path, plan_path):
    seal = read(seal_path)
    assert seal["confirmation_plan_sha256"] == sha(plan_path)
    assert seal["selection_notes"] and seal["models"]
    for model in seal["models"].values():
        folder = Path(model["path"])
        for name in ("weights.npz", "metadata.json", "config.json"):
            assert sha(folder / name) == model["files"][name]
    return seal


def prepare(root, selection, destination):
    plan_path = root / "confirmation-plan.json"
    check_selection(selection, plan_path)
    plan, config = read(plan_path), load_config(root / "config.json")
    assert plan["geometry_kinds"] == list(KINDS)
    assert plan["source_split"] == "final" and plan["crop_size"] == 512
    if destination.exists():
        manifest = read(destination / "cases.json")
        assert manifest["selection_seal_sha256"] == sha(selection)
        assert manifest["confirmation_plan_sha256"] == sha(plan_path)
        for name, digest in read(destination / "input-sha256.json").items():
            assert sha(destination / name) == digest
        return destination
    folder = destination.with_name(destination.name + "-building")
    folder.mkdir(parents=True, exist_ok=False)
    sources = discover(config.data)
    training, dev, final = split_experiment(sources, config.data, config.train.seed)
    assert len(final) == plan["source_count"]
    assert dataset_description(sources, training, dev, final) == read(root / "dataset.json")
    random = np.random.default_rng(plan["seed"])
    operations = {
        (family, severity): jax.jit(
            partial(
                augment_sketch,
                config=SketchAugmentationConfig(
                    mode=family, probability=1.0, **plan["corruption_ranges"][severity]
                ),
            )
        )
        for family in plan["families"]
        if family != "clean"
        for severity in plan["severities"][family]
    }
    records, targets = [], []

    def normalized(gray):
        return np.clip((gray * 2 - 1) * config.data.image_scale, -1, 1).astype(np.float32)

    def add_target(clean, description, probes=None, connections=None):
        target_index = len(targets)
        corruption_seed = plan["seed"] + plan["corruption_seed_offset"] + target_index
        key = jax.random.PRNGKey(corruption_seed)
        inputs = {("clean", "clean"): clean} | {
            pair: np.asarray(operation(clean[None], key))[0]
            for pair, operation in operations.items()
        }
        targets.append(
            description
            | dict(
                ink_fraction=float((clean < 0).mean()),
                normalized_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
            )
        )

        def store(source, family, severity, strength, partial_edit):
            index = len(records)
            mask = np.full_like(clean, strength)
            if partial_edit:
                mask[:102] = 0
                mask[:, :170] = 0
                mask[384:, 384:] = 0
            arrays = dict(image=clean, source=source, strength=mask)
            if probes is not None:
                arrays.update(probes=probes, connections=connections)
            name = f"case-{index:04d}.npz"
            np.savez_compressed(folder / name, **arrays)
            records.append(
                description
                | dict(
                    file=name,
                    family=family,
                    severity=severity,
                    strength=strength,
                    partial=partial_edit,
                    suite=("matched_" if family == "clean" else "sketch_")
                    + ("real" if description["kind"] == "real" else "geometry"),
                    seed=plan["seed"] + plan["noise_seed_offset"] + index,
                    corruption_seed=corruption_seed,
                )
            )

        for (family, severity), source in inputs.items():
            for strength in plan["strengths"]:
                for partial_edit in plan["partial_edits"]:
                    store(source, family, severity, strength, partial_edit)
        if plan["per_target_zero_strength_control"]:
            store(inputs["strokes", "strong"], "strokes", "strong", 0.0, False)

    for index, source in enumerate(final):
        variant = index % len(source.variants)
        path, width, height = source.variants[variant]
        assert width >= 512 and height >= 512
        x, y = int(random.integers(width - 512 + 1)), int(random.integers(height - 512 + 1))
        clean = normalized(read_crop(path, x, y, 512, alpha_mode=config.data.alpha_mode))
        add_target(
            clean,
            dict(
                kind="real",
                source=source.name,
                matched_crop=index,
                scale_level=config.data.resolved_scale_levels[variant],
                crop=[x, y, 512],
            ),
        )
    for kind in plan["geometry_kinds"]:
        for variant in plan["geometry_variants"]:
            gray, probes, connections = line_case(kind, variant)
            add_target(
                normalized(gray),
                dict(kind=kind, variant=variant, scale_level=0),
                probes,
                connections,
            )
    target_count = plan["source_count"] + len(plan["geometry_kinds"]) * len(
        plan["geometry_variants"]
    )
    per_target = sum(len(plan["severities"][f]) for f in plan["families"])
    per_target *= len(plan["strengths"]) * len(plan["partial_edits"])
    per_target += int(plan["per_target_zero_strength_control"])
    assert len(targets) == target_count and len(records) == target_count * per_target
    write(
        folder / "cases.json",
        dict(
            format_version=3,
            dataset=dataset_description(sources, training, dev, final),
            data_config=data_settings(config),
            cases=records,
            targets=targets,
            corruption_ranges=plan["corruption_ranges"],
            split="confirmation_new_synthetic_draws",
            selection_seal_sha256=sha(selection),
            confirmation_plan_sha256=sha(plan_path),
            source_sha256=sha(Path(__file__)),
            limitations=plan["source_unseenness"],
        ),
    )
    write(
        folder / "input-sha256.json",
        {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    )
    folder.rename(destination)
    print(dict(destination=str(destination), targets=len(targets), cases=len(records)))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.selection.resolve(), args.destination.resolve())
