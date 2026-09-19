"""Open the reserved sources only after model selection has been sealed."""

import argparse
import datetime
import hashlib
from functools import partial
from pathlib import Path

import jax
import numpy as np
from analyze_completion_campaign import check_seal
from campaign_protocol import read, sha, write
from evaluate_training_methods import data_settings

from lineartgen.config import SketchAugmentationConfig, load_config
from lineartgen.data.images import read_crop
from lineartgen.data.pipeline import dataset_description, discover, split_experiment
from lineartgen.evaluation import KINDS, line_case
from lineartgen.sketch import augment_sketch


def prepare(root):
    check_seal(root)  # Must precede discovery or decoding of reserved sources.
    destination = root / "final"
    if destination.exists():
        assert read(destination / "cases.json")["selection_seal_sha256"] == sha(
            root / "selection-seal.json"
        )
        for name, digest in read(destination / "input-sha256.json").items():
            assert sha(destination / name) == digest
        return destination
    config = load_config(root / "config.json")
    protocol = read(root / "protocol.json")
    sources = discover(config.data)
    train, dev, final = split_experiment(sources, config.data, config.train.seed)
    assert len(final) == 64
    final_names = {s.name for s in final}
    assert not final_names & {s.name for s in (*train, *dev)}
    folder = root / "final-building"
    if folder.exists():
        folder.rename(root / f"final-interrupted-{datetime.datetime.now().timestamp():.6f}")
    folder.mkdir()
    random = np.random.default_rng(protocol["final_seed"])
    ranges = {
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
            partial(
                augment_sketch,
                config=SketchAugmentationConfig(mode=family, probability=1.0, **ranges[severity]),
            )
        )
        for family, severity in (("strokes", "mild"), ("strokes", "strong"), ("blur", "strong"))
    }
    records = []
    targets = []

    def normalized(gray):
        return np.clip((gray * 2 - 1) * config.data.image_scale, -1, 1).astype(np.float32)

    def add_target(clean, record, probes=None, connections=None):
        target_index = len(targets)
        key = jax.random.PRNGKey(protocol["final_seed"] + 1000 + target_index)
        rough = {(f, s): np.asarray(fn(clean[None], key))[0] for (f, s), fn in operations.items()}
        inputs = {(None, None): clean, **rough}
        targets.append(
            dict(
                **record,
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
            name = f"case-{index:03d}.npz"
            np.savez_compressed(folder / name, **arrays)
            suite = (
                ("matched_real" if record["kind"] == "real" else "geometry")
                if family is None
                else ("sketch_real" if record["kind"] == "real" else "sketch_geometry")
            )
            records.append(
                dict(
                    record,
                    file=name,
                    suite=suite,
                    family=family,
                    severity=severity,
                    strength=strength,
                    partial=partial_edit,
                    seed=protocol["final_seed"] + 100000 + index,
                    corruption_seed=protocol["final_seed"] + 1000 + target_index,
                )
            )

        for (family, severity), source in inputs.items():
            for strength in (0.1, 0.5):
                for partial_edit in (False, True):
                    store(source, family, severity, strength, partial_edit)
        store(rough["strokes", "strong"], "strokes", "strong", 0.0, False)

    for index, source in enumerate(final):
        variant = index % len(config.data.scale_dirs)
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
    for kind in KINDS:
        for variant in protocol["final_geometry_variants"]:
            gray, probes, connections = line_case(kind, variant)
            add_target(
                normalized(gray),
                dict(kind=kind, variant=variant, scale_level=0),
                probes,
                connections,
            )
    expected = (64 + len(KINDS) * len(protocol["final_geometry_variants"])) * protocol[
        "final_cases_per_source"
    ]
    assert len(records) == expected == 1632
    write(
        folder / "cases.json",
        dict(
            format_version=3,
            dataset=dataset_description(sources, train, dev, final),
            data_config=data_settings(config),
            cases=records,
            targets=targets,
            corruption_ranges=ranges,
            split="final",
            selection_seal_sha256=sha(root / "selection-seal.json"),
            source_sha256=sha(__file__),
            protocol_sha256=sha(root / "protocol.json"),
            limitations="Synthetic corruptions of held-out clean images. "
            "One random crop per source; blanks are retained. No retuning.",
        ),
    )
    write(
        folder / "input-sha256.json",
        {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
    )
    folder.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root.resolve()))


if __name__ == "__main__":
    main()
