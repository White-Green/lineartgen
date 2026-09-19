import importlib
import json
import shutil
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from PIL import Image

from lineartgen.config import ExperimentConfig, load_config, save_config
from lineartgen.data.images import prepare_image


@pytest.mark.parametrize("skip_mode", ["post_pool", "pre_pool"])
def test_png_to_tiles_preserves_resumed_model_optimizer_ema_and_inputs(
    tmp_path, monkeypatch, skip_mode
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    trainer = importlib.import_module("compare_nested_training")
    migration = importlib.import_module("migrate_campaign_cache")
    images, cache = tmp_path / "data/images", tmp_path / "tiles"
    images.mkdir(parents=True)
    for i in range(5):
        pixels = np.full((24, 24, 4), 255, np.uint8)
        pixels[:, 3 + i : 6 + i, :3] = [20, 70, 140]
        pixels[:8, :, 3] = 100
        path = images / f"{i}.png"
        Image.fromarray(pixels).save(path)
        prepare_image(path, cache / "images" / f"{i}.png.tif")
    config = ExperimentConfig.model_validate(
        dict(
            model=dict(
                architecture="nested",
                channels=[2, 3],
                nested_channels=[[3, 5]],
                convolutions_per_block=1,
                skip_mode=skip_mode,
                bottleneck_dilation=2,
                output_mode="residual",
            ),
            data=dict(
                root=str(images.parent),
                scale_dirs=["images"],
                crop_size=16,
                validation_count=1,
                workers=1,
                alpha_mode="white",
            ),
            diffusion=dict(pattern_weights=[1, 1, 1, 0], sample_steps=1),
            matching=dict(method="independent", pool_size=2),
            sketch=dict(mode="strokes", probability=0.5),
            geometry=dict(probability=0.2),
            train=dict(
                steps=3,
                warmup_steps=0,
                batch_size=1,
                accumulate_steps=2,
                evaluate_every=3,
                checkpoint_every=1,
                validation_batches=1,
            ),
        )
    )
    reference, interrupted, resumed = (tmp_path / name for name in ("full", "png", "tiff"))
    for root in (reference, interrupted):
        root.mkdir()
        save_config(config, root / "config.json")
        (root / "variants.json").write_text(json.dumps({"model": config.model_dump(mode="json")}))

    def run(root, count, resume=False):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare_nested_training.py",
                "--config",
                str(root / "config.json"),
                "--variants",
                str(root / "variants.json"),
                "--run-dir",
                str(root / "training"),
                "--stop-after",
                str(count),
                *(["--resume"] if resume else []),
            ],
        )
        trainer.main()

    run(reference, 3)
    run(interrupted, 1)
    original = {
        str(p.relative_to(interrupted)): migration.sha(p)
        for p in interrupted.rglob("*")
        if p.is_file()
    }
    shutil.copytree(interrupted, resumed)
    changes = migration.migrate_configs(resumed, cache)
    assert changes and load_config(resumed / "config.json").data.prepared_root == cache
    assert migration.compare_states(interrupted / "training/model", resumed / "training/model", 1)[
        "exact"
    ]
    run(resumed, 2, True)
    jax.tree.map(
        np.testing.assert_array_equal,
        migration.restored_payload(reference / "training/model", 3),
        migration.restored_payload(resumed / "training/model", 3),
    )
    fields = ("step", "scale_level", "recursion_levels", "strengths", "mask_areas", "generated")
    logs = [
        [json.loads(line) for line in (root / "training/timings.jsonl").read_text().splitlines()]
        for root in (reference, resumed)
    ]
    # Model/optimizer equality above covers all consumed augmented inputs, not just log summaries.
    assert [{k: row[k] for k in fields} for row in logs[0]] == [
        {k: row[k] for k in fields} for row in logs[1]
    ]
    for name, digest in original.items():
        assert migration.sha(interrupted / name) == digest
