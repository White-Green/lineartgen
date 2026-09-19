import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image

from lineartgen.config import ExperimentConfig, save_config
from lineartgen.training import checkpoints


def test_explicit_architectures_resume_optimizer_model_and_ema(tmp_path, monkeypatch):
    images = tmp_path / "data/images"
    images.mkdir(parents=True)
    for i in range(5):
        pixels = np.full((20, 20), 255, np.uint8)
        pixels[:, 3 + i : 5 + i] = 0
        Image.fromarray(pixels).save(images / f"{i}.png")
    config = ExperimentConfig.model_validate(
        dict(
            model=dict(
                architecture="nested",
                channels=[2, 3],
                nested_channels=[[3, 5]],
                convolutions_per_block=1,
                skip_mode="pre_pool",
                bottleneck_dilation=2,
                output_mode="residual",
            ),
            data=dict(
                root=str(images.parent),
                scale_dirs=["images"],
                crop_size=16,
                validation_count=1,
                workers=1,
            ),
            diffusion=dict(pattern_weights=[1, 1, 1, 0], sample_steps=1),
            matching=dict(method="independent", pool_size=2),
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
    save_config(config, tmp_path / "config.json")
    variants = dict(
        shared=config,
        core=config.model_copy(
            update={"model": config.model.model_copy(update={"nested_channels": ()})}
        ),
        large=config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"architecture": "unet", "nested_channels": (), "channels": (3, 5)}
                )
            }
        ),
    )
    (tmp_path / "variants.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in variants.items()})
    )
    path = Path(__file__).resolve().parents[1] / "scripts/compare_nested_training.py"
    spec = importlib.util.spec_from_file_location("campaign_test_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    for root, count, resume in ((full, 3, False), (resumed, 1, False), (resumed, 2, True)):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(path),
                "--config",
                str(tmp_path / "config.json"),
                "--variants",
                str(tmp_path / "variants.json"),
                "--run-dir",
                str(root),
                "--stop-after",
                str(count),
                *(["--resume"] if resume else []),
            ],
        )
        module.main()
    for name in variants:
        states = []
        for root in (full, resumed):
            with checkpoints.manager(root / name / "checkpoints") as manager:
                states.append(manager.restore(3, args=ocp.args.StandardRestore()))
        jax.tree.map(np.testing.assert_array_equal, *states)
    with (
        np.load(full / "initial/shared/weights.npz") as shared,
        np.load(full / "initial/core/weights.npz") as core,
    ):
        for name in core.files:
            np.testing.assert_array_equal(shared[name], core[name])
