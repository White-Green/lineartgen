import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest
from PIL import Image

from lineartgen.config import ExperimentConfig
from lineartgen.training import checkpoints
from lineartgen.training.trainer import Trainer


@pytest.mark.parametrize("nested", [False, True])
def test_resume_matches_uninterrupted_training(tmp_path, nested):
    root = tmp_path / "dataset"
    for name, size in (("images", 20), ("images_2x", 40)):
        (root / name).mkdir(parents=True)
        for index in range(5):
            pixels = np.random.default_rng(index).integers(0, 256, (size, size), np.uint8)
            Image.fromarray(pixels).save(root / name / f"{index}.png")
    config = ExperimentConfig.model_validate(
        {
            "model": {"channels": [2, 4], "convolutions_per_block": 1},
            "data": {
                "root": root,
                "scale_dirs": ["images", "images_2x"],
                "crop_size": 16,
                "validation_count": 1,
                "workers": 1,
            },
            "diffusion": {"original_steps": 1, "sample_steps": 1},
            "matching": {"pool_size": 4},
            "train": {
                "steps": 2,
                "warmup_steps": 0,
                "batch_size": 1,
                "accumulate_steps": 2,
                "evaluate_every": 2,
                "checkpoint_every": 1,
                "validation_batches": 1,
            },
        }
    )
    if nested:
        config = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"architecture": "nested", "nested_channels": ((3, 5),)}
                )
            }
        )
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert Trainer(config, full).fit() == 2
    assert Trainer(config, resumed).fit(stop_after=1) == 1
    assert Trainer(config, resumed).fit(resume=True) == 2
    states = []
    for run in (full, resumed):
        with checkpoints.manager(run / "checkpoints") as saved:
            states.append(saved.restore(2, args=ocp.args.StandardRestore()))
        assert (run / "samples" / "step-00000002.png").is_file()
    # Compare parameters, optimizer moments/counters and EMA, not just loss.
    jax.tree.map(np.testing.assert_array_equal, *states)
