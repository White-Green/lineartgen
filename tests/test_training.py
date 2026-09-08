import jax
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image

from lineartgen.config import ExperimentConfig
from lineartgen.training import checkpoints
from lineartgen.training.trainer import Trainer


def test_resume_matches_uninterrupted_training(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    for index in range(5):
        pixels = np.random.default_rng(index).integers(0, 256, (20, 20), np.uint8)
        Image.fromarray(pixels).save(root / "images" / f"{index}.png")
    config = ExperimentConfig.model_validate(
        {
            "model": {"channels": [2, 4], "convolutions_per_block": 1},
            "data": {
                "root": root,
                "scale_dirs": ["images"],
                "crop_size": 16,
                "validation_count": 1,
                "workers": 1,
            },
            "diffusion": {"recursion_levels": 2, "sample_steps": 1},
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
