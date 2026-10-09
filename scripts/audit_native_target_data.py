"""CPU data audit before freezing native-target training."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment

root = Path("runs/native-target-training-20261009").resolve()
plan = read(root / "plan.json")
parent = Path(plan["parent"])
config = load_config(root / "configs/native.json")
old = load_config(parent / "configs/mixed.json")
a, b = config.model_dump(mode="json"), old.model_dump(mode="json")
a["data"]["native_training_targets"] = False
assert a == b
sources = discover(config.data)
training, _, _ = split_experiment(sources, config.data, config.train.seed)
expected = [json.loads(x) for x in (parent / "training/training.jsonl").read_text().splitlines()]
for row in expected:
    step = row["step"] - 1
    stream = batches(training, old.data, batch_size=4, count=1, seed=old.train.seed, start=step)
    try:
        batch = next(stream)
    finally:
        stream.close()
    clean, _ = mix_geometry(
        batch.images, old.geometry, seed=old.train.seed, step=step, image_scale=old.data.image_scale
    )
    assert hashlib.sha256(clean.tobytes()).hexdigest() == row["clean_sha256"]
    assert batch.scale_level == row["scale_level"]


def take(data, start, count):
    stream = batches(training, data, batch_size=4, count=count, seed=old.train.seed, start=start)
    try:
        return list(stream)
    finally:
        stream.close()


full = take(config.data, 0, 12)
resumed = take(config.data, 6, 6)
for original, restored in zip(full[6:], resumed, strict=True):
    np.testing.assert_array_equal(original.images, restored.images)
    assert original.scale_level == restored.scale_level
old_batches = take(old.data, 0, 12)
assert [x.scale_level for x in full] == [x.scale_level for x in old_batches]
# Scale choices are indexed solely by seed/batch index and unchanged across all updates.
counts = Counter()
for step in range(4000):
    level = int(np.random.default_rng([old.train.seed, step, 1]).integers(6))
    depth = config.diffusion.levels_for_scale(512, config.model.minimum_size, level)
    counts[depth] += 1
old_counts = Counter()
for segment in ("0-6", "6-1000", "1000-4000"):
    old_counts.update(
        {
            int(k): v
            for k, v in read(parent / f"training/segment-{segment}.json")["depth_counts"].items()
        }
    )
assert counts == old_counts
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        replayed_old_clean_hashes=len(expected),
        old_clean_hashes_all_exact=True,
        config_only_difference="data.native_training_targets",
        depth_counts=counts,
        all_virtual_depths_unchanged=True,
        script_sha256=sha(Path(__file__)),
    ),
)
print(read(root / "data-audit.json"), flush=True)
