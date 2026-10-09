"""CPU data audit before freezing uniform-loss training."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from train_procedural_structure import prepare_batch

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment

root = Path("runs/source-flat-loss-20261009").resolve()
plan = read(root / "plan.json")
parent = Path(plan["parent"])
config = load_config(root / "configs/flat.json")
old = load_config(parent / "configs/native.json")
a, b = config.model_dump(mode="json"), old.model_dump(mode="json")
a["loss"]["noise_weight_floor"] = old.loss.noise_weight_floor
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
for a, b in zip(full, old_batches, strict=True):
    np.testing.assert_array_equal(a.images, b.images)
# Compare all prepared arrays, not just clean crops, at every recursive depth.
prepare_old, prepare_flat = prepare_batch(old), prepare_batch(config)
for depth in range(1, 7):
    clean = jnp.asarray(full[depth - 1].images)
    key = jax.random.fold_in(jax.random.PRNGKey(old.train.seed), depth - 1)
    a, b = prepare_old(clean, key, depth), prepare_flat(clean, key, depth)
    jax.tree.map(np.testing.assert_array_equal, a, b)
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
        prepared_arrays_all_depths_exact=True,
        config_only_difference="loss.noise_weight_floor",
        depth_counts=counts,
        all_virtual_depths_unchanged=True,
        script_sha256=sha(Path(__file__)),
    ),
)
print(read(root / "data-audit.json"), flush=True)
