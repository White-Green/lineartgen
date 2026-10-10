"""Replay the parent and audit only the declared change in teacher mixture."""

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
from lineartgen.data.coverage import COVERAGE_KINDS
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment
from lineartgen.diffusion import pyramid, restoration_target

root = Path("runs/clearance-geometry-balance-20261010").resolve()
plan = read(root / "plan.json")
parent = Path(plan["parent"])
config = load_config(root / "configs/balance.json")
old = load_config(parent / "configs/clearance.json")
a, b = config.model_dump(mode="json"), old.model_dump(mode="json")
assert a["geometry"]["probability"] == 0.5 and a["geometry"]["coverage_probability"] == 0.2
assert b["geometry"]["probability"] == 0.2 and b["geometry"]["coverage_probability"] == 0.125
a["geometry"]["probability"] = 0.2
a["geometry"]["coverage_probability"] = 0.125
assert a == b
training, _, _ = split_experiment(discover(config.data), config.data, config.train.seed)


def take(start, count):
    stream = batches(
        training, config.data, batch_size=4, count=count, seed=config.train.seed, start=start
    )
    try:
        return list(stream)
    finally:
        stream.close()


def mixed(batch, step, c):
    return mix_geometry(
        batch.images, c.geometry, seed=c.train.seed, step=step, image_scale=c.data.image_scale
    )


expected = [json.loads(x) for x in (parent / "training/training.jsonl").read_text().splitlines()]
assert len(expected) == 166 and expected[-1]["step"] == 4000
new_hashes = {}
counts = Counter()
for row in expected:
    step = row["step"] - 1
    batch = take(step, 1)[0]
    before, old_records = mixed(batch, step, old)
    after, records = mixed(batch, step, config)
    assert hashlib.sha256(before.tobytes()).hexdigest() == row["clean_sha256"]
    assert batch.scale_level == row["scale_level"]
    new_hashes[str(row["step"])] = hashlib.sha256(after.tobytes()).hexdigest()
    previous = {x["item"]: x for x in old_records}
    current = {x["item"]: x for x in records}
    for item in range(4):
        p, c = previous.get(item), current.get(item)
        if p is not None and not p.get("coverage"):
            assert p == c
            np.testing.assert_array_equal(before[item], after[item])
            counts["old_geometry_retained"] += 1
        elif c is None:
            assert p is None
            np.testing.assert_array_equal(before[item], after[item])
            counts["native_retained"] += 1
        elif p is not None and p.get("coverage") and c.get("coverage"):
            assert p == c
            np.testing.assert_array_equal(before[item], after[item])
            counts["coverage_retained"] += 1
        if not np.array_equal(before[item], after[item]):
            counts["changed_items"] += 1
assert counts["changed_items"] > 0 and counts["old_geometry_retained"] > 0
full, resumed = take(0, 12), take(6, 6)
for step, batch in enumerate(resumed, start=6):
    np.testing.assert_array_equal(batch.images, full[step].images)
    np.testing.assert_array_equal(mixed(batch, step, config)[0], mixed(full[step], step, config)[0])
    assert batch.scale_level == full[step].scale_level
preparers = [prepare_batch(c) for c in [old, config]]
depth_records = []
for depth in range(1, 7):
    step = depth - 1
    clean = [jnp.asarray(mixed(full[step], step, c)[0]) for c in [old, config]]
    key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), step)
    before, after = [f(c, key, depth) for c, f in zip(clean, preparers, strict=True)]
    jax.tree.map(np.testing.assert_array_equal, before.noise_levels, after.noise_levels)
    active = np.asarray(after.noise_levels[0]) > 0
    np.testing.assert_array_equal(
        np.asarray(after.targets[0])[active], np.asarray(clean[1])[active]
    )
    np.testing.assert_array_equal(
        np.asarray(after.targets[0])[~active], np.asarray(after.guides[0])[~active]
    )
    target = restoration_target(clean[1], after.guides[0], after.noise_levels[0])
    for x, y in [
        (after.targets, pyramid(target, depth)),
        (after.guides, pyramid(after.guides[0], depth)),
    ]:
        jax.tree.map(lambda u, v: np.testing.assert_allclose(u, v, atol=1e-6, rtol=0), x, y)
    error = 0.0
    for k in range(depth):
        t = np.asarray(after.noise_levels[k])
        x = np.asarray(before.noisy[k]) - np.asarray(before.guides[k]) * np.sqrt(1 - t)
        y = np.asarray(after.noisy[k]) - np.asarray(after.guides[k]) * np.sqrt(1 - t)
        np.testing.assert_allclose(x, y, atol=1e-6, rtol=0)
        error = max(error, float(np.max(np.abs(x - y))))
    depth_records.append(dict(depth=depth, gaussian_residual_max_error=error))
first = [
    hashlib.sha256(mixed(batch, step, config)[0].tobytes()).hexdigest()
    for step, batch in enumerate(full)
]
actual_old = [
    hashlib.sha256(mixed(batch, step, old)[0].tobytes()).hexdigest()
    for step, batch in enumerate(full)
]
assert actual_old == read(parent / "data-audit.json")["first_12_target_hashes"]
teachers, depths, sampling = Counter(), Counter(), []
for step in range(4000):
    selected = np.random.default_rng([config.train.seed, step, 0x47454F4D]).random(4) < 0.5
    covered = np.random.default_rng([config.train.seed, step, 0x434F5653]).random(4) < 0.2
    count = Counter()
    for item in range(4):
        if selected[item]:
            count["original_geometry"] += 1
        elif covered[item]:
            rng = np.random.default_rng([config.train.seed, step, item, 0x434F5645])
            count["coverage_" + COVERAGE_KINDS[int(rng.integers(5))]] += 1
    count["native_corpus"] = 4 - sum(count.values())
    level = int(np.random.default_rng([config.train.seed, step, 1]).integers(6))
    depth = config.diffusion.levels_for_scale(512, config.model.minimum_size, level)
    depths[str(depth)] += 1
    teachers.update(count)
    sampling.append(dict(step=step + 1, depth=depth, teacher_counts=dict(count)))
assert dict(depths) == read(parent / "data-audit.json")["depth_counts"]
assert sum(teachers.values()) == 16000
write(root / "expected-sampling.json", sampling)
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        replayed_old_clean_hashes=166,
        old_clean_hashes_all_exact=True,
        old_geometry_examples_preserved=True,
        prepared_masks_all_depths_exact=True,
        targets_and_guides_all_depths_correct=True,
        downsample_before_noise=True,
        config_only_differences=["geometry.probability", "geometry.coverage_probability"],
        audited_new_clean_hashes=new_hashes,
        audited_item_counts=counts,
        depth_records=depth_records,
        full_run_expected_counts=teachers,
        depth_counts=depths,
        first_12_target_hashes=first,
        expected_sampling_sha256=sha(root / "expected-sampling.json"),
        parent_data_audit_sha256=sha(parent / "data-audit.json"),
        script_sha256=sha(Path(__file__)),
    ),
)
print(
    dict(
        audited_item_counts=counts, full_run_expected_counts=teachers, depth_records=depth_records
    ),
    flush=True,
)
