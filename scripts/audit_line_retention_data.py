"""Audit clean-source replay, intended data changes, shared masks and recurrence choices."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from train_line_retention_data import METHODS, prepare_arms_batch
from train_procedural_structure import prepare_batch

from lineartgen.config import load_config
from lineartgen.data.coverage import COVERAGE_KINDS
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment

root = Path("runs/line-retention-data-20261010").resolve()
plan = read(root / "plan.json")
parent = Path(plan["parent"])
old = load_config(parent / "configs/l1.json")
configs = {n: load_config(root / f"configs/{n}.json") for n in METHODS}
for name, config in configs.items():
    a = config.model_dump(mode="json")
    b = old.model_dump(mode="json")
    if name == "geometry50":
        assert a["geometry"]["probability"] == 0.5 and a["geometry"]["coverage_probability"] == 0.2
        a["geometry"]["probability"] = 0.2
        a["geometry"]["coverage_probability"] = 0.125
    else:
        assert a["sketch"]["mode"] == "blur"
        a["sketch"]["mode"] = "mixed"
    assert a == b
training, _, _ = split_experiment(discover(old.data), old.data, old.train.seed)


def take(start, count):
    stream = batches(
        training, old.data, batch_size=4, count=count, seed=old.train.seed, start=start
    )
    try:
        return list(stream)
    finally:
        stream.close()


def mix(config, batch, step):
    return mix_geometry(
        batch.images,
        config.geometry,
        seed=config.train.seed,
        step=step,
        image_scale=config.data.image_scale,
    )


expected = [json.loads(x) for x in (parent / "training/training.jsonl").read_text().splitlines()]
assert expected[-1]["step"] == 4000
for row in expected:
    step = row["step"] - 1
    batch = take(step, 1)[0]
    clean, _ = mix(old, batch, step)
    assert hashlib.sha256(clean.tobytes()).hexdigest() == row["clean_sha256"]
    np.testing.assert_array_equal(mix(configs["blur80"], batch, step)[0], clean)
    assert batch.scale_level == row["scale_level"]
full = take(0, 64)
resumed = take(6, 6)
hashes = {n: [] for n in METHODS}
changed_items = 0
for step, batch in enumerate(full):
    baseline, old_records = mix(old, batch, step)
    for name, config in configs.items():
        clean, records = mix(config, batch, step)
        if name == "geometry50":
            by_item = {x["item"]: x for x in records}
            for record in old_records:
                if not record.get("coverage"):
                    assert by_item[record["item"]] == record
                    np.testing.assert_array_equal(clean[record["item"]], baseline[record["item"]])
            changed_items += sum(
                not np.array_equal(a, b) for a, b in zip(clean, baseline, strict=True)
            )
        else:
            np.testing.assert_array_equal(clean, baseline)
        if step < 12:
            hashes[name].append(hashlib.sha256(clean.tobytes()).hexdigest())
assert changed_items > 0
for step, batch in enumerate(resumed, start=6):
    np.testing.assert_array_equal(batch.images, full[step].images)
    for name, config in configs.items():
        assert (
            hashlib.sha256(mix(config, batch, step)[0].tobytes()).hexdigest() == hashes[name][step]
        )
preparers = {name: prepare_batch(config) for name, config in configs.items()}
previous = prepare_batch(old)
key = jax.random.PRNGKey(old.train.seed)
source_difference = False
for depth in range(1, 7):
    step = depth - 1
    cleans, records, prepared = prepare_arms_batch(
        configs, preparers, full[step].images, key, step, depth
    )
    control = previous(
        jnp.asarray(mix(old, full[step], step)[0]), jax.random.fold_in(key, step), depth
    )
    for name, p in prepared.items():
        assert hashlib.sha256(cleans[name].tobytes()).hexdigest() == hashes[name][step]
        jax.tree.map(np.testing.assert_array_equal, p.noise_levels, control.noise_levels)
        active = np.asarray(p.noise_levels[0]) > 0
        np.testing.assert_array_equal(np.asarray(p.targets[0])[active], cleans[name][active])
        np.testing.assert_array_equal(
            np.asarray(p.targets[0])[~active], np.asarray(p.guides[0])[~active]
        )
    source_difference |= not np.array_equal(
        np.asarray(control.guides[0]), np.asarray(prepared["blur80"].guides[0])
    )
assert source_difference
counts = {}
depth_counts = Counter()
for name, config in configs.items():
    count = Counter()
    for step in range(4000):
        level = int(np.random.default_rng([old.train.seed, step, 1]).integers(6))
        if name == METHODS[0]:
            depth_counts[old.diffusion.levels_for_scale(512, old.model.minimum_size, level)] += 1
        selected = (
            np.random.default_rng([old.train.seed, step, 0x47454F4D]).random(4)
            < config.geometry.probability
        )
        covered = (
            np.random.default_rng([old.train.seed, step, 0x434F5653]).random(4)
            < config.geometry.coverage_probability
        )
        for item in range(4):
            if selected[item]:
                count["original_geometry"] += 1
            elif covered[item]:
                rng = np.random.default_rng([old.train.seed, step, item, 0x434F5645])
                count["coverage_" + COVERAGE_KINDS[int(rng.integers(5))]] += 1
            else:
                count["native_corpus"] += 1
    assert sum(count.values()) == 16000
    counts[name] = count
old_audit = read(parent / "data-audit.json")
assert counts["blur80"] == old_audit["full_run_expected_counts"]
assert dict(depth_counts) == {int(k): v for k, v in old_audit["depth_counts"].items()}
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        replayed_old_clean_hashes=len(expected),
        old_clean_hashes_all_exact=True,
        old_geometry_examples_preserved=True,
        blur_clean_targets_exact=True,
        prepared_masks_all_depths_exact=True,
        blur_source_changes_observed=True,
        first_12_target_hashes=hashes,
        full_run_expected_counts=counts,
        depth_counts=depth_counts,
        changed_items_in_64_batches=changed_items,
        parent_data_audit_sha256=sha(parent / "data-audit.json"),
        script_sha256=sha(Path(__file__)),
    ),
)
print(
    {k: v for k, v in read(root / "data-audit.json").items() if k != "first_12_target_hashes"},
    flush=True,
)
