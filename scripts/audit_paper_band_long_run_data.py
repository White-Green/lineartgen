"""Audit continuation config and the unchanged indexed data clock."""

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

root = Path("runs/paper-band-long-run-20261010").resolve()
plan = read(root / "plan.json")
configs = {}
originals = {}
for name, entry in plan["parents"].items():
    old = load_config(Path(entry["root"]) / f"configs/{entry['method']}.json")
    new = load_config(root / f"configs/{name}.json")
    a, b = new.model_dump(mode="json"), old.model_dump(mode="json")
    assert a["train"]["steps"] == 16000 and a["train"]["learning_rate_decay_steps"] == 4000
    a["train"]["steps"] = 4000
    a["train"]["learning_rate_decay_steps"] = None
    assert a == b
    configs[name] = new
    originals[name] = old
config = configs["band"]
training, _, _ = split_experiment(discover(config.data), config.data, config.train.seed)


def take(start, count):
    stream = batches(
        training, config.data, batch_size=4, count=count, seed=config.train.seed, start=start
    )
    try:
        return list(stream)
    finally:
        stream.close()


def mixed(batch, step):
    return mix_geometry(
        batch.images,
        config.geometry,
        seed=config.train.seed,
        step=step,
        image_scale=config.data.image_scale,
    )[0]


journals = {
    name: [
        json.loads(x)
        for x in (Path(entry["root"]) / "training/training.jsonl").read_text().splitlines()
    ]
    for name, entry in plan["parents"].items()
}
assert {len(x) for x in journals.values()} == {166}
for row in journals["band"]:
    step = row["step"] - 1
    batch = take(step, 1)[0]
    assert hashlib.sha256(mixed(batch, step).tobytes()).hexdigest() == row["clean_sha256"]
    assert batch.scale_level == row["scale_level"]
crossing, first, resumed = take(3994, 18), take(4000, 12), take(4006, 6)
for i, batch in enumerate(first):
    step = 4000 + i
    np.testing.assert_array_equal(batch.images, crossing[i + 6].images)
    np.testing.assert_array_equal(mixed(batch, step), mixed(crossing[i + 6], step))
    assert batch.scale_level == crossing[i + 6].scale_level
for i, batch in enumerate(resumed):
    step = 4006 + i
    np.testing.assert_array_equal(batch.images, first[i + 6].images)
    np.testing.assert_array_equal(mixed(batch, step), mixed(first[i + 6], step))
    assert batch.scale_level == first[i + 6].scale_level
for name, new in configs.items():
    before, after = prepare_batch(originals[name]), prepare_batch(new)
    for depth in range(1, 7):
        step = 4000 + depth - 1
        clean = jnp.asarray(mixed(first[depth - 1], step))
        key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), step)
        jax.tree.map(
            np.testing.assert_array_equal, before(clean, key, depth), after(clean, key, depth)
        )
counts, depths = Counter(), Counter()
for step in range(4000, 16000):
    level = int(np.random.default_rng([config.train.seed, step, 1]).integers(6))
    depths[config.diffusion.levels_for_scale(512, config.model.minimum_size, level)] += 1
    selected = (
        np.random.default_rng([config.train.seed, step, 0x47454F4D]).random(4)
        < config.geometry.probability
    )
    covered = (
        np.random.default_rng([config.train.seed, step, 0x434F5653]).random(4)
        < config.geometry.coverage_probability
    )
    for item in range(4):
        if selected[item]:
            counts["original_geometry"] += 1
        elif covered[item]:
            rng = np.random.default_rng([config.train.seed, step, item, 0x434F5645])
            counts["coverage_" + COVERAGE_KINDS[int(rng.integers(5))]] += 1
        else:
            counts["native_corpus"] += 1
assert sum(counts.values()) == 48000 and sum(depths.values()) == 12000
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        boundary_4000_exact=True,
        boundary_4006_exact=True,
        replayed_parent_clean_hashes=166,
        old_clean_hashes_all_exact=True,
        prepared_arrays_all_depths_exact=True,
        config_only_differences=["train.steps", "train.learning_rate_decay_steps"],
        extension_expected_teacher_counts=counts,
        extension_expected_depth_counts=depths,
        first_12_extension_target_hashes=[
            hashlib.sha256(mixed(batch, 4000 + i).tobytes()).hexdigest()
            for i, batch in enumerate(first)
        ],
        script_sha256=sha(Path(__file__)),
    ),
)
print(read(root / "data-audit.json"), flush=True)
