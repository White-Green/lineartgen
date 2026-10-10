"""Verify capacity is the only change and all training inputs match R."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from train_procedural_structure import prepare_batch

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment

root = Path("runs/local-tone-balance-20261010").resolve()
plan = read(root / "plan.json")
parent = Path(plan["parent"])
configs = []
originals = []
for name, original in [("local8", "interval")]:
    config = load_config(root / f"configs/{name}.json")
    old = load_config(parent / f"configs/{original}.json")
    a = config.model_dump(mode="json")
    b = old.model_dump(mode="json")
    assert a["loss"]["local_tone_radius"] == 8 and b["loss"]["local_tone_radius"] == 0
    a["loss"]["local_tone_radius"] = 0
    assert a == b
    configs.append(config)
    originals.append(old)
config = configs[0]
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


expected = [
    json.loads(line) for line in (parent / "training/training.jsonl").read_text().splitlines()
]
assert expected[-1]["step"] == 4000
for row in expected:
    step = row["step"] - 1
    batch = take(step, 1)[0]
    assert hashlib.sha256(mixed(batch, step).tobytes()).hexdigest() == row["clean_sha256"]
    assert batch.scale_level == row["scale_level"]
full = take(0, 12)
resumed = take(6, 6)
for step, batch in enumerate(resumed, start=6):
    np.testing.assert_array_equal(batch.images, full[step].images)
    np.testing.assert_array_equal(mixed(batch, step), mixed(full[step], step))
    assert batch.scale_level == full[step].scale_level
preparers = [prepare_batch(c) for c in originals + configs]
for depth in range(1, 7):
    clean = jnp.asarray(mixed(full[depth - 1], depth - 1))
    key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), depth - 1)
    prepared = [f(clean, key, depth) for f in preparers]
    for value in prepared[1:]:
        jax.tree.map(np.testing.assert_array_equal, prepared[0], value)
previous = read(parent / "data-audit.json")
hashes = [
    hashlib.sha256(mixed(batch, step).tobytes()).hexdigest() for step, batch in enumerate(full)
]
assert hashes == previous["first_12_target_hashes"]
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        replayed_old_clean_hashes=len(expected),
        old_clean_hashes_all_exact=True,
        prepared_arrays_all_depths_exact=True,
        config_only_differences=["loss.local_tone_radius"],
        full_run_expected_counts=previous["full_run_expected_counts"],
        depth_counts=previous["depth_counts"],
        first_12_target_hashes=hashes,
        parent_data_audit_sha256=sha(parent / "data-audit.json"),
        script_sha256=sha(Path(__file__)),
    ),
)
print(read(root / "data-audit.json"), flush=True)
