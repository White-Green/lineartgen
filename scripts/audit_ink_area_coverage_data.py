"""Audit replacement coverage without touching validation or final-test examples."""

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

root = Path("runs/ink-area-coverage-20261010").resolve()
plan = read(root / "plan.json")
config = load_config(root / "configs/coverage.json")
flat = load_config(root / "configs/flat_coverage.json")
parent = Path(plan["control_parents"]["pixel"]["root"])
old = load_config(parent / "configs/native.json")
flat_parent = Path(plan["control_parents"]["flat"]["root"])
old_flat = load_config(flat_parent / "configs/flat.json")
for new, baseline in ((config, old), (flat, old_flat)):
    a, b = new.model_dump(mode="json"), baseline.model_dump(mode="json")
    a["geometry"]["coverage_probability"] = 0
    assert a == b
sources = discover(config.data)
training, _, _ = split_experiment(sources, config.data, config.train.seed)
expected = [json.loads(x) for x in (parent / "training/training.jsonl").read_text().splitlines()]


def take(start, count):
    stream = batches(
        training, config.data, batch_size=4, count=count, seed=config.train.seed, start=start
    )
    try:
        return list(stream)
    finally:
        stream.close()


def mix(batch, step, settings):
    return mix_geometry(
        batch.images,
        settings.geometry,
        seed=settings.train.seed,
        step=step,
        image_scale=settings.data.image_scale,
    )


for row in expected:
    step = row["step"] - 1
    batch = take(step, 1)[0]
    clean, _ = mix(batch, step, old)
    assert hashlib.sha256(clean.tobytes()).hexdigest() == row["clean_sha256"]
    assert batch.scale_level == row["scale_level"]

full = take(0, 64)
resumed = take(6, 6)
new_targets, audit_counts = [], Counter()
for step, batch in enumerate(full):
    original, old_records = mix(batch, step, old)
    changed, new_records = mix(batch, step, config)
    repeated, same = mix(batch, step, flat)
    np.testing.assert_array_equal(changed, repeated)
    assert new_records == same
    assert [r for r in new_records if not r.get("coverage")] == old_records
    changed_items = [r["item"] for r in new_records if r.get("coverage")]
    unchanged = [i for i in range(4) if i not in changed_items]
    np.testing.assert_array_equal(changed[unchanged], original[unchanged])
    for row in new_records:
        audit_counts[row["kinds"][0] if row.get("coverage") else "original_geometry"] += 1
    if step < 12:
        new_targets.append(changed)
for step, batch in enumerate(resumed, start=6):
    np.testing.assert_array_equal(batch.images, full[step].images)
    np.testing.assert_array_equal(mix(batch, step, config)[0], new_targets[step])
    assert batch.scale_level == full[step].scale_level

prepare, prepare_flat = prepare_batch(config), prepare_batch(flat)
for depth in range(1, 7):
    clean = jnp.asarray(new_targets[depth - 1])
    key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), depth - 1)
    a, b = prepare(clean, key, depth), prepare_flat(clean, key, depth)
    jax.tree.map(np.testing.assert_array_equal, a, b)

# Independently derive full-run counts directly from the predeclared item keys.
counts, depth_counts = Counter(), Counter()
for step in range(4000):
    level = int(np.random.default_rng([config.train.seed, step, 1]).integers(6))
    depth_counts[config.diffusion.levels_for_scale(512, config.model.minimum_size, level)] += 1
    selected = np.random.default_rng([config.train.seed, step, 0x47454F4D]).random(4) < 0.2
    covered = np.random.default_rng([config.train.seed, step, 0x434F5653]).random(4) < 0.125
    for item in range(4):
        if selected[item]:
            counts["original_geometry"] += 1
        elif covered[item]:
            rng = np.random.default_rng([config.train.seed, step, item, 0x434F5645])
            counts["coverage_" + COVERAGE_KINDS[int(rng.integers(5))]] += 1
        else:
            counts["native_corpus"] += 1
assert sum(counts.values()) == 16000
assert dict(depth_counts) == {
    int(k): v for k, v in read(parent / "data-audit.json")["depth_counts"].items()
}
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        replayed_old_clean_hashes=len(expected),
        old_clean_hashes_all_exact=True,
        original_geometry_unchanged=True,
        untouched_corpus_items_exact=True,
        prepared_arrays_all_depths_exact=True,
        full_run_expected_counts=counts,
        audited_first_64_batches_counts=audit_counts,
        depth_counts=depth_counts,
        first_12_target_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in new_targets],
        script_sha256=sha(Path(__file__)),
    ),
)
print(read(root / "data-audit.json"), flush=True)
