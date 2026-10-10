"""Replay baseline teachers and bind the intended narrower-geometry teacher stream."""

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
from lineartgen.diffusion import pyramid
from lineartgen.sketch import restoration_target

root = Path("runs/teacher-width-prior-20261010").resolve()
plan = read(root / "plan.json")
parent = Path(plan["parent"])
config = load_config(root / "configs/width6.json")
old = load_config(parent / "configs/interval.json")
a, b = config.model_dump(mode="json"), old.model_dump(mode="json")
assert a["geometry"]["maximum_width"] == 6 and b["geometry"]["maximum_width"] == 12
a["geometry"]["maximum_width"] = 12
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


def mixed(batch, step, c, paths=False):
    return mix_geometry(
        batch.images,
        c.geometry,
        seed=c.train.seed,
        step=step,
        image_scale=c.data.image_scale,
        include_paths=paths,
    )


expected = [
    json.loads(line) for line in (parent / "training/training.jsonl").read_text().splitlines()
]
assert len(expected) == 166 and expected[-1]["step"] == 4000
new_hashes = {}
counts = Counter()
widths = {"control": [], "width6": []}
for row in expected:
    step = row["step"] - 1
    batch = take(step, 1)[0]
    before, ma = mixed(batch, step, old, True)
    after, mb = mixed(batch, step, config, True)
    assert hashlib.sha256(before.tobytes()).hexdigest() == row["clean_sha256"]
    assert batch.scale_level == row["scale_level"]
    new_hashes[str(step + 1)] = hashlib.sha256(after.tobytes()).hexdigest()
    assert [x["item"] for x in ma] == [x["item"] for x in mb]
    geom = {x["item"] for x in ma if not x.get("coverage")}
    cov = {x["item"] for x in ma if x.get("coverage")}
    for item in range(4):
        label = "geometry" if item in geom else "coverage" if item in cov else "native"
        counts[label] += 1
        if item not in geom:
            np.testing.assert_array_equal(before[item], after[item])
        elif np.any(before[item] != after[item]):
            counts["changed_geometry"] += 1
    for x, y in zip(ma, mb, strict=True):
        assert x["kinds"] == y["kinds"]
        if x.get("coverage"):
            assert x == y
        else:
            assert x["paths"] == y["paths"]
            for pa, pb in zip(x.get("path_geometry", []), y.get("path_geometry", []), strict=True):
                wa, wb = np.asarray(pa["widths"]), np.asarray(pb["widths"])
                assert np.all(wb <= wa) and np.max(wb) <= 6
                for name, w in [("control", wa), ("width6", wb)]:
                    widths[name].append(float(w.mean()))
                if pa["kind"] == "thin":
                    np.testing.assert_array_equal(wa, wb)
                if pa["kind"] not in ["gap", "parallel"]:
                    np.testing.assert_array_equal(pa["points"], pb["points"])
assert counts["changed_geometry"] > 0
full, resumed = take(0, 12), take(6, 6)
for step, batch in enumerate(resumed, start=6):
    np.testing.assert_array_equal(batch.images, full[step].images)
    np.testing.assert_array_equal(mixed(batch, step, config)[0], mixed(full[step], step, config)[0])
    assert batch.scale_level == full[step].scale_level
preparers = [prepare_batch(c) for c in [old, config]]
depth_records = []
for depth in range(1, 7):
    step = depth - 1
    cleans = [jnp.asarray(mixed(full[step], step, c)[0]) for c in [old, config]]
    key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), step)
    before, after = [f(c, key, depth) for c, f in zip(cleans, preparers, strict=True)]
    jax.tree.map(np.testing.assert_array_equal, before.noise_levels, after.noise_levels)
    active = np.asarray(after.noise_levels[0]) > 0
    np.testing.assert_array_equal(
        np.asarray(after.targets[0])[active], np.asarray(cleans[1])[active]
    )
    np.testing.assert_array_equal(
        np.asarray(after.targets[0])[~active], np.asarray(after.guides[0])[~active]
    )
    target = restoration_target(cleans[1], after.guides[0], after.noise_levels[0])
    for x, y in [
        (after.targets, pyramid(target, depth)),
        (after.guides, pyramid(after.guides[0], depth)),
    ]:
        jax.tree.map(lambda u, v: np.testing.assert_allclose(u, v, atol=1e-6, rtol=0), x, y)
    noise_error = 0.0
    for k in range(depth):
        t = np.asarray(after.noise_levels[k])
        a = np.asarray(before.noisy[k]) - np.asarray(before.guides[k]) * np.sqrt(1 - t)
        b = np.asarray(after.noisy[k]) - np.asarray(after.guides[k]) * np.sqrt(1 - t)
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)
        noise_error = max(noise_error, float(np.max(np.abs(a - b))))
    depth_records.append(dict(depth=depth, gaussian_residual_max_error=noise_error))
previous = read(parent / "data-audit.json")
old_hashes = [
    hashlib.sha256(mixed(batch, step, old)[0].tobytes()).hexdigest()
    for step, batch in enumerate(full)
]
assert old_hashes == previous["first_12_target_hashes"]
hashes = [
    hashlib.sha256(mixed(batch, step, config)[0].tobytes()).hexdigest()
    for step, batch in enumerate(full)
]
summary = {
    name: dict(
        paths=len(values),
        median_of_path_mean_widths=float(np.median(values)),
        mean_of_path_mean_widths=float(np.mean(values)),
        fraction_path_mean_at_most_18=float(np.mean(np.asarray(values) <= 1.8)),
    )
    for name, values in widths.items()
}
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        replayed_old_clean_hashes=len(expected),
        old_clean_hashes_all_exact=True,
        new_expected_training_hashes=new_hashes,
        native_and_coverage_items_exact=True,
        changes_only_original_geometry=True,
        kind_and_path_counts_preserved=True,
        masks_exact=True,
        gaussian_draws_unchanged_with_roundoff=True,
        downsample_before_noise=True,
        active_targets_clean_and_protected_targets_source=True,
        config_only_differences=["geometry.maximum_width"],
        audited_item_counts=dict(counts),
        paired_path_width_summary=summary,
        depth_records=depth_records,
        full_run_expected_counts=previous["full_run_expected_counts"],
        depth_counts=previous["depth_counts"],
        first_12_target_hashes=hashes,
        parent_data_audit_sha256=sha(parent / "data-audit.json"),
        script_sha256=sha(Path(__file__)),
    ),
)
print(
    {
        k: v
        for k, v in read(root / "data-audit.json").items()
        if k not in ["new_expected_training_hashes", "first_12_target_hashes"]
    },
    flush=True,
)
