"""Bind unchanged teachers/masks/noise draws and the intended source-only change."""

import hashlib
import importlib.util
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
from lineartgen.diffusion import pyramid
from lineartgen.sketch import augment_sketch, restoration_target

root = Path("runs/strong-corruption-mixture-20261010").resolve()
plan = read(root / "plan.json")
parent = Path(plan["parent"])
config = load_config(root / "configs/strong50.json")
old = load_config(parent / "configs/interval.json")
a, b = config.model_dump(mode="json"), old.model_dump(mode="json")
assert a["sketch"]["strong_probability"] == 0.5 and b["sketch"]["strong_probability"] == 0
a["sketch"]["strong_probability"] = 0
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
full, resumed = take(0, 12), take(6, 6)
for step, batch in enumerate(resumed, start=6):
    np.testing.assert_array_equal(batch.images, full[step].images)
    np.testing.assert_array_equal(mixed(batch, step), mixed(full[step], step))
    assert batch.scale_level == full[step].scale_level

# Load the actual frozen parent implementation for default-path compatibility.
legacy_path = parent.parents[1] / "src/lineartgen/sketch.py"
spec = importlib.util.spec_from_file_location("frozen_parent_sketch", legacy_path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
legacy_apply = jax.jit(lambda c, k: legacy.augment_sketch(c, k, old.sketch))
current_apply = jax.jit(lambda c, k: augment_sketch(c, k, old.sketch))
preparers = [prepare_batch(c) for c in [old, config]]
depth_records = []
for depth in range(1, 7):
    clean = jnp.asarray(mixed(full[depth - 1], depth - 1))
    key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), depth - 1)
    sketch_key = jax.random.fold_in(key, 0x534B4554)
    np.testing.assert_array_equal(legacy_apply(clean, sketch_key), current_apply(clean, sketch_key))
    before, after = [f(clean, key, depth) for f in preparers]
    jax.tree.map(np.testing.assert_array_equal, before.noise_levels, after.noise_levels)
    active = np.asarray(after.noise_levels[0]) > 0
    np.testing.assert_array_equal(np.asarray(after.targets[0])[active], np.asarray(clean)[active])
    np.testing.assert_array_equal(
        np.asarray(after.targets[0])[~active], np.asarray(after.guides[0])[~active]
    )
    expected_target = restoration_target(clean, after.guides[0], after.noise_levels[0])
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, atol=1e-6, rtol=0),
        after.targets,
        pyramid(expected_target, depth),
    )
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, atol=1e-6, rtol=0),
        after.guides,
        pyramid(after.guides[0], depth),
    )
    noise_error = 0.0
    for index in range(depth):
        t = np.asarray(after.noise_levels[index])
        old_noise = np.asarray(before.noisy[index]) - np.asarray(before.guides[index]) * np.sqrt(
            1 - t
        )
        new_noise = np.asarray(after.noisy[index]) - np.asarray(after.guides[index]) * np.sqrt(
            1 - t
        )
        np.testing.assert_allclose(old_noise, new_noise, atol=1e-6, rtol=0)
        noise_error = max(noise_error, float(np.max(np.abs(old_noise - new_noise))))
    enabled = np.asarray(jax.random.bernoulli(jax.random.fold_in(sketch_key, 0), 0.8, (4,)))
    selected = np.asarray(jax.random.bernoulli(jax.random.fold_in(sketch_key, 5), 0.5, (4,)))
    eligible = enabled & selected
    source_error = np.max(
        np.abs(np.asarray(before.guides[0]) - np.asarray(after.guides[0])), axis=(1, 2, 3)
    )
    assert np.all(source_error[~eligible] <= 1e-6)
    changed = source_error > 1e-6
    depth_records.append(
        dict(
            depth=depth,
            eligible_items=int(eligible.sum()),
            changed_items=int(changed.sum()),
            gaussian_residual_max_error=noise_error,
        )
    )
assert sum(x["changed_items"] for x in depth_records) > 0
previous = read(parent / "data-audit.json")
hashes = [
    hashlib.sha256(mixed(batch, step).tobytes()).hexdigest() for step, batch in enumerate(full)
]
assert hashes == previous["first_12_target_hashes"]


def exposure(sketch):
    def draw(step):
        key = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), step)
        key = jax.random.fold_in(key, 0x534B4554)
        shape = (4, 1, 1, 1)
        choose = jax.random.bernoulli(jax.random.fold_in(key, 5), sketch.strong_probability, shape)
        enabled = jax.random.bernoulli(jax.random.fold_in(key, 0), sketch.probability, shape)
        family = jax.random.randint(jax.random.fold_in(key, 4), shape, 0, 3)
        opacity = jax.random.uniform(
            jax.random.fold_in(key, 1), shape, minval=sketch.opacity_min, maxval=sketch.opacity_max
        )
        strong_opacity = jax.random.uniform(
            jax.random.fold_in(key, 1),
            shape,
            minval=sketch.strong_opacity_min,
            maxval=sketch.strong_opacity_max,
        )
        sigma = jax.random.uniform(
            jax.random.fold_in(key, 2),
            (4,),
            minval=sketch.blur_sigma_min,
            maxval=sketch.blur_sigma_max,
        )
        strong_sigma = jax.random.uniform(
            jax.random.fold_in(key, 2),
            (4,),
            minval=sketch.strong_blur_sigma_min,
            maxval=sketch.strong_blur_sigma_max,
        )
        return (
            enabled.reshape(4),
            family.reshape(4),
            choose.reshape(4),
            jnp.where(choose, strong_opacity, opacity).reshape(4),
            jnp.where(choose.reshape(4), strong_sigma, sigma),
        )

    enabled, family, selected, opacity, sigma = map(
        np.asarray, jax.jit(jax.vmap(draw))(jnp.arange(4000))
    )
    geometry = np.array(
        [
            np.random.default_rng([config.train.seed, step, 0x47454F4D]).random(4)
            < config.geometry.probability
            for step in range(4000)
        ]
    )
    assert int(geometry.sum()) == 3197
    result = {}
    for label, mask in [("all", np.ones((4000, 4), bool)), ("geometry", geometry)]:
        result[label] = dict(
            items=int(mask.sum()),
            enabled=int((enabled & mask).sum()),
            selected_enabled=int((selected & enabled & mask).sum()),
            families={},
        )
        for index, name in enumerate(("tone", "blur", "strokes")):
            active = mask & enabled & (family == index)
            severe = active & (opacity <= 0.45)
            if name != "tone":
                severe &= sigma >= 1.5
            result[label]["families"][name] = dict(
                items=int(active.sum()), severe_region_items=int(severe.sum())
            )
    return result


exposures = dict(control=exposure(old.sketch), candidate=exposure(config.sketch))
assert exposures["control"]["geometry"]["families"]["blur"]["severe_region_items"] == 53
assert exposures["control"]["geometry"]["families"]["strokes"]["severe_region_items"] == 53
write(
    root / "data-audit.json",
    dict(
        crop_resume_exact=True,
        replayed_old_clean_hashes=len(expected),
        old_clean_hashes_all_exact=True,
        default_augmentation_matches_frozen_parent=True,
        masks_exact=True,
        gaussian_draws_unchanged_with_roundoff=True,
        downsample_before_noise=True,
        active_targets_clean_and_protected_targets_source=True,
        source_changes_only_enabled_selected_images=True,
        config_only_differences=["sketch.strong_probability"],
        depth_records=depth_records,
        augmentation_exposure=exposures,
        full_run_expected_counts=previous["full_run_expected_counts"],
        depth_counts=previous["depth_counts"],
        first_12_target_hashes=hashes,
        parent_data_audit_sha256=sha(parent / "data-audit.json"),
        legacy_augmentation_sha256=sha(legacy_path),
        script_sha256=sha(Path(__file__)),
    ),
)
print(read(root / "data-audit.json"), flush=True)
