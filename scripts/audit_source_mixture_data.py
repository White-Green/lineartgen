"""Validate the actual indexed crop stream and paired rough-input contract before training."""

import hashlib
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import write
from train_procedural_structure import prepare_batch
from train_source_mixture import SEED

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment

root = Path("runs/source-mixture-training-20261009").resolve()
configs = {n: load_config(root / f"configs/{n}.json") for n in ("pixel", "mixed")}
a, b = (c.model_dump() for c in configs.values())
a["sketch"]["mode"] = b["sketch"]["mode"]
assert a == b
config = configs["pixel"]
training, _, _ = split_experiment(discover(config.data), config.data, SEED)


def fetch(start, count):
    stream = batches(training, config.data, batch_size=4, count=count, seed=SEED, start=start)
    result = []
    try:
        for step, batch in enumerate(stream, start=start):
            clean, _ = mix_geometry(
                batch.images,
                config.geometry,
                seed=SEED,
                step=step,
                image_scale=config.data.image_scale,
            )
            result.append((batch.scale_level, clean))
    finally:
        stream.close()
    return result


original, resumed = fetch(0, 12), fetch(6, 6)
for a, b in zip(original[6:], resumed, strict=True):
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
preparers = {n: prepare_batch(c) for n, c in configs.items()}
rows = []
for depth in range(1, 7):
    clean = original[depth - 1][1]
    key = jax.random.fold_in(jax.random.PRNGKey(SEED), depth - 1)
    pairs = {n: f(jax.numpy.asarray(clean), key, depth) for n, f in preparers.items()}
    left, right = pairs.values()
    jax.tree.map(np.testing.assert_array_equal, left.noise_levels, right.noise_levels)
    active = np.asarray(left.noise_levels[0]) > 0
    for p in pairs.values():
        np.testing.assert_array_equal(np.asarray(p.targets[0])[active], clean[active])
        np.testing.assert_array_equal(
            np.asarray(p.targets[0])[~active], np.asarray(p.guides[0])[~active]
        )
    # Recover the Gaussian contribution before multiplication by sqrt(t).
    residuals = [
        [
            np.asarray(x - g * jax.numpy.sqrt(1 - t))
            for x, g, t in zip(p.noisy, p.guides, p.noise_levels, strict=True)
        ]
        for p in pairs.values()
    ]
    max_delta = max(float(np.abs(a - b).max()) for a, b in zip(*residuals, strict=True))
    assert max_delta <= 5e-7
    rows.append(
        dict(
            depth=depth,
            gaussian_contribution_max_difference=max_delta,
            source_changed=bool(np.any(np.asarray(left.guides[0]) != np.asarray(right.guides[0]))),
            clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
        )
    )
    print(rows[-1], flush=True)
assert any(r["source_changed"] for r in rows)
write(
    root / "data-audit.json",
    dict(
        config_only_difference="sketch.mode",
        crop_resume_exact=True,
        resumed_steps=[6, 7, 8, 9, 10, 11],
        all_depths_checked=True,
        masks_equal=True,
        clean_active_targets_exact=True,
        protected_source_targets_exact=True,
        rows=rows,
    ),
)
