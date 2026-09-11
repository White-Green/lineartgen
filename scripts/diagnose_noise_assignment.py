"""Measure the selected noise distribution and corruption cost before any learning."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lineartgen.diffusion import downsample
from lineartgen.noise_matching import assignment_indices, pairwise_noise_costs


@jax.jit
def measure(target, level, key):
    candidates = jax.random.normal(key, (256, *target.shape[1:]))
    costs = pairwise_noise_costs(target, level, candidates)
    indices = (
        assignment_indices(costs),
        assignment_indices(costs[:, :16]),
        jnp.arange(16),
    )
    active = level > 0
    count = jnp.maximum(active.sum(), 1)
    image_mean = (target * active).sum() / count
    image_variance = (((target - image_mean) ** 2) * active).sum() / count
    rows = []
    for selected in indices:
        noise = candidates[selected]
        mean = (noise * active).sum() / count
        variance = (((noise - mean) ** 2) * active).sum() / count
        covariance = ((target - image_mean) * (noise - mean) * active).sum() / count
        corrupted = target * jnp.sqrt(1 - level) + noise * jnp.sqrt(level)
        mse = (((corrupted - target) ** 2) * active).sum() / count
        rows.append(
            jnp.stack(
                (
                    noise.mean(),
                    noise.var(),
                    mean,
                    variance,
                    covariance / jnp.sqrt(jnp.maximum(variance * image_variance, 1e-12)),
                    mse,
                    costs[jnp.arange(16), selected].mean(),
                )
            )
        )
    return jnp.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=Path, default=Path("runs/training-speed-20260909/batch16-512.f32")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 512])
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    batch = np.fromfile(args.batch, np.float32).reshape(16, 512, 512, 1)
    fields = [
        "noise_mean",
        "noise_variance",
        "active_noise_mean",
        "active_noise_variance",
        "active_image_noise_correlation",
        "active_corruption_mse",
        "assignment_cost",
    ]
    rows = []
    for size in args.sizes:
        if size not in (16, 32, 64, 128, 256, 512):
            parser.error("sizes must be powers of two from 16 through 512")
        target = jax.device_put(batch)
        while target.shape[1] > size:
            target = downsample(target)
        for strength in (0.1, 0.3, 0.5, 0.7):
            for partial in (False, True):
                level = jnp.full_like(target, strength)
                if partial:
                    level = level.at[:, : size // 5].set(0).at[:, :, : size // 3].set(0)
                for seed in range(args.seeds):
                    values = np.asarray(measure(target, level, jax.random.PRNGKey(20260911 + seed)))
                    for method, metrics in zip(
                        ("current", "ot_full", "independent_noise"), values, strict=True
                    ):
                        rows.append(
                            dict(
                                size=size,
                                strength=strength,
                                partial=partial,
                                seed=seed,
                                method=method,
                                **dict(zip(fields, metrics.astype(float), strict=True)),
                            )
                        )
        print(f"Measured noise assignment at {size}px", flush=True)
    summary = []
    for size, strength, partial, method in sorted(
        {(r["size"], r["strength"], r["partial"], r["method"]) for r in rows}
    ):
        selected = [
            r
            for r in rows
            if (r["size"], r["strength"], r["partial"], r["method"])
            == (size, strength, partial, method)
        ]
        summary.append(
            dict(
                size=size,
                strength=strength,
                partial=partial,
                method=method,
                **{field: float(np.mean([r[field] for r in selected])) for field in fields},
            )
        )
    report = dict(
        batch=str(args.batch),
        batch_sha256=hashlib.sha256(args.batch.read_bytes()).hexdigest(),
        devices=[str(x) for x in jax.devices()],
        generation_pool=256,
        batch_size=16,
        seeds=args.seeds,
        results=rows,
        summary=summary,
        limitations=(
            "Fixed real batch and repeated seeds; marginal equality of the 16-candidate set "
            "does not imply independence conditional on images or masks. "
            "Residual uses the same preparation as current."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
