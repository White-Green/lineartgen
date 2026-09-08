"""Compare equivalent cost formulas and exact solvers, excluding JIT warmup.

The reference is a JAX transcription of Burn's chunked broadcast calculation,
not a measurement of the Rust backend. Timings include block_until_ready.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from lineartgen.noise_matching import assignment_indices, pairwise_noise_costs


def broadcast_costs(clean, level, candidates):
    batch, height, width, channels = clean.shape
    output_h, output_w = min(height, 32), min(width, 32)
    chunks = []
    for start in range(0, candidates.shape[0], 32):
        noise = candidates[start : start + 32]
        noisy = clean[:, None] * jnp.sqrt(1 - level[:, None]) + noise[None] * jnp.sqrt(
            level[:, None]
        )
        error = (clean[:, None] - noisy) ** 2
        pooled = error.reshape(
            batch,
            noise.shape[0],
            output_h,
            height // output_h,
            output_w,
            width // output_w,
            channels,
        ).mean((3, 5))
        chunks.append(pooled.sum((2, 3, 4)))
    return jnp.concatenate(chunks, axis=1)


def measure(function, args, repeats):
    jax.block_until_ready(function(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(*args))
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--pool", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 16 or args.size & (args.size - 1) or args.pool < args.batch or args.repeats < 1:
        parser.error("Use a power-of-two size >=16, pool >= batch and repeats >=1")
    key, noise_key = jax.random.split(jax.random.PRNGKey(42))
    clean = jax.random.uniform(key, (args.batch, args.size, args.size, 1), minval=-1, maxval=1)
    level = jnp.full_like(clean, 0.5).at[:, : args.size // 4].set(0)
    candidates = jax.random.normal(noise_key, (args.pool, args.size, args.size, 1))
    inputs = (clean, level, candidates)
    reference, factored = jax.jit(broadcast_costs), jax.jit(pairwise_noise_costs)
    expected, costs = np.asarray(reference(*inputs)), factored(*inputs)
    actual = np.asarray(costs)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
    optax_solver = jax.jit(lambda costs: assignment_indices(costs, solver="optax"))
    selected = np.asarray(optax_solver(costs))
    scipy_indices = linear_sum_assignment(actual)[1]
    np.testing.assert_allclose(
        actual[np.arange(args.batch), selected].sum(),
        actual[np.arange(args.batch), scipy_indices].sum(),
        rtol=1e-6,
    )

    def scipy_with_transfer(costs):
        return jax.device_put(linear_sum_assignment(np.asarray(costs))[1].astype(np.int32))

    result = {
        "devices": str(jax.devices()),
        "size": args.size,
        "batch": args.batch,
        "pool": args.pool,
        "max_abs_cost_error": float(np.abs(actual - expected).max()),
        "broadcast_cost_ms": measure(reference, inputs, args.repeats),
        "factored_cost_ms": measure(factored, inputs, args.repeats),
        "optax_assignment_ms": measure(optax_solver, (costs,), args.repeats),
        "scipy_callback_ms": measure(jax.jit(assignment_indices), (costs,), args.repeats),
        "factored_plus_scipy_callback_ms": measure(
            jax.jit(lambda x, t, z: assignment_indices(pairwise_noise_costs(x, t, z))),
            inputs,
            args.repeats,
        ),
        "scipy_assignment_with_transfer_ms": measure(scipy_with_transfer, (costs,), args.repeats),
        "matched_cost": float(actual[np.arange(args.batch), selected].mean()),
        "first_candidates_cost": float(actual[np.arange(args.batch), np.arange(args.batch)].mean()),
        "scope": (
            "Synthetic input; equivalent objectives; JIT warmup excluded; not a quality benchmark"
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
