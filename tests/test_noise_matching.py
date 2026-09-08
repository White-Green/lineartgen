import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lineartgen.noise_matching import assignment_indices, match_noise, pairwise_noise_costs


def burn_reference_cost(clean, level, candidates, resolution):
    noisy = clean[:, None] * np.sqrt(1 - level[:, None]) + candidates[None] * np.sqrt(
        level[:, None]
    )
    error = (clean[:, None] - noisy) ** 2
    height, width = clean.shape[1:3]
    out_h, out_w = min(height, resolution), min(width, resolution)
    costs = np.zeros(error.shape[:2], np.float64)
    for y in range(out_h):
        for x in range(out_w):
            ys = slice(y * height // out_h, ((y + 1) * height + out_h - 1) // out_h)
            xs = slice(x * width // out_w, ((x + 1) * width + out_w - 1) // out_w)
            costs += error[:, :, ys, xs].mean((2, 3)).sum(-1)
    return costs


@pytest.mark.parametrize("shape,resolution", [((16, 16), 32), ((64, 32), 32), ((7, 10), 4)])
def test_factored_cost_matches_burn_masked_corruption_cost(shape, resolution):
    rng = np.random.default_rng(1)
    clean = rng.uniform(-1, 1, (3, *shape, 1)).astype(np.float32)
    level = rng.uniform(0, 1, clean.shape).astype(np.float32)
    level[:, :3] = 0
    candidates = rng.normal(size=(5, *shape, 1)).astype(np.float32)
    expected = burn_reference_cost(clean, level, candidates, resolution)
    actual = jax.jit(lambda x, t, z: pairwise_noise_costs(x, t, z, resolution=resolution))(
        clean, level, candidates
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
    selected = np.asarray(jax.jit(assignment_indices)(actual))
    optimum = min(
        sum(expected[row, col] for row, col in enumerate(cols))
        for cols in itertools.permutations(range(5), 3)
    )
    np.testing.assert_allclose(expected[np.arange(3), selected].sum(), optimum, rtol=1e-6)


@pytest.mark.parametrize(
    "costs",
    [
        [[4, 1, 3], [2, 0, 5], [3, 2, 2]],
        [[9, 1, 8, 2], [8, 0, 5, 3]],
        [[0, 0, 0], [0, 0, 0]],
        [[3, 1, 2]],
    ],
)
@pytest.mark.parametrize("solver", ["scipy", "optax"])
def test_assignment_is_globally_optimal_without_reusing_candidates(costs, solver):
    costs = np.asarray(costs, np.float32)
    selected = np.asarray(jax.jit(lambda c: assignment_indices(c, solver=solver))(costs))
    assert len(set(selected)) == costs.shape[0]
    optimum = min(
        sum(costs[row, col] for row, col in enumerate(cols))
        for cols in itertools.permutations(range(costs.shape[1]), costs.shape[0])
    )
    assert costs[np.arange(costs.shape[0]), selected].sum() == optimum


def test_zero_noise_pixels_do_not_affect_matching_and_noise_is_not_averaged():
    clean = jnp.zeros((2, 4, 4, 1))
    level = jnp.zeros_like(clean).at[:, 2:].set(0.5)
    candidates = jax.random.normal(jax.random.PRNGKey(3), (4, 4, 4, 1))
    changed = candidates.at[:, :2].set(999)
    np.testing.assert_array_equal(
        pairwise_noise_costs(clean, level, candidates), pairwise_noise_costs(clean, level, changed)
    )
    selected = np.asarray(jax.jit(match_noise)(clean, level, candidates))
    assert all(
        any(np.array_equal(row, candidate) for candidate in np.asarray(candidates))
        for row in selected
    )
    assert not np.array_equal(selected[0], selected[1])


def test_matching_does_not_differentiate_through_costs():
    clean = jnp.ones((2, 4, 4, 1))
    candidates = jnp.zeros((3, 4, 4, 1))
    gradient = jax.grad(lambda x: pairwise_noise_costs(x, x * 0.5, candidates).sum())(clean)
    np.testing.assert_array_equal(gradient, np.zeros_like(clean))
