import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cascade_endpoint import sample_cascade
from cascade_endpoint_eval import run_sample
from flax import nnx
from source_band_fit import run_sample as original_sample
from test_progress_modulation import modulated


class CountingHead(nnx.Module):
    def __init__(self):
        self.counts = nnx.Variable(jnp.zeros(3, jnp.int32))
        self.wrong_order = nnx.Variable(jnp.zeros((), jnp.int32))
        self.insert_sums = nnx.Variable(jnp.zeros(3))
        self.insert_variation = nnx.Variable(jnp.zeros(()))

    def __call__(self, x, t, insert, s, u, *, return_network_output=False):
        assert return_network_output
        i = {16: 0, 32: 1, 64: 2}[x.shape[1]]
        n = self.counts[i]
        if i > 0:
            self.wrong_order[...] += (self.counts[i - 1] < 32).astype(jnp.int32)
        total = insert.sum()
        first = jnp.where(n == 0, total, self.insert_sums[i])
        self.insert_sums[...] = self.insert_sums[...].at[i].set(first)
        self.insert_variation[...] = jnp.maximum(self.insert_variation[...], jnp.abs(total - first))
        self.counts[...] = self.counts[...].at[i].add(1)
        return (s - x) / jnp.sqrt(jnp.maximum(t, 1e-6))


class InsertHead(nnx.Module):
    def __call__(self, x, t, insert, s, u, *, return_network_output=False):
        assert return_network_output
        endpoint = jnp.repeat(jnp.repeat(insert, 2, axis=1), 2, axis=2)
        return (endpoint - x) / jnp.sqrt(jnp.maximum(t, 1e-6))


def test_cascade_finishes_small_stages_before_larger_and_keeps_insert_fixed():
    model = CountingHead()
    s = jnp.ones((1, 64, 64, 1)).at[:, 10:50, 29:35].set(-0.5)
    t = jnp.full_like(s, 0.5).at[:, :, :32].set(0)
    out = sample_cascade(model, s, t, jax.random.PRNGKey(14), depth=3)
    np.testing.assert_array_equal(model.counts[...], [32, 32, 32])
    assert int(model.wrong_order[...]) == 0 and float(model.insert_variation[...]) == 0
    np.testing.assert_allclose(out, s, atol=2e-7, rtol=0)
    mask = np.asarray(t) == 0
    np.testing.assert_array_equal(np.asarray(out)[mask], np.asarray(s)[mask])


def test_oracle_runs_only_finest_stage_and_teacher_can_change_output():
    s = jnp.ones((1, 64, 64, 1))
    t = jnp.full_like(s, 0.5).at[:, :, :32].set(0)
    key = jax.random.PRNGKey(14)
    model = CountingHead()
    sample_cascade(model, s, t, key, depth=3, target=-s, oracle=True)
    np.testing.assert_array_equal(model.counts[...], [0, 0, 32])
    assert float(model.insert_variation[...]) == 0
    outputs = [
        sample_cascade(InsertHead(), s, t, key, depth=3, target=y * s, oracle=True)
        for y in (-0.8, 0.5)
    ]
    assert float(jnp.max(jnp.abs(outputs[0] - outputs[1]))) > 1
    for out in outputs:
        assert np.isfinite(out).all()
        mask = np.asarray(t) == 0
        np.testing.assert_array_equal(np.asarray(out)[mask], np.asarray(s)[mask])


@pytest.mark.parametrize("mode", ["recursive", "cascade"])
def test_non_oracle_rejects_teacher(mode):
    s = jnp.ones((1, 32, 32, 1))
    with pytest.raises(ValueError):
        run_sample(InsertHead(), s, s * 0.5, jax.random.PRNGKey(0), 2, f"control_{mode}", target=s)


def test_oracle_requires_teacher():
    s = jnp.ones((1, 32, 32, 1))
    with pytest.raises(ValueError):
        run_sample(InsertHead(), s, s * 0.5, jax.random.PRNGKey(0), 2, "control_oracle")


def test_actual_depth1_parity_and_unchanged_recursive_baseline(tmp_path):
    _, model, _, p = modulated(tmp_path)
    s, t = p["guides"][0], p["levels"][0]
    key = jax.random.PRNGKey(38)
    baseline = run_sample(model, s, t, key, 1, "control_recursive")
    np.testing.assert_array_equal(baseline, original_sample(model, s, t, key, 1))
    for mode in ("cascade", "oracle"):
        out = run_sample(
            model, s, t, key, 1, f"control_{mode}", p["target"] if mode == "oracle" else None
        )
        np.testing.assert_allclose(out, baseline, atol=1e-6, rtol=0)
        mask = np.asarray(t) == 0
        np.testing.assert_array_equal(out[mask], np.asarray(s)[mask])
