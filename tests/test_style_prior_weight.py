import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from plausible_objective import LinePrior
from style_prior_weight_study import update
from style_prior_weights import load_prior, save_prior
from test_style_prior import inputs, make


def test_pretrained_critic_transfer_exact_and_rejects_invalid_state(tmp_path):
    source = LinePrior(rngs=nnx.Rngs(7))
    target = LinePrior(rngs=nnx.Rngs(9))
    source.head.bias[...] += 2
    path = tmp_path / "prior.npz"
    save_prior(source, path)
    load_prior(target, path)
    jax.tree.map(np.testing.assert_array_equal, nnx.state(source), nnx.state(target))
    image = jnp.arange(64 * 64, dtype=jnp.float32).reshape(1, 64, 64, 1) / 4096
    np.testing.assert_array_equal(source(image, 0.4), target(image, 0.4))
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    key = next(iter(data))
    data[key] = data[key].astype(np.float64)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="shape/dtype"):
        load_prior(target, path)
    np.savez(path, wrong=np.ones(1))
    with pytest.raises(ValueError, match="keys"):
        load_prior(target, path)


def test_weight_changes_generator_update_without_changing_first_critic_update():
    _, prepared, candidates, annotation = inputs()
    annotation = annotation | dict(prior_ramp=jnp.asarray(1.0))
    arms = [make(), make()]
    results = []
    for weight, arm in zip((0.2, 2.0), arms, strict=True):
        arm["prior_weight"] = weight
        results.append(update(arm, prepared, candidates, annotation))
    for k in ("family", "profile", "range", "prior", "critic_loss", "critic_r1"):
        assert results[0][k] == results[1][k]
    np.testing.assert_allclose(
        results[1]["loss"] - results[0]["loss"], 1.8 * results[0]["prior"], rtol=1e-6
    )
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(arms[0]["critic"]), nnx.state(arms[1]["critic"])
    )
    assert any(
        np.any(np.asarray(a) != np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(nnx.state(arms[0]["model"])),
            jax.tree.leaves(nnx.state(arms[1]["model"])),
            strict=True,
        )
    )
