import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import nnx

from lineartgen.training.lineart_adversarial import PatchCritic

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from probe_critic_repair_gradients import repair_gradient
from train_critic_defect_pilot import spatial_logits


@pytest.mark.parametrize("mode", ["scalar", "spatial"])
def test_repair_direction_matches_finite_difference(mode):
    model = PatchCritic(rngs=nnx.Rngs(12))
    fake = jax.random.normal(jax.random.PRNGKey(7), (2, 64, 64, 3)) * 0.1
    real = fake.at[..., 2].add(0.2)
    gain, cosine = repair_gradient(mode)(model, real, fake)

    def losses(x):
        logits = spatial_logits(model, x)
        return (
            jax.nn.softplus(-logits.mean((1, 2, 3)))
            if mode == "scalar"
            else jax.nn.softplus(-logits).mean((1, 2, 3))
        )

    epsilon = 0.001
    difference = (
        losses(fake - epsilon * (real - fake)) - losses(fake + epsilon * (real - fake))
    ) / (2 * epsilon)
    np.testing.assert_allclose(gain, difference, atol=0.0001, rtol=0.02)
    assert np.isfinite(cosine).all() and np.max(np.abs(cosine)) <= 1
    unchanged, _ = repair_gradient(mode)(model, fake, fake)
    np.testing.assert_array_equal(unchanged, [0, 0])
