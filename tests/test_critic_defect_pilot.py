import sys
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from lineartgen.training.lineart_adversarial import PatchCritic

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from train_critic_defect_pilot import spatial_logits


def test_spatial_logits_match_original_critic_and_local_mask_receptive_field():
    critic = PatchCritic(rngs=nnx.Rngs(17))
    image = jax.random.normal(jax.random.PRNGKey(3), (1, 64, 64, 3))
    np.testing.assert_allclose(
        np.asarray(spatial_logits(critic, image).mean((1, 2, 3))),
        np.asarray(critic(image)),
        atol=0,
        rtol=0,
    )
    gradient = jax.grad(lambda x: spatial_logits(critic, x)[0, 3, 3, 0])(image)
    support = np.argwhere(np.any(np.asarray(gradient)[0] != 0, axis=-1))
    np.testing.assert_array_equal(support.min(0), [16, 16])
    np.testing.assert_array_equal(support.max(0), [46, 46])
