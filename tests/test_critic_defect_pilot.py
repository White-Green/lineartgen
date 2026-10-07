import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from lineartgen.training.lineart_adversarial import PatchCritic

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from train_critic_defect_pilot import critic_states, make_step, resume_states, spatial_logits


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


def test_critic_resume_preserves_optimizer_and_next_update(tmp_path):
    def create():
        model = PatchCritic(rngs=nnx.Rngs(17))
        optimizer = nnx.Optimizer(model, optax.adam(0.0001), wrt=nnx.Param)
        return {"localized": model}, {"localized": optimizer}

    models, optimizers = create()
    real = jax.random.normal(jax.random.PRNGKey(8), (2, 64, 64, 3))
    fake = real.at[..., 2].multiply(0.8)
    weights = jnp.ones((2, 8, 8, 1))
    step = make_step("localized")
    step(models["localized"], optimizers["localized"], real, fake, weights)
    path = tmp_path / "checkpoint"
    with ocp.StandardCheckpointer() as saver:
        saver.save(path, critic_states(models, optimizers))
        saver.wait_until_finished()
    restored, restored_optimizers = create()
    resume_states(path, restored, restored_optimizers, 1)
    step(models["localized"], optimizers["localized"], real, fake, weights)
    step(restored["localized"], restored_optimizers["localized"], real, fake, weights)
    jax.tree.map(
        np.testing.assert_array_equal,
        critic_states(models, optimizers),
        critic_states(restored, restored_optimizers),
    )
