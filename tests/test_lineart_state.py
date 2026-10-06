import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lineartgen.config import ExperimentConfig
from lineartgen.diffusion import training_inputs
from lineartgen.models import create_model
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart import Profiles
from lineartgen.training.lineart_study import create_arm, update_arm


def test_joint_critic_generator_optimizer_and_ema_resume_matches_uninterrupted(tmp_path):
    config = ExperimentConfig.model_validate(
        dict(
            model=dict(channels=[2, 4], convolutions_per_block=1, output_mode="residual"),
            matching=dict(method="independent"),
            train=dict(batch_size=1, accumulate_steps=2),
        )
    )
    initial = create_model(config.model, rngs=nnx.Rngs(3))

    def create():
        return {"prior": create_arm(config, initial, adversarial=True, patch_size=16)}

    full, resumed = create(), create()

    def advance(arms, step):
        clean = jnp.ones((2, 32, 32, 1)).at[:, :, 15:17].set(-1)
        prepared = training_inputs(
            clean, jax.random.PRNGKey(step), config.diffusion, config.matching, levels=2
        )
        profiles = Profiles(jnp.zeros((2, 1, 3, 2)), jnp.zeros((2, 1)))
        coordinates = jnp.zeros((2, 4, 2), jnp.int32)
        update_arm(arms["prior"], prepared, profiles, coordinates, prior_weight=0.1)

    advance(full, 0)
    advance(resumed, 0)
    with checkpoints.manager(tmp_path / "checkpoint") as saved:
        lineart_state.save(saved, resumed, 1)
        resumed = create()
        assert lineart_state.restore(saved, resumed) == 1
    advance(full, 1)
    advance(resumed, 1)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(full, 2),
        lineart_state.payload(resumed, 2),
    )
