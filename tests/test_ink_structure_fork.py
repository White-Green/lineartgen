import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from fork_ink_structure_study import transfer


def test_selected_arms_continue_exactly_after_full_state_fork(tmp_path):
    config = ExperimentConfig.model_validate(
        dict(
            model=dict(channels=[2, 4], convolutions_per_block=1, output_mode="residual"),
            matching=dict(method="independent"),
            train=dict(batch_size=1, accumulate_steps=1),
        )
    )
    initial = create_model(config.model, rngs=nnx.Rngs(3))

    def create():
        return {
            name: create_arm(
                config.model_copy(
                    update={"loss": config.loss.model_copy(update={"ink_keep_weight": weight})}
                ),
                initial,
                adversarial=False,
                patch_size=16,
            )
            for name, weight in [("control", 0.0), ("candidate", 0.2), ("dropped", 0.3)]
        }

    def advance(arms, step):
        clean = jnp.ones((1, 32, 32, 1)).at[:, :, 15:17].set(-1)
        prepared = training_inputs(
            clean, jax.random.PRNGKey(step), config.diffusion, config.matching, levels=2
        )
        profiles = Profiles(jnp.zeros((1, 1, 3, 2)), jnp.zeros((1, 1)))
        coordinates = jnp.zeros((1, 4, 2), jnp.int32)
        for arm in arms.values():
            update_arm(arm, prepared, profiles, coordinates, prior_weight=0.0)

    arms = create()
    advance(arms, 0)
    with checkpoints.manager(tmp_path / "parent") as saved:
        lineart_state.save(saved, arms, 1)
    child = transfer(tmp_path / "parent", tmp_path / "child", create, ["control", "candidate"], 1)
    assert set(child) == {"control", "candidate"}
    advance(arms, 1)
    advance(child, 1)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({k: arms[k] for k in child}, 2),
        lineart_state.payload(child, 2),
    )
