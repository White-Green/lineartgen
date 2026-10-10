import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import ExperimentConfig, ModelConfig, TrainConfig
from lineartgen.models import create_model
from lineartgen.training.steps import create_optimizer


def test_explicit_decay_horizon_preserves_old_schedule_past_the_original_end():
    config = ExperimentConfig(
        model=ModelConfig(channels=(2, 4), convolutions_per_block=1),
        train=TrainConfig(
            steps=4000,
            batch_size=4,
            accumulate_steps=1,
            learning_rate=5e-5,
            minimum_learning_rate=1e-5,
            warmup_steps=100,
        ),
    )
    model = create_model(config.model, rngs=nnx.Rngs(42))
    extended = config.model_copy(
        update={
            "train": config.train.model_copy(
                update={"steps": 16000, "learning_rate_decay_steps": 4000}
            )
        }
    )
    _, old_schedule = create_optimizer(model, config)
    _, new_schedule = create_optimizer(model, extended)
    steps = jnp.arange(16001)
    np.testing.assert_array_equal(old_schedule(steps), new_schedule(steps))
    np.testing.assert_allclose(new_schedule(steps)[4000:], 1e-5, rtol=1e-6)
    assert extended.train.peak_learning_rate == config.train.peak_learning_rate == 1e-4
    assert TrainConfig().learning_rate_decay_steps is None


def test_explicit_decay_horizon_rejects_warmup_longer_than_decay():
    with pytest.raises(ValueError, match="learning_rate_decay_steps"):
        ExperimentConfig(
            train=TrainConfig(steps=1000, learning_rate_decay_steps=100, warmup_steps=100)
        )
