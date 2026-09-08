import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lineartgen.config import DiffusionConfig, ExperimentConfig
from lineartgen.diffusion import TrainingInputs
from lineartgen.models import UNet
from lineartgen.training.steps import create_optimizer, make_steps


def test_dataset_scale_controls_recursion_and_caps_at_model_minimum():
    recursion = DiffusionConfig()
    assert [recursion.levels_for_scale(512, 16, scale) for scale in range(6)] == [1, 2, 3, 4, 5, 6]
    assert recursion.levels_for_scale(512, 16, 100) == 6
    assert recursion.levels_for_scale(512, 64, 5) == 4
    stopped = DiffusionConfig(original_steps=2, stop_size=128)
    assert [stopped.levels_for_scale(512, 16, scale) for scale in range(6)] == [2, 3, 3, 3, 3, 3]


class SeparateScaleModel(nnx.Module):
    """Independent parameters make a detached recursive insert observable."""

    def __init__(self):
        self.small = nnx.Param(jnp.array(0.1))
        self.large = nnx.Param(jnp.array(0.2))

    def __call__(self, image, noise, insert):
        value = self.small[...] if image.shape[1] == 8 else self.large[...] + insert.mean()
        return jnp.full_like(image, value)


@pytest.mark.parametrize("multiplier", [0.0, 0.5])
def test_high_resolution_loss_backpropagates_through_smaller_prediction(multiplier):
    config = ExperimentConfig.model_validate(
        {
            "diffusion": {"scale_loss_multiplier": multiplier},
            "loss": {"balance_tones": False, "edge_weight": 0, "contrast_weight": 0},
        }
    )
    model = SeparateScaleModel()
    targets = (jnp.zeros((1, 16, 16, 1)), jnp.full((1, 8, 8, 1), 0.75))
    prepared = TrainingInputs(
        targets, tuple(jnp.ones_like(x) for x in targets), tuple(jnp.zeros_like(x) for x in targets)
    )
    gradient_step, _, _ = make_steps(config)
    metrics, gradients = gradient_step(model, prepared)
    expected = (0.3**2 + multiplier * (0.1 - 0.75) ** 2) / (1 + multiplier)
    np.testing.assert_allclose(metrics["loss"], expected, rtol=1e-5)
    np.testing.assert_allclose(gradients["large"][...], 0.6 / (1 + multiplier), rtol=1e-5)
    np.testing.assert_allclose(
        gradients["small"][...], (0.6 + multiplier * 2 * (0.1 - 0.75)) / (1 + multiplier), rtol=1e-4
    )


@pytest.mark.parametrize("batch,accumulate", [(4, 4), (8, 2), (16, 1), (4, 16)])
def test_learning_rate_scales_with_effective_batch_not_microbatch(batch, accumulate):
    config = ExperimentConfig.model_validate(
        {
            "model": {"channels": [2], "convolutions_per_block": 1},
            "train": {
                "batch_size": batch,
                "accumulate_steps": accumulate,
                "warmup_steps": 2,
                "steps": 10,
                "learning_rate": 1e-4,
                "minimum_learning_rate": 1e-5,
            },
        }
    )
    model = UNet(config.model, rngs=nnx.Rngs(0))
    _, schedule = create_optimizer(model, config)
    peak = 1e-4 * np.sqrt(batch * accumulate)
    np.testing.assert_allclose(schedule(2), peak, rtol=1e-6)
    np.testing.assert_allclose(schedule(0), peak / 10, rtol=1e-6)
    np.testing.assert_allclose(schedule(10), 1e-5, rtol=1e-6)


def test_minimum_rate_is_absolute_and_validated_against_scaled_peak():
    config = ExperimentConfig.model_validate(
        {
            "train": {
                "batch_size": 4,
                "accumulate_steps": 1,
                "learning_rate": 1e-4,
                "minimum_learning_rate": 1.5e-4,
            }
        }
    )
    assert config.train.peak_learning_rate == 2e-4
