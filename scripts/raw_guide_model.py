"""Add an initially silent raw-source cue to the physical D4 field CNN."""

import jax
import jax.numpy as jnp
from flax import nnx
from group_field_model import action
from physical_field_model import PhysicalGroupFieldCNN

TAG = "d4_group_physical_raw_guide"


class RawLift(nnx.Module):
    def __init__(self, width):
        self.kernel = nnx.Param(jnp.zeros((3, 3, 1, width), jnp.float32))

    def __call__(self, raw):
        kernel = jnp.concatenate([action(self.kernel[...], g) for g in range(8)], axis=-1)
        return jax.lax.conv_general_dilated(
            raw, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )


class RawGuideFieldCNN(PhysicalGroupFieldCNN):
    def __init__(self, config, *, rngs):
        super().__init__(config, rngs=rngs)
        self.research_model = TAG
        self.uses_raw_guide = True
        self.raw_stem = RawLift(self.width)


def add_raw_guide(model, config):
    assert model.research_model == "d4_group_physical"
    result = RawGuideFieldCNN(config.model, rngs=nnx.Rngs(0))
    nnx.update(result, nnx.state(model, nnx.Param))
    return result
