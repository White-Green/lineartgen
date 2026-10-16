"""Initially silent spatial-context residual branch for the physical D4 CNN."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx
from group_field_model import GroupBlock
from physical_field_model import PhysicalGroupFieldCNN
from physical_field_model import load_field as load_physical
from scale_dilation_model import convolve
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS, infer
from subpixel_field_model import save_field as save_legacy

from lineartgen.config import load_config
from lineartgen.models.serialization import _arrays
from lineartgen.models.unet import progress_basis
from lineartgen.models.upsample import nearest_double

TAG = "d4_group_physical_context"
CONTEXT_DILATION = 32


class ContextReachFieldCNN(PhysicalGroupFieldCNN):
    def __init__(self, config, *, rngs, context_dilation=CONTEXT_DILATION):
        super().__init__(config, rngs=rngs)
        assert context_dilation in (1, CONTEXT_DILATION)
        self.research_model = TAG
        self.context_block = GroupBlock(self.width, context_dilation, rngs=rngs)
        self.context_block.second.kernel[...] = jnp.zeros_like(self.context_block.second.kernel)
        self.context_block.second.bias[...] = jnp.zeros_like(self.context_block.second.bias)

    def __call__(
        self,
        image,
        noise_level,
        insert,
        guide,
        progress,
        *,
        pyramid_level=0,
        return_network_output=False,
    ):
        assert return_network_output
        assert image.shape == noise_level.shape == guide.shape == progress.shape
        features = jnp.concatenate(
            (guide - 1, guide - 1, noise_level, nearest_double(insert) - 1), -1
        )
        basis = progress_basis(jnp.full_like(progress, pyramid_level / 5))
        x = nnx.silu(
            self.stem_modulation(convolve(self.stem, features, pyramid_level, "physical"), basis)
        )
        for block in self.blocks:
            y = nnx.silu(
                block.modulation(convolve(block.first, x, pyramid_level, "physical"), basis)
            )
            x = nnx.silu(x + convolve(block.second, y, pyramid_level, "physical"))
        block = self.context_block
        y = nnx.silu(block.modulation(convolve(block.first, x, pyramid_level, "physical"), basis))
        # Deliberately no final SiLU: zero residual must preserve the old function.
        x = x + convolve(block.second, y, pyramid_level, "physical")
        return self.head(x)


def add_context(model, config, *, dilation=CONTEXT_DILATION):
    assert model.research_model == "d4_group_physical"
    result = ContextReachFieldCNN(config.model, rngs=nnx.Rngs(0), context_dilation=dilation)
    nnx.update(result, nnx.state(model, nnx.Param))
    return result


def save_field(model, config, destination, *, provenance):
    assert model.research_model == TAG
    dilation = model.context_block.first.dilation
    assert dilation == model.context_block.second.dilation and dilation in (1, CONTEXT_DILATION)
    dest = save_legacy(model, config, destination, provenance=provenance)
    p = dest / "metadata.json"
    metadata = json.loads(p.read_text())
    metadata.update(
        research_model=TAG,
        group_order=8,
        dilation_rule="max(1, base // 2**pyramid_level)",
        extra_context_dilation=dilation,
        extra_context_skip="linear",
    )
    p.write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def load_field(source):
    source = Path(source)
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("research_model") != TAG:
        return load_physical(source)
    assert metadata["format_version"] == 2
    assert metadata["representation"] == "subpixel_geometry_field"
    assert metadata["subpixels"] == SUBPIXELS and metadata["distance_range"] == DISTANCE_RANGE
    assert metadata["group_order"] == 8
    assert metadata["dilation_rule"] == "max(1, base // 2**pyramid_level)"
    assert metadata["extra_context_dilation"] in (1, CONTEXT_DILATION)
    assert metadata["extra_context_skip"] == "linear"
    assert metadata["production_runtime_supported"] is False
    cfg = load_config(source / "config.json")
    model = ContextReachFieldCNN(
        cfg.model, rngs=nnx.Rngs(0), context_dilation=metadata["extra_context_dilation"]
    )
    state = nnx.state(model, nnx.Param)
    template = nnx.to_pure_dict(state)
    with np.load(source / "weights.npz", allow_pickle=False) as z:
        assert set(z.files) == {k for k, _ in _arrays(template)}

        def restore(tree, path=()):
            if isinstance(tree, dict):
                return {k: restore(v, (*path, k)) for k, v in tree.items()}
            a = z[json.dumps(path)]
            assert a.shape == tree.shape and a.dtype == tree.dtype
            return jnp.asarray(a)

        nnx.replace_by_pure_dict(state, restore(template))
    nnx.update(model, state)
    return model, cfg


__all__ = ["ContextReachFieldCNN", "add_context", "infer", "load_field", "save_field"]
