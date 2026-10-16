"""Train one D4 contribution as a complete image instead of averaging fields."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx
from group_field_model import action
from mixed_context_model import CONTEXT_DILATION, MixedContextFieldCNN
from mixed_context_model import load_field as load_previous
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS, infer
from subpixel_field_model import save_field as save_legacy

from lineartgen.config import load_config
from lineartgen.models.serialization import _arrays

TAG = "d4_group_physical_choice_readout"


class ChoiceHead(nnx.Module):
    def __init__(self, linear, component=0):
        assert 0 <= component < 8
        self.linear = linear
        self.component = component

    def __call__(self, x):
        b, h, w, c = x.shape
        patches = self.linear(x.reshape(b, h, w, 8, c // 8)).reshape(b, h, w, 8, 4, 4)
        patch = action(patches[..., self.component, :, :], self.component, axes=(3, 4))
        return patch.reshape(b, h, w, 16)


class ChoiceReadoutFieldCNN(MixedContextFieldCNN):
    def __init__(self, config, *, rngs):
        super().__init__(config, rngs=rngs)
        self.research_model = TAG
        self.head = ChoiceHead(self.head.linear)


def select_readout(model, config):
    assert model.research_model == "d4_group_physical_mixed_context"
    result = ChoiceReadoutFieldCNN(config.model, rngs=nnx.Rngs(0))
    nnx.update(result, nnx.state(model, nnx.Param))
    return result


def save_field(model, config, destination, *, provenance):
    assert model.research_model == TAG and model.head.component == 0
    dest = save_legacy(model, config, destination, provenance=provenance)
    p = dest / "metadata.json"
    metadata = json.loads(p.read_text())
    metadata.update(
        research_model=TAG,
        group_order=8,
        dilation_rule="max(1, base // 2**pyramid_level)",
        extra_context_dilation=CONTEXT_DILATION,
        extra_context_skip="linear",
        head_readout="single_group_contribution",
        component=0,
        output_equivariance="component set only; selected output need not be D4 equivariant",
        recursive_readout="same selected contribution at every level",
    )
    p.write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def load_field(source):
    source = Path(source)
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("research_model") != TAG:
        return load_previous(source)
    assert metadata["format_version"] == 2
    assert metadata["representation"] == "subpixel_geometry_field"
    assert metadata["subpixels"] == SUBPIXELS and metadata["distance_range"] == DISTANCE_RANGE
    assert metadata["group_order"] == 8
    assert metadata["dilation_rule"] == "max(1, base // 2**pyramid_level)"
    assert tuple(metadata["extra_context_dilation"]) == CONTEXT_DILATION
    assert metadata["extra_context_skip"] == "linear"
    assert metadata["head_readout"] == "single_group_contribution"
    assert metadata["component"] == 0
    assert metadata["output_equivariance"] == (
        "component set only; selected output need not be D4 equivariant"
    )
    assert metadata["recursive_readout"] == "same selected contribution at every level"
    assert metadata["production_runtime_supported"] is False
    cfg = load_config(source / "config.json")
    model = ChoiceReadoutFieldCNN(cfg.model, rngs=nnx.Rngs(0))
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


__all__ = [
    "ChoiceReadoutFieldCNN",
    "ChoiceHead",
    "select_readout",
    "infer",
    "load_field",
    "save_field",
]
