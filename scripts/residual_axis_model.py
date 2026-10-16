"""Serializable research model with a mean base and a mean/selected residual head."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx
from mixed_context_model import MixedContextFieldCNN
from mixed_context_model import load_field as load_previous
from residual_choice_model import ResidualChoiceHead
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS, infer
from subpixel_field_model import save_field as save_legacy

from lineartgen.config import load_config
from lineartgen.models.serialization import _arrays

TAG = "d4_group_physical_residual_readout"


class ResidualAxisCNN(MixedContextFieldCNN):
    def __init__(self, config, *, mode, rngs):
        super().__init__(config, rngs=rngs)
        self.research_model = TAG
        self.head = ResidualChoiceHead(self.head.linear, mode=mode, rngs=rngs)


def add_readout(model, config, *, mode):
    assert model.research_model == "d4_group_physical_mixed_context"
    result = ResidualAxisCNN(config.model, mode=mode, rngs=nnx.Rngs(0))
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
        extra_context_dilation=[1, 32],
        extra_context_skip="linear",
        base_readout="eight_component_mean",
        residual_readout=model.head.mode,
        component=0,
        recursive_readout="same residual readout at every level",
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
    assert metadata["extra_context_dilation"] == [1, 32]
    assert metadata["extra_context_skip"] == "linear"
    assert metadata["base_readout"] == "eight_component_mean"
    assert metadata["residual_readout"] in ("mean", "choice")
    assert metadata["component"] == 0
    assert metadata["recursive_readout"] == "same residual readout at every level"
    assert metadata["production_runtime_supported"] is False
    cfg = load_config(source / "config.json")
    model = ResidualAxisCNN(cfg.model, mode=metadata["residual_readout"], rngs=nnx.Rngs(0))
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


__all__ = ["ResidualAxisCNN", "add_readout", "infer", "load_field", "save_field"]
