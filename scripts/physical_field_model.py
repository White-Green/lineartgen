"""D4 geometry CNN whose dilation is measured in full-resolution pixel units."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx
from group_field_model import GroupFieldCNN
from scale_dilation_model import convolve
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS, infer
from subpixel_field_model import load_field as load_legacy
from subpixel_field_model import save_field as save_legacy

from lineartgen.config import load_config
from lineartgen.models.serialization import _arrays
from lineartgen.models.unet import progress_basis
from lineartgen.models.upsample import nearest_double

TAG = "d4_group_physical"


class PhysicalGroupFieldCNN(GroupFieldCNN):
    def __init__(self, config, *, rngs):
        super().__init__(config, rngs=rngs)
        self.research_model = TAG

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
        return self.head(x)


def convert(model, config):
    assert model.research_model == "d4_group_local"
    result = PhysicalGroupFieldCNN(config.model, rngs=nnx.Rngs(0))
    nnx.update(result, nnx.state(model, nnx.Param))
    return result


def save_field(model, config, destination, *, provenance):
    assert model.research_model == TAG
    dest = save_legacy(model, config, destination, provenance=provenance)
    p = dest / "metadata.json"
    metadata = json.loads(p.read_text())
    metadata.update(
        research_model=TAG, group_order=8, dilation_rule="max(1, base // 2**pyramid_level)"
    )
    p.write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def load_field(source):
    source = Path(source)
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("research_model") != TAG:
        return load_legacy(source)
    assert metadata["format_version"] == 2
    assert metadata["representation"] == "subpixel_geometry_field"
    assert metadata["subpixels"] == SUBPIXELS and metadata["distance_range"] == DISTANCE_RANGE
    assert metadata["group_order"] == 8
    assert metadata["dilation_rule"] == "max(1, base // 2**pyramid_level)"
    assert metadata["production_runtime_supported"] is False
    cfg = load_config(source / "config.json")
    model = PhysicalGroupFieldCNN(cfg.model, rngs=nnx.Rngs(0))
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


__all__ = ["PhysicalGroupFieldCNN", "convert", "infer", "load_field", "save_field"]
