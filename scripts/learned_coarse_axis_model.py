"""Learn spatial shape features on a coarse guide, preserving the pooled axis model."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from group_field_model import GroupConv
from pooled_axis_model import TAG as PREVIOUS_TAG
from pooled_axis_model import PooledAxisCNN
from pooled_axis_model import load_field as load_previous
from pooled_context_model import POOL_RADIUS, pooled_cue
from scale_dilation_model import convolve
from subpixel_field_model import DISTANCE_RANGE, SUBPIXELS, infer
from subpixel_field_model import save_field as save_legacy

from lineartgen.config import load_config
from lineartgen.models.serialization import _arrays
from lineartgen.models.unet import progress_basis
from lineartgen.models.upsample import nearest_double

TAG = "d4_group_pooled_axis_learned_coarse"
COARSE_FACTOR = 16
COARSE_WIDTH = 2
COARSE_RULE = (
    "area mean, factor=max(1,16//2**level); three3x3 D4 convolutions; "
    "shared linear projection; bilinear upsample"
)


def coarse_average(guide, level):
    factor = max(1, COARSE_FACTOR // 2**level)
    b, h, w, c = guide.shape
    assert h % factor == w % factor == 0
    return (guide - 1).reshape(b, h // factor, factor, w // factor, factor, c).mean((2, 4))


class CoarseBranch(nnx.Module):
    def __init__(self, outputs, *, rngs):
        self.lift = GroupConv(1, COARSE_WIDTH, lifting=True, rngs=rngs)
        self.first = GroupConv(COARSE_WIDTH, COARSE_WIDTH, rngs=rngs)
        self.second = GroupConv(COARSE_WIDTH, COARSE_WIDTH, rngs=rngs)
        self.project = nnx.Linear(
            COARSE_WIDTH,
            outputs,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, guide, level):
        x = nnx.silu(self.lift(coarse_average(guide, level)))
        x = nnx.silu(self.first(x))
        x = nnx.silu(self.second(x))
        b, h, w, _ = x.shape
        x = self.project(x.reshape(b, h, w, 8, COARSE_WIDTH))
        x = x.reshape(b, h, w, -1)
        return jax.image.resize(
            x, (b, guide.shape[1], guide.shape[2], x.shape[-1]), method="linear", antialias=False
        )


class LearnedCoarseAxisCNN(PooledAxisCNN):
    def __init__(self, config, *, mode, rngs):
        super().__init__(config, mode=mode, rngs=rngs)
        self.research_model = TAG
        self.coarse_branch = CoarseBranch(self.width, rngs=rngs)

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
        stem = convolve(self.stem, features, pyramid_level, "physical")
        stem = stem + self.pooled_stem(pooled_cue(guide, pyramid_level))
        stem = stem + self.coarse_branch(guide, pyramid_level)
        x = nnx.silu(self.stem_modulation(stem, basis))
        for block in self.blocks:
            y = nnx.silu(
                block.modulation(convolve(block.first, x, pyramid_level, "physical"), basis)
            )
            x = nnx.silu(x + convolve(block.second, y, pyramid_level, "physical"))
        block = self.context_block
        y = nnx.silu(block.modulation(convolve(block.first, x, pyramid_level, "physical"), basis))
        x = x + convolve(block.second, y, pyramid_level, "physical")
        return self.head(x)


def add_coarse(model, config):
    assert model.research_model == PREVIOUS_TAG
    result = LearnedCoarseAxisCNN(config.model, mode=model.head.mode, rngs=nnx.Rngs(0))
    nnx.update(result, nnx.state(model, nnx.Param))
    return result


def metadata_values(mode):
    return dict(
        research_model=TAG,
        group_order=8,
        dilation_rule="max(1, base // 2**pyramid_level)",
        extra_context_dilation=[1, 32],
        extra_context_skip="linear",
        pooled_radius=POOL_RADIUS,
        pooled_rule="square mean, radius=max(1,64//2**level)",
        pooled_boundary="white",
        pooled_input="normalized guide minus one",
        base_readout="eight_component_mean",
        residual_readout=mode,
        component=0,
        recursive_readout="same residual readout at every level",
        coarse_factor=COARSE_FACTOR,
        coarse_width=COARSE_WIDTH,
        coarse_rule=COARSE_RULE,
        coarse_input="normalized guide minus one",
    )


def save_field(model, config, destination, *, provenance):
    assert model.research_model == TAG and model.head.component == 0
    assert model.head.mode in ("mean", "choice")
    dest = save_legacy(model, config, destination, provenance=provenance)
    p = dest / "metadata.json"
    metadata = json.loads(p.read_text())
    metadata.update(metadata_values(model.head.mode))
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
    mode = metadata["residual_readout"]
    assert mode in ("mean", "choice")
    assert all(metadata[k] == v for k, v in metadata_values(mode).items())
    assert metadata["production_runtime_supported"] is False
    cfg = load_config(source / "config.json")
    model = LearnedCoarseAxisCNN(cfg.model, mode=mode, rngs=nnx.Rngs(0))
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


__all__ = ["LearnedCoarseAxisCNN", "add_coarse", "infer", "load_field", "save_field"]
