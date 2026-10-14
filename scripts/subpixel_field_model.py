"""Research-only subpixel geometry head and recursive latent field conditioning."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from lineartgen.config import load_config
from lineartgen.diffusion import source_guides
from lineartgen.models import create_model
from lineartgen.models.serialization import _arrays, save_inference

SUBPIXELS = 4
DISTANCE_RANGE = 8.0


def create_field_model(config, *, seed=0, initial=None):
    model = (
        create_model(config.model, rngs=nnx.Rngs(seed)) if initial is None else nnx.clone(initial)
    )
    model.head = nnx.Conv(
        config.model.channels[0],
        SUBPIXELS**2,
        (1, 1),
        kernel_init=nnx.initializers.zeros_init(),
        bias_init=nnx.initializers.ones_init(),
        rngs=nnx.Rngs(seed),
    )
    return model


def unpack(head):
    b, h, w, _ = head.shape
    return (
        head.reshape(b, h, w, SUBPIXELS, SUBPIXELS)
        .transpose(0, 1, 3, 2, 4)
        .reshape(b, h * SUBPIXELS, w * SUBPIXELS, 1)
    )


def coverage(distance):
    b, h, w, c = distance.shape
    a = jnp.clip(0.5 - SUBPIXELS * distance, 0, 1)
    a = a.reshape(b, h // SUBPIXELS, SUBPIXELS, w // SUBPIXELS, SUBPIXELS, c).mean((2, 4))
    return 1 - 2 * a


def field_forward(model, prepared):
    guides = source_guides(model, prepared.guides)
    outputs = [None] * len(guides)
    g = guides[-1]
    insert = jnp.zeros((g.shape[0], g.shape[1] // 2, g.shape[2] // 2, 1), g.dtype)
    for level in reversed(range(len(guides))):
        head = model(
            prepared.noisy[level],
            prepared.noise_levels[level],
            insert,
            guides[level],
            prepared.progress[level],
            pyramid_level=level,
            return_network_output=True,
        )
        outputs[level] = unpack(head) * DISTANCE_RANGE
        # Internal recursive state is a bounded signed geometry summary, not opacity.
        insert = jnp.clip(head.mean(-1, keepdims=True), -1, 1)
    return tuple(outputs)


@nnx.jit
def infer(model, prepared):
    fields = field_forward(model, prepared)
    image = coverage(fields[0])
    return jnp.where(prepared.noise_levels[0] > 0, image, prepared.guides[0])


def save_field(model, config, destination, *, provenance):
    dest = save_inference(model, config, destination, provenance=provenance)
    p = dest / "metadata.json"
    metadata = json.loads(p.read_text())
    metadata.update(
        format_version=2,
        representation="subpixel_geometry_field",
        subpixels=SUBPIXELS,
        distance_range=DISTANCE_RANGE,
        production_runtime_supported=False,
    )
    p.write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def load_field(source):
    source = Path(source)
    metadata = json.loads((source / "metadata.json").read_text())
    assert (
        metadata["format_version"] == 2 and metadata["representation"] == "subpixel_geometry_field"
    )
    assert metadata["subpixels"] == SUBPIXELS and metadata["distance_range"] == DISTANCE_RANGE
    cfg = load_config(source / "config.json")
    model = create_field_model(cfg)
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
