"""Embed a trained physical group CNN in a wider CNN with zero outgoing new paths."""

import jax.numpy as jnp
from flax import nnx
from physical_field_model import TAG, PhysicalGroupFieldCNN


def widen(model, config, *, width=12, seed=2026101516):
    assert model.research_model == TAG and width > model.width
    old = model.width
    cfg = config.model_copy(
        update=dict(model=config.model.model_copy(update=dict(channels=(width,))))
    )
    result = PhysicalGroupFieldCNN(cfg.model, rngs=nnx.Rngs(seed))
    result.stem.kernel[...] = result.stem.kernel[...].at[..., :old].set(model.stem.kernel[...])
    result.stem.bias[...] = result.stem.bias[...].at[:old].set(model.stem.bias[...])

    def film(target, source):
        for name in ("kernel", "bias"):
            a, b = getattr(target.linear, name), getattr(source.linear, name)
            value = a[...].at[..., :old].set(b[...][..., :old])
            a[...] = value.at[..., width : width + old].set(b[...][..., old:])

    film(result.stem_modulation, model.stem_modulation)
    for target, source in zip(result.blocks, model.blocks, strict=True):
        for name in ("first", "second"):
            a, b = getattr(target, name), getattr(source, name)
            value = a.kernel[...].at[..., :, :old].set(0)
            a.kernel[...] = value.at[..., :old, :old].set(b.kernel[...])
            a.bias[...] = a.bias[...].at[:old].set(b.bias[...])
        film(target.modulation, source.modulation)
    result.head.linear.kernel[...] = (
        jnp.zeros_like(result.head.linear.kernel[...]).at[:old].set(model.head.linear.kernel[...])
    )
    result.head.linear.bias[...] = model.head.linear.bias[...]
    return result, cfg
