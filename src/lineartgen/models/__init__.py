"""Add model architectures here; training depends only on the forward contract."""

from .nested import NestedUNet
from .unet import UNet


def create_model(config, *, rngs):
    return (NestedUNet if config.architecture == "nested" else UNet)(config, rngs=rngs)


__all__ = ["NestedUNet", "UNet", "create_model"]
