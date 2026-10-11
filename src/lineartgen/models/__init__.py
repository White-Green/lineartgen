"""Add model architectures here; training depends only on the forward contract."""

from .local import LocalCNN
from .nested import NestedUNet
from .unet import UNet


def create_model(config, *, rngs):
    return {"nested": NestedUNet, "unet": UNet, "local": LocalCNN}[config.architecture](
        config, rngs=rngs
    )


__all__ = ["NestedUNet", "UNet", "LocalCNN", "create_model"]
