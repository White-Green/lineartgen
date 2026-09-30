"""ONNX inference only: importing this package never imports JAX or Flax."""

__version__ = "0.2.2"


def __getattr__(name):
    if name == "LineartModel":
        from .session import LineartModel

        return LineartModel
    raise AttributeError(name)
