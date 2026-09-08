"""One-time, Python-only import of the original MPK weights; not used for new training.

The default JAX architecture has identical parameter shapes. This makes the
existing learned model usable in the plugin while new JAX experiments train.
"""

import argparse
import hashlib
from pathlib import Path

import jax.numpy as jnp
import msgpack
import numpy as np
from flax import nnx

from lineartgen.config import ExperimentConfig
from lineartgen.export import export_model
from lineartgen.models import UNet


def import_checkpoint(path):
    data = msgpack.unpackb(Path(path).read_bytes(), raw=False)
    unet = data["item"]["model"]["u_net"]
    model = UNet(ExperimentConfig().model, rngs=nnx.Rngs(0))

    def tensor(parameter):
        value = parameter["param"]
        if value["dtype"] not in ("F16", "F32"):
            raise ValueError(f"Unsupported saved dtype {value['dtype']}")
        return (
            np.frombuffer(value["bytes"], "<f2" if value["dtype"] == "F16" else "<f4")
            .reshape(value["shape"])
            .astype(np.float32)
        )

    def block(source, destination):
        for index, layer in enumerate(destination.layers):
            conv = source[f"conv{index + 1}"]
            weight = tensor(conv["weight"]).transpose(2, 3, 1, 0)
            bias = tensor(conv["bias"])
            if weight.shape != layer.kernel.shape or bias.shape != layer.bias.shape:
                raise ValueError("Checkpoint architecture does not match the default JAX U-Net")
            layer.kernel[...] = jnp.asarray(weight)
            layer.bias[...] = jnp.asarray(bias)

    for name in ("encoders", "decoders"):
        for source, destination in zip(unet[name], getattr(model, name), strict=True):
            block(source["conv"], destination)
    block(unet["final_conv"], model.head)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result = export_model(
        import_checkpoint(args.checkpoint),
        ExperimentConfig(),
        args.output,
        provenance=f"Original epoch 1500 checkpoint; MPK SHA256 {digest}",
    )
    print(result)


if __name__ == "__main__":
    main()
