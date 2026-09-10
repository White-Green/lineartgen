"""Portable inference weights: strict config + named NumPy arrays, no optimizer state."""

import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from lineartgen.config import load_config, save_config

from . import create_model


def _arrays(tree, path=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _arrays(value, (*path, key))
    else:
        yield json.dumps(path), tree


def save_inference(model, config, destination, *, provenance):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Inference destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    save_config(config, destination / "config.json")
    state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    arrays = {key: np.asarray(value) for key, value in _arrays(state)}
    np.savez(destination / "weights.npz", **arrays)
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "parameters": sum(value.size for value in arrays.values()),
                "subnets": getattr(model, "num_subnets", 1),
                "provenance": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def load_inference(source):
    source = Path(source)
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported inference weight format")
    config = load_config(source / "config.json")
    model = create_model(config.model, rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    template = nnx.to_pure_dict(state)
    with np.load(source / "weights.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {key for key, _ in _arrays(template)}:
            raise ValueError("Weight keys do not match the saved architecture")

        def restore(tree, path=()):
            if isinstance(tree, dict):
                return {key: restore(value, (*path, key)) for key, value in tree.items()}
            value = arrays[json.dumps(path)]
            if value.shape != tree.shape or value.dtype != tree.dtype:
                raise ValueError(f"Weight shape/dtype mismatch at {path}")
            return jax.numpy.asarray(value)

        nnx.replace_by_pure_dict(state, restore(template))
    nnx.update(model, state)
    return model, config


def extract_run(run_dir, destination, index, *, step=None):
    from lineartgen.training.checkpoints import load_run_model

    model, config, step = load_run_model(run_dir, step=step)
    if not hasattr(model, "extract"):
        raise ValueError("This run has no nested subnets")
    model = model.extract(index)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"nested_channels": config.model.nested_channels[:index]}
            )
        }
    )
    return save_inference(
        model, config, destination, provenance=f"{run_dir}: EMA step {step}, subnet {index}"
    )
