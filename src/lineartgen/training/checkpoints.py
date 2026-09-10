"""Orbax checkpoints include model, optimizer, EMA and the next input-stream position."""

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


def manager(directory, keep=3):
    return ocp.CheckpointManager(
        directory.resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep,
            enable_async_checkpointing=False,
        ),
    )


def payload(model, optimizer, ema, step):
    return {
        "model": nnx.to_pure_dict(nnx.state(model)),
        "optimizer": nnx.to_pure_dict(nnx.state(optimizer)),
        "ema": nnx.to_pure_dict(ema),
        "step": np.asarray(step, dtype=np.int64),
    }


def restore_into(node, values):
    state = nnx.state(node)
    nnx.replace_by_pure_dict(state, jax.tree.map(jax.numpy.asarray, values))
    nnx.update(node, state)


def restore(checkpoints, model, optimizer, ema, step=None):
    step = checkpoints.latest_step() if step is None else step
    if step is None:
        raise FileNotFoundError("No checkpoint exists in this run")
    restored = checkpoints.restore(
        step, args=ocp.args.StandardRestore(payload(model, optimizer, ema, step))
    )
    restore_into(model, restored["model"])
    restore_into(optimizer, restored["optimizer"])
    nnx.replace_by_pure_dict(ema, jax.tree.map(jax.numpy.asarray, restored["ema"]))
    return int(restored["step"]), ema


def save(checkpoints, model, optimizer, ema, step):
    checkpoints.save(step, args=ocp.args.StandardSave(payload(model, optimizer, ema, step)))
    checkpoints.wait_until_finished()


def load_run_model(run_dir, *, step=None, use_ema=True):
    from pathlib import Path

    from lineartgen.config import load_config
    from lineartgen.models import create_model

    from .steps import create_optimizer

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.json")
    model = create_model(config.model, rngs=nnx.Rngs(config.train.initialization_seed))
    optimizer, _ = create_optimizer(model, config)
    ema = nnx.state(model, nnx.Param)
    with manager(run_dir / "checkpoints", config.train.keep_checkpoints) as saved:
        step, ema = restore(saved, model, optimizer, ema, step)
    if use_ema:
        nnx.update(model, ema)
    return model, config, step
