"""Atomic group checkpoints include the patch critic and both optimizer states."""

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from . import checkpoints


def payload(arms, step):
    state = {}
    for name, arm in arms.items():
        item = checkpoints.payload(arm["model"], arm["optimizer"], arm["ema"], step)
        if arm.get("critic") is not None:
            item["critic"] = nnx.to_pure_dict(nnx.state(arm["critic"]))
            item["critic_optimizer"] = nnx.to_pure_dict(nnx.state(arm["critic_optimizer"]))
        state[name] = item
    return dict(arms=state, step=np.asarray(step, np.int64))


def save(manager, arms, step):
    manager.save(step, args=ocp.args.StandardSave(payload(arms, step)))
    manager.wait_until_finished()


def restore(manager, arms, step=None):
    step = manager.latest_step() if step is None else step
    if step is None:
        raise FileNotFoundError("No saved lineart loss study checkpoint")
    result = manager.restore(step, args=ocp.args.StandardRestore(payload(arms, step)))
    for name, arm in arms.items():
        values = result["arms"][name]
        assert int(values["step"]) == step
        checkpoints.restore_into(arm["model"], values["model"])
        checkpoints.restore_into(arm["optimizer"], values["optimizer"])
        nnx.replace_by_pure_dict(arm["ema"], jax.tree.map(jax.numpy.asarray, values["ema"]))
        if arm.get("critic") is not None:
            checkpoints.restore_into(arm["critic"], values["critic"])
            checkpoints.restore_into(arm["critic_optimizer"], values["critic_optimizer"])
    return int(result["step"])
