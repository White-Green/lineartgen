"""Preserve a group-mean readout while learning an asymmetric residual."""

import jax.numpy as jnp
from flax import nnx
from group_field_model import GroupHead, action


class ResidualChoiceHead(nnx.Module):
    def __init__(self, linear, *, mode, rngs):
        assert mode in ("mean", "choice")
        self.linear = linear
        self.residual = nnx.Linear(
            linear.kernel.shape[0],
            16,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.mode = mode
        self.component = 0

    def __call__(self, x):
        base = GroupHead.__call__(self, x)
        b, h, w, c = x.shape
        patches = self.residual(x.reshape(b, h, w, 8, c // 8)).reshape(b, h, w, 8, 4, 4)
        if self.mode == "mean":
            parts = [action(patches[..., g, :, :], g, axes=(3, 4)) for g in range(8)]
            delta = jnp.stack(parts).mean(0)
        else:
            delta = action(patches[..., self.component, :, :], self.component, axes=(3, 4))
        return base + delta.reshape(b, h, w, 16)


def add_residual(model, *, mode):
    result = nnx.clone(model)
    result.head = ResidualChoiceHead(result.head.linear, mode=mode, rngs=nnx.Rngs(0))
    return result
