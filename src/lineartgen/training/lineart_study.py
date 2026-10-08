"""Factories for the five fixed loss ablations and one full optimizer update."""

import math

import jax
import numpy as np
from flax import nnx

from lineartgen.models import create_model

from .lineart_adversarial import PatchCritic
from .lineart_steps import accumulate_critic, accumulate_generator, make_lineart_steps
from .steps import create_optimizer, make_steps

METHODS = ("control", "no_contrast", "regions", "profiles", "patch_prior")


def assert_initial_parameters(initial, model):
    """Exact inherited tensors and function-preserving optional additions."""
    original = nnx.to_pure_dict(nnx.state(initial, nnx.Param))
    current = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    extras = current.keys() - original.keys()
    assert extras <= {"guide_encoder", "guide_head", "detail_head"}
    assert ("guide_encoder" in extras) == ("guide_head" in extras)
    jax.tree.map(np.testing.assert_array_equal, original, {k: current[k] for k in original})
    for key in extras:
        state = current[key]
        if key == "detail_head":
            # Only the last projection is zero. Zeroing all layers would prevent
            # the hidden path from learning through ReLU.
            state = state["layers"][max(state["layers"])]
        for value in jax.tree.leaves(state):
            np.testing.assert_array_equal(value, 0)


def study_methods(protocol):
    """Explicit arm names allow subsequent studies to reuse the paired trainer."""
    names = tuple(protocol.get("methods", METHODS))
    if not names or len(set(names)) != len(names):
        raise ValueError("Study methods must be nonempty and unique")
    if any(
        not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
        for name in names
    ):
        raise ValueError("Invalid study method name")
    return names


def study_configs(base, coefficients):
    no_contrast = base.loss.model_copy(update={"contrast_weight": 0.0})
    return {
        "control": base,
        "no_contrast": base.model_copy(update={"loss": no_contrast}),
        "regions": base.model_copy(
            update={
                "loss": no_contrast.model_copy(
                    update={"line_region_weight": coefficients["regions"]}
                )
            }
        ),
        "profiles": base.model_copy(
            update={
                "loss": no_contrast.model_copy(
                    update={"line_profile_weight": coefficients["profiles"]}
                )
            }
        ),
        "patch_prior": base.model_copy(update={"loss": no_contrast}),
    }


def create_arm(config, initial, *, adversarial=False, patch_size=64):
    model = create_model(config.model, rngs=nnx.Rngs(config.train.initialization_seed))
    nnx.update(model, jax.tree.map(lambda x: x.copy(), nnx.state(initial, nnx.Param)))
    optimizer, schedule = create_optimizer(model, config)
    gradient, critic_gradient, critic_update = make_lineart_steps(
        config, adversarial=adversarial, patch_size=patch_size
    )
    arm = dict(
        config=config,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        ema=jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param)),
        gradient=gradient,
        update=make_steps(config)[1],
        critic=None,
        critic_gradient=critic_gradient,
        critic_update=critic_update,
    )
    if adversarial:
        arm["critic"] = PatchCritic(rngs=nnx.Rngs(config.train.initialization_seed + 5301))
        critic_config = config.model_copy(
            update={
                "train": config.train.model_copy(
                    update={
                        "learning_rate": config.train.learning_rate / 2,
                        "minimum_learning_rate": config.train.minimum_learning_rate / 2,
                    }
                )
            }
        )
        arm["critic_optimizer"], arm["critic_schedule"] = create_optimizer(
            arm["critic"], critic_config
        )
    return arm


def update_arm(arm, prepared, profiles, coordinates, *, prior_weight=0.0):
    train = arm["config"].train
    logged = {}
    if arm["critic"] is not None:
        metrics, gradient, norm = accumulate_critic(
            arm["critic_gradient"], arm["critic"], arm["model"], prepared, coordinates, train
        )
        if not math.isfinite(float(norm)) or not math.isfinite(float(metrics["critic_loss"])):
            raise FloatingPointError("Nonfinite critic loss/gradient")
        arm["critic_update"](arm["critic"], arm["critic_optimizer"], gradient)
        logged = {k: float(v) for k, v in metrics.items()} | {"critic_gradient_norm": float(norm)}
    metrics, gradient, norm = accumulate_generator(
        arm["gradient"],
        arm["model"],
        prepared,
        profiles,
        coordinates,
        arm["critic"],
        prior_weight,
        train,
    )
    if not math.isfinite(float(norm)) or not math.isfinite(float(metrics["loss"])):
        raise FloatingPointError("Nonfinite generator loss/gradient")
    _, arm["ema"] = arm["update"](arm["model"], arm["optimizer"], gradient, arm["ema"])
    jax.block_until_ready((nnx.state(arm["model"]), arm["ema"]))
    return logged | {k: float(v) for k, v in metrics.items()} | {"gradient_norm": float(norm)}
