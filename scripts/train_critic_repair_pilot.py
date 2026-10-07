"""Compare endpoint classification, repair interpolation and conditional WGAN-GP.

Critic-only synthetic diagnostic. There are no generator or deployment updates.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write
from flax import nnx
from train_critic_defect_pilot import critic_states, dataset

from lineartgen.config import load_config
from lineartgen.models.serialization import load_inference
from lineartgen.training import lineart_state
from lineartgen.training.lineart_study import create_arm, study_methods


def candidate_gradient_penalty(model, real, fake, alpha):
    candidate = alpha * real[..., 2:] + (1 - alpha) * fake[..., 2:]

    def score(x):
        return model(jnp.concatenate((fake[..., :2], x), -1)).sum()

    gradient = jax.grad(score)(candidate)
    norm = jnp.sqrt(jnp.square(gradient).sum((1, 2, 3)) + 1e-12)
    return jnp.square(norm - 1).mean()


def critic_objective(model, real, fake, alpha, *, mode):
    positive, negative = model(real), model(fake)
    if mode == "wgan_gp":
        return (negative - positive).mean() + 10 * candidate_gradient_penalty(
            model, real, fake, alpha
        )
    endpoint = jax.nn.softplus(-positive) + jax.nn.softplus(negative)
    if mode == "endpoint":
        return endpoint.mean()
    mixed = model(alpha * real + (1 - alpha) * fake)
    interpolation = jax.nn.softplus(mixed) - alpha[:, 0, 0, 0] * mixed
    return ((endpoint + interpolation) * (2 / 3)).mean()


def make_step(mode):
    @nnx.jit
    def step(model, optimizer, real, fake, alpha):
        def objective(m):
            return critic_objective(m, real, fake, alpha, mode=mode)

        loss, gradient = nnx.value_and_grad(objective)(model)
        optimizer.update(model, gradient)
        return loss, optax.global_norm(gradient)

    return step


@nnx.jit
def diagnostic(model, real, fake):
    positive, negative = model(real), model(fake)
    gradient = jax.grad(lambda x: model(x).sum())(fake)[..., 2]
    repair = real[..., 2] - fake[..., 2]
    gain = (gradient * repair).sum((1, 2))
    cosine = gain / jnp.maximum(
        jnp.linalg.norm(gradient, axis=(1, 2)) * jnp.linalg.norm(repair, axis=(1, 2)), 1e-12
    )
    return positive - negative, gain, cosine


def run(parent, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = dict(
        steps=10000,
        methods=["endpoint", "interpolation", "wgan_gp"],
        batch_size=32,
        optimizer=dict(name="adam", learning_rate=0.0001, b1=0.9, b2=0.999),
        train_variants=list(range(64, 80)),
        train_seed=2026100813,
        interpolation_seed=2026100814,
        evaluation_variants=list(range(128, 136)),
        evaluation_seed=2026101113,
        generator_updates=0,
        loss=dict(
            endpoint="softplus(-real)+softplus(fake)",
            interpolation="2/3*(endpoint+softplus(mix)-alpha*mix), mix image uses repair alpha",
            wgan_gp="fake-real + 10*(candidate-gradient-L2-norm-1)^2 at repair interpolation",
        ),
        baseline_replay="Endpoint should replay the previous scalar 10k critic exactly; "
        "audit states",
        endpoint_reference="runs/critic-hard-negative-20261007-extended/checkpoint",
        gate="In both clean and strokes_strong conditions, require >=80% clean preference AND "
        ">=80% favorable repair direction for cut/bridge/weak; >=90% for halo. This is only a "
        "critic diagnostic gate, never permission to adopt a generator.",
        source_files={
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("scripts/train_critic_defect_pilot.py"),
                Path("src/lineartgen/training/lineart_adversarial.py"),
            ]
        },
        reference="https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf",
        limitations="Fixed paired artificial negatives, same geometry generator; no actual G "
        "outputs or natural sketches. Candidate channel only GP, common Adam settings, not a "
        "reproduction of the original full GAN experiment. GP does not guarantee repair direction.",
    )
    write(output / "plan.json", plan)
    protocol = read(parent / "protocol.json")
    initial, _ = load_inference(protocol["initial"])
    arms = {
        name: create_arm(
            load_config(parent / "configs" / f"{name}.json"),
            initial,
            adversarial=name == "patch_prior",
            patch_size=64,
        )
        for name in study_methods(protocol)
    }
    checkpoint = parent / "milestone-checkpoints"
    archive = read(checkpoint / "step-10000-sha256.json")
    for path, digest in archive["files"].items():
        assert sha(checkpoint / "10000" / path) == digest
    with ocp.CheckpointManager(
        checkpoint.resolve(), options=ocp.CheckpointManagerOptions(read_only=True)
    ) as saved:
        assert lineart_state.restore(saved, arms, 10000) == 10000
    original = arms["patch_prior"]["critic"]
    models = {name: nnx.clone(original) for name in plan["methods"]}
    optimizers = {
        name: nnx.Optimizer(model, optax.adam(0.0001), wrt=nnx.Param)
        for name, model in models.items()
    }
    updates = {name: make_step(name) for name in models}
    train, records = dataset(plan["train_variants"], plan["train_seed"], False)
    groups = {
        name: [i for i, r in enumerate(records) if r["defect"] == name]
        for name in ["halo", "weak", "cut", "bridge"]
    }
    random = np.random.default_rng(plan["train_seed"])
    alpha_random = np.random.default_rng(plan["interpolation_seed"])
    history = []
    for step in range(plan["steps"]):
        indices = np.concatenate([random.choice(items, 8) for items in groups.values()])
        real, fake = (jax.device_put(x[indices]) for x in train[:2])
        alpha = jax.device_put(alpha_random.uniform(size=(32, 1, 1, 1)).astype(np.float32))
        metrics = {}
        for name in models:
            loss, norm = updates[name](models[name], optimizers[name], real, fake, alpha)
            assert np.isfinite(float(loss)) and np.isfinite(float(norm)), (step, name)
            metrics[name] = dict(loss=float(loss), gradient_norm=float(norm))
        if (step + 1) % 100 == 0:
            row = dict(step=step + 1, methods=metrics)
            history.append(row)
            write(output / "progress.json", row)
            print(row, flush=True)
    state = critic_states(models, optimizers)
    with ocp.StandardCheckpointer() as saver:
        saver.save((output / "checkpoint").resolve(), state)
        saver.wait_until_finished()
    reference = Path(plan["endpoint_reference"])
    with ocp.StandardCheckpointer() as saved:
        old = saved.restore(
            reference.resolve(),
            target={name: state["endpoint"] for name in ["scalar", "spatial", "localized"]},
        )
    jax.tree.map(np.testing.assert_array_equal, state["endpoint"], old["scalar"])
    write(
        output / "replay-audit.json",
        dict(endpoint_model_optimizer_exact=True, reference=str(reference)),
    )
    values, records = dataset(plan["evaluation_variants"], plan["evaluation_seed"], True)
    rows, summary = [], []
    for name, model in ({"original": original} | models).items():
        grouped = defaultdict(list)
        for start in range(0, len(records), 64):
            margin, gain, cosine = diagnostic(
                model, values[0][start : start + 64], values[1][start : start + 64]
            )
            for record, a, b, c in zip(
                records[start : start + 64],
                np.asarray(margin),
                np.asarray(gain),
                np.asarray(cosine),
                strict=True,
            ):
                row = record | dict(
                    method=name, margin=float(a), repair_gain=float(b), repair_cosine=float(c)
                )
                assert all(np.isfinite(v) for v in [a, b, c])
                rows.append(row)
                grouped[record["source"], record["defect"]].append(row)
        for (source, defect), items in sorted(grouped.items()):
            limit = 0.9 if defect == "halo" else 0.8
            preference = float(np.mean([r["margin"] > 0 for r in items]))
            repair = float(np.mean([r["repair_gain"] > 0 for r in items]))
            summary.append(
                dict(
                    method=name,
                    source=source,
                    defect=defect,
                    pairs=len(items),
                    clean_preferred_fraction=preference,
                    repair_favored_fraction=repair,
                    median_repair_cosine=float(np.median([r["repair_cosine"] for r in items])),
                    passes=preference >= limit and repair >= limit,
                )
            )
    write(
        output / "results.json",
        dict(
            plan_sha256=sha(output / "plan.json"),
            history=history,
            rows=rows,
            summary=summary,
            endpoint_replay_exact=True,
            passing_methods=[
                name for name in models if all(r["passes"] for r in summary if r["method"] == name)
            ],
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())
