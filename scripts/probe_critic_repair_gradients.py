"""Check whether frozen critic gradients locally point toward known defect repair."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from flax import nnx
from train_critic_defect_pilot import dataset, resume_states, spatial_logits

from lineartgen.training.lineart_adversarial import PatchCritic


def repair_gradient(mode):
    @nnx.jit
    def measure(model, real, fake):
        def objective(images):
            logits = spatial_logits(model, images)
            losses = (
                jax.nn.softplus(-logits.mean((1, 2, 3)))
                if mode == "scalar"
                else jax.nn.softplus(-logits).mean((1, 2, 3))
            )
            return losses.sum()

        gradient = jax.grad(objective)(fake)[..., 2]
        repair = real[..., 2] - fake[..., 2]
        gain = -(gradient * repair).sum((1, 2))
        cosine = gain / jnp.maximum(
            jnp.linalg.norm(gradient, axis=(1, 2)) * jnp.linalg.norm(repair, axis=(1, 2)), 1e-12
        )
        return gain, cosine

    return measure


def run(parent, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    previous = read(parent / "plan.json")
    plan = dict(
        variants=list(range(112, 120)),
        seed=2026101013,
        milder_defects=True,
        parent=str(parent),
        parent_plan_sha256=sha(parent / "plan.json"),
        parent_results_sha256=sha(parent / "results.json"),
        checkpoint_files={
            str(p.relative_to(parent)): sha(p)
            for p in sorted((parent / "checkpoint").rglob("*"))
            if p.is_file()
        },
        source_files={
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("scripts/train_critic_defect_pilot.py"),
                Path("src/lineartgen/training/lineart_adversarial.py"),
            ]
        },
        meaning="Positive gain: an infinitesimal move of defect pixels toward teacher reduces "
        "the generator's non-saturating critic objective. This is an image-space directional "
        "derivative, not a generator-parameter update or a proof of improved output quality.",
    )
    write(output / "plan.json", plan)
    models = {name: PatchCritic(rngs=nnx.Rngs(0)) for name in previous["methods"]}
    optimizers = {
        name: nnx.Optimizer(model, optax.adam(previous["learning_rate"]), wrt=nnx.Param)
        for name, model in models.items()
    }
    resume_states(parent / "checkpoint", models, optimizers, previous["steps"])
    values, records = dataset(plan["variants"], plan["seed"], True)
    real, fake, _ = values
    rows, summary = [], []
    for name, model in models.items():
        measure = repair_gradient(name)
        gains, cosines = [], []
        for start in range(0, len(real), 64):
            gain, cosine = measure(model, real[start : start + 64], fake[start : start + 64])
            gains.extend(np.asarray(gain).tolist())
            cosines.extend(np.asarray(cosine).tolist())
        grouped = defaultdict(list)
        for record, gain, cosine in zip(records, gains, cosines, strict=True):
            assert np.isfinite(gain) and np.isfinite(cosine)
            row = record | dict(method=name, repair_gain=gain, repair_cosine=cosine)
            rows.append(row)
            grouped[record["source"], record["defect"]].append(row)
        for (source, defect), items in sorted(grouped.items()):
            summary.append(
                dict(
                    method=name,
                    source=source,
                    defect=defect,
                    pairs=len(items),
                    repair_favored_fraction=float(np.mean([r["repair_gain"] > 0 for r in items])),
                    mean_gain=float(np.mean([r["repair_gain"] for r in items])),
                    median_cosine=float(np.median([r["repair_cosine"] for r in items])),
                )
            )
    write(
        output / "results.json",
        dict(
            plan_sha256=sha(output / "plan.json"),
            rows=rows,
            summary=summary,
            parameters_modified=False,
        ),
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())
