"""Check teacher-relative feature losses before considering generator experiments."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from flax import nnx
from train_critic_defect_pilot import dataset, resume_states

from lineartgen.training.feature_matching import feature_distance
from lineartgen.training.lineart_adversarial import PatchCritic


def make_diagnostic(metric, layers):
    @nnx.jit
    def measure(critic, real, fake):
        def distances(prediction):
            target = real[..., 2:]
            if metric == "pixels":
                return (jnp.sqrt(jnp.square(prediction - target) + 1e-6) - 0.001).mean((1, 2, 3))
            return feature_distance(
                critic,
                prediction,
                target,
                fake[..., :2],
                jnp.ones_like(target, bool),
                metric=metric,
                layers=layers,
            )

        values = distances(fake[..., 2:])
        gradient = jax.grad(lambda p: distances(p).sum())(fake[..., 2:])
        repair = real[..., 2:] - fake[..., 2:]
        gain = -(gradient * repair).sum((1, 2, 3))
        cosine = gain / jnp.maximum(
            jnp.sqrt(jnp.square(gradient).sum((1, 2, 3)))
            * jnp.sqrt(jnp.square(repair).sum((1, 2, 3))),
            1e-12,
        )
        return values, gain, cosine

    return measure


def run(parent, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    previous = read(parent / "plan.json")
    definitions = dict(
        pixels=dict(critic="random", metric="pixels", layers=[0, 1, 2]),
        random_all=dict(critic="random", metric="smooth_l1", layers=[0, 1, 2]),
        endpoint_all=dict(critic="endpoint", metric="smooth_l1", layers=[0, 1, 2]),
        endpoint_deep=dict(critic="endpoint", metric="smooth_l1", layers=[2]),
        endpoint_l2=dict(critic="endpoint", metric="l2", layers=[0, 1, 2]),
        wgan_all=dict(critic="wgan_gp", metric="smooth_l1", layers=[0, 1, 2]),
    )
    plan = dict(
        variants=list(range(136, 144)),
        seed=2026101213,
        definitions=definitions,
        parent=str(parent),
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
                Path("src/lineartgen/training/feature_matching.py"),
                Path("scripts/train_critic_defect_pilot.py"),
            ]
        },
        gate="At least 95% favorable repair directions for every defect and source condition. "
        "This rejects obvious gradient pathologies only; pixels/random features are controls, "
        "and passing does not show a quality improvement over either control.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/papers/"
        "Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.pdf",
        limitations="Fixed synthetic defects; image-space gradients only. No generator update. "
        "Different discriminator and smoothed L1 from the reference, not a paper reproduction.",
    )
    write(output / "plan.json", plan)
    models = {name: PatchCritic(rngs=nnx.Rngs(0)) for name in previous["methods"]}
    optimizers = {
        name: nnx.Optimizer(model, optax.adam(0.0001), wrt=nnx.Param)
        for name, model in models.items()
    }
    resume_states(parent / "checkpoint", models, optimizers, previous["steps"])
    models["random"] = PatchCritic(rngs=nnx.Rngs(5343))
    values, records = dataset(plan["variants"], plan["seed"], True)
    rows, summary = [], []
    for name, settings in definitions.items():
        measure = make_diagnostic(settings["metric"], tuple(settings["layers"]))
        grouped = defaultdict(list)
        for start in range(0, len(records), 64):
            loss, gain, cosine = measure(
                models[settings["critic"]],
                values[0][start : start + 64],
                values[1][start : start + 64],
            )
            for record, a, b, c in zip(
                records[start : start + 64],
                np.asarray(loss),
                np.asarray(gain),
                np.asarray(cosine),
                strict=True,
            ):
                assert all(np.isfinite(v) for v in [a, b, c])
                row = record | dict(
                    method=name, loss=float(a), repair_gain=float(b), repair_cosine=float(c)
                )
                rows.append(row)
                grouped[record["source"], record["defect"]].append(row)
        for (source, defect), items in sorted(grouped.items()):
            fraction = float(np.mean([r["repair_gain"] > 0 for r in items]))
            summary.append(
                dict(
                    method=name,
                    source=source,
                    defect=defect,
                    pairs=len(items),
                    repair_favored_fraction=fraction,
                    passes=fraction >= 0.95,
                    median_repair_cosine=float(np.median([r["repair_cosine"] for r in items])),
                )
            )
    write(
        output / "results.json",
        dict(
            plan_sha256=sha(output / "plan.json"),
            rows=rows,
            summary=summary,
            passing_methods=[
                name
                for name in definitions
                if all(r["passes"] for r in summary if r["method"] == name)
            ],
        ),
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())
