"""Toy two-target ambiguity: compare a crisp mode to the arithmetic mean.

Images here are constructed alternatives, not outputs from a trained generator.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from flax import nnx
from PIL import Image, ImageDraw
from train_critic_defect_pilot import resume_states

from lineartgen.evaluation import line_case
from lineartgen.training.feature_matching import feature_distance
from lineartgen.training.lineart_adversarial import PatchCritic


def losses(critic, prediction, target, condition, *, metric, layers):
    difference = prediction - target
    if metric == "pixel_l1":
        return jnp.abs(difference).mean((1, 2, 3))
    if metric == "pixel_l2":
        return jnp.square(difference).mean((1, 2, 3))
    if metric == "pixels":
        return (jnp.sqrt(jnp.square(difference) + 1e-6) - 0.001).mean((1, 2, 3))
    return feature_distance(
        critic,
        prediction,
        target,
        condition,
        jnp.ones_like(target, bool),
        metric=metric,
        layers=layers,
    )


def make_comparison(metric, layers):
    @nnx.jit
    def compare(model, a, b, condition):
        def distance(p, y):
            return losses(model, p, y, condition, metric=metric, layers=layers)

        mean = (a + b) / 2
        return (distance(a, a) + distance(a, b)) / 2, (distance(mean, a) + distance(mean, b)) / 2

    return compare


def run(parent, feature_probe, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    definitions = read(feature_probe / "plan.json")["definitions"] | {
        name: dict(critic="random", metric=name, layers=[0, 1, 2])
        for name in ["pixel_l1", "pixel_l2"]
    }
    plan = dict(
        kinds=["straight", "thin", "curve", "parallel", "taper"],
        variants=list(range(144, 152)),
        vertical_displacements=[1, 2, 4],
        strengths=[0.1, 0.5],
        seed=2026101313,
        definitions=definitions,
        interpretation="Equal-probability clean alternatives A and B share an input. Compare "
        "expected loss of output A with output (A+B)/2. Positive relative excess for the mean "
        "favors a sharp mode; use >1% as a descriptive margin, not a deployment gate.",
        limitation="Toy ambiguity, not trained generator outputs or proof of general perceptual "
        "quality. Some small displacements can represent valid subpixel antialiasing.",
        degeneracy="Exclude only bit-identical target alternatives, independently of all losses. "
        "A translation along an unbroken straight line can leave the central crop unchanged.",
        parent=str(parent),
        parent_results_sha256=sha(parent / "results.json"),
        feature_probe_plan_sha256=sha(feature_probe / "plan.json"),
        source_files={
            str(p): sha(p)
            for p in [Path(__file__), Path("src/lineartgen/training/feature_matching.py")]
        },
    )
    write(output / "plan.json", plan)
    previous = read(parent / "plan.json")
    models = {name: PatchCritic(rngs=nnx.Rngs(0)) for name in previous["methods"]}
    optimizers = {
        name: nnx.Optimizer(model, optax.adam(0.0001), wrt=nnx.Param)
        for name, model in models.items()
    }
    resume_states(parent / "checkpoint", models, optimizers, previous["steps"])
    models["random"] = PatchCritic(rngs=nnx.Rngs(5343))
    targets_a, targets_b, conditions, records, panels, excluded = [], [], [], [], [], []
    for kind in plan["kinds"]:
        for variant in plan["variants"]:
            gray, _, _ = line_case(kind, variant)
            image = np.clip((gray * 2 - 1) * 1.1, -1, 1)
            for offset in plan["vertical_displacements"]:
                a = image[224 - offset : 288 - offset, 224:288]
                b = image[224 + offset : 288 + offset, 224:288]
                if np.array_equal(a, b):
                    excluded.append(
                        dict(
                            kind=kind,
                            variant=variant,
                            displacement=offset,
                            reason="identical target crops",
                        )
                    )
                    continue
                if variant == 144 and offset == 2:
                    panels.append((kind, a, b, (a + b) / 2))
                for strength in plan["strengths"]:
                    random = np.random.default_rng([plan["seed"], len(records)])
                    noisy = (a + b) / 2 * (1 - strength) + random.normal(size=a.shape) * strength
                    condition = np.concatenate(
                        (np.clip(noisy, -3, 3) / 3, np.full_like(a, strength)), -1
                    )
                    targets_a.append(a)
                    targets_b.append(b)
                    conditions.append(condition)
                    records.append(
                        dict(kind=kind, variant=variant, displacement=offset, strength=strength)
                    )
    write(output / "excluded-identical-targets.json", excluded)
    arrays = [np.asarray(x, np.float32) for x in [targets_a, targets_b, conditions]]
    rows, summary = [], []
    for name, settings in definitions.items():
        compare = make_comparison(settings["metric"], tuple(settings["layers"]))
        grouped = defaultdict(list)
        for start in range(0, len(records), 64):
            mode, mean = compare(
                models[settings["critic"]], *(x[start : start + 64] for x in arrays)
            )
            for record, a, b in zip(
                records[start : start + 64], np.asarray(mode), np.asarray(mean), strict=True
            ):
                assert np.isfinite(a) and np.isfinite(b) and a > 0, (name, record, a, b)
                excess = float((b - a) / a)
                if name == "pixel_l2":
                    np.testing.assert_allclose(excess, -0.5, atol=1e-6)
                if name == "pixel_l1":
                    np.testing.assert_allclose(excess, 0, atol=1e-6)
                row = record | dict(
                    method=name, mode_loss=float(a), mean_loss=float(b), relative_mean_excess=excess
                )
                rows.append(row)
                grouped[record["displacement"]].append(excess)
        for offset, values in sorted(grouped.items()):
            summary.append(
                dict(
                    method=name,
                    displacement=offset,
                    cases=len(values),
                    mode_preferred_over_one_percent=float((np.asarray(values) > 0.01).mean()),
                    median_relative_mean_excess=float(np.median(values)),
                )
            )
    write(
        output / "results.json",
        dict(
            plan_sha256=sha(output / "plan.json"),
            rows=rows,
            summary=summary,
            analytic_pixel_controls_verified=True,
        ),
    )
    canvas = Image.new("L", (384, len(panels) * 152), 255)
    for y, (kind, *images) in enumerate(panels):
        for x, (label, array) in enumerate(
            zip(["mode A", "mode B", "mean (constructed)"], images, strict=True)
        ):
            pixels = np.rint((array[..., 0] + 1) * 127.5).astype(np.uint8)
            canvas.paste(
                Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST),
                (x * 128, y * 152 + 24),
            )
            ImageDraw.Draw(canvas).text((x * 128 + 3, y * 152 + 4), kind + ": " + label, fill=0)
    canvas.save(output / "toy-mixtures.png")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--feature-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.feature_probe.resolve(), args.output.resolve())
