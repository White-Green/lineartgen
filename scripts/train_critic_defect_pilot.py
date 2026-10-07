"""Bounded critic-only pilot: artificial negatives and spatial/local logistic loss.

All variants start from the same old 10k critic. The generator is never updated.
Evaluation uses new geometry variants and milder perturbations, after 1000 updates.
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
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from lineartgen.config import SketchAugmentationConfig, load_config
from lineartgen.evaluation import KINDS, line_case
from lineartgen.models.serialization import load_inference
from lineartgen.sketch import augment_sketch
from lineartgen.training import lineart_state
from lineartgen.training.lineart_study import create_arm, study_methods


def altered(target, *, evaluation):
    ink = (1 - target[..., 0]) / 2
    values = dict(
        halo=np.maximum(
            ink,
            (0.3 if evaluation else 0.5) * ndimage.gaussian_filter(ink, 1.0 if evaluation else 1.5),
        ),
        weak=ink * (0.8 if evaluation else 0.6),
    )
    points = np.argwhere(ink > 0.5)
    center = points[np.argmin(np.square(points - 32).sum(1))]
    yy, xx = np.indices(ink.shape)
    cut = ink.copy()
    cut[(yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= (4 if evaluation else 9) ** 2] = 0
    values["cut"] = cut
    labels, count = ndimage.label(ink > 0.5, np.ones((3, 3)))
    if count == 2:
        left, right = np.argwhere(labels == 1), np.argwhere(labels == 2)
        distances, indices = cKDTree(right).query(left)
        index = int(np.argmin(distances))
        if 1 < distances[index] <= 40:
            a, b = left[index], right[indices[index]]
            drawing = Image.new("L", (256, 256), 0)
            ImageDraw.Draw(drawing).line(
                [(float(a[1] * 4), float(a[0] * 4)), (float(b[1] * 4), float(b[0] * 4))],
                fill=255,
                width=4 if evaluation else 12,
            )
            connector = np.asarray(drawing.resize((64, 64), Image.Resampling.LANCZOS)) / 255
            values["bridge"] = np.maximum(ink, connector)
    return {name: (1 - 2 * value[..., None]).astype(np.float32) for name, value in values.items()}


def dataset(variants, seed, evaluation):
    sketch = SketchAugmentationConfig(
        mode="strokes",
        probability=1.0,
        opacity_min=0.2,
        opacity_max=0.45,
        blur_sigma_min=1.5,
        blur_sigma_max=2.5,
        dilation_radius=2,
        displacement=3.0,
    )
    augment = jax.jit(lambda x, key: augment_sketch(x, key, sketch))
    real, fake, weights, records = [], [], [], []
    for kind in KINDS:
        for variant in variants:
            gray, _, _ = line_case(kind, variant)
            target = np.clip((gray[224:288, 224:288] * 2 - 1) * 1.1, -1, 1)
            rough = np.asarray(augment(target[None], jax.random.PRNGKey(seed + variant)))[0]
            defects = altered(target, evaluation=evaluation)
            if kind not in ("gap", "parallel"):
                defects.pop("bridge", None)
            for family, source in [("clean", target), ("strokes_strong", rough)]:
                for strength in [0.1, 0.5]:
                    rng = np.random.default_rng(
                        [seed, list(KINDS).index(kind), variant, int(strength * 100)]
                    )
                    noisy = source * (1 - strength) + rng.normal(size=source.shape) * strength
                    condition = np.concatenate(
                        (np.clip(noisy, -3, 3) / 3, np.full_like(target, strength)), -1
                    )
                    for defect, value in defects.items():
                        real.append(np.concatenate((condition, target), -1))
                        fake.append(np.concatenate((condition, value), -1))
                        changed = np.abs(value[..., 0] - target[..., 0]) > 0.005
                        # Three SAME stride-2 convolutions have centers 7 + 8*i;
                        # with the stride-1 head, each logit sees a 31px window.
                        local = ndimage.maximum_filter(changed, size=31, mode="constant")
                        weights.append(local[7::8, 7::8, None])
                        records.append(
                            dict(
                                kind=kind,
                                variant=variant,
                                source=family,
                                strength=strength,
                                defect=defect,
                            )
                        )
    return tuple(np.asarray(x, np.float32) for x in (real, fake, weights)), records


def spatial_logits(model, images):
    for layer in model.layers:
        images = jax.nn.leaky_relu(layer(images), negative_slope=0.2)
    return model.head(images)


def make_step(mode):
    def objective(model, real, fake, weights):
        a, b = spatial_logits(model, real), spatial_logits(model, fake)
        if mode == "scalar":
            return (jax.nn.softplus(-a.mean((1, 2, 3))) + jax.nn.softplus(b.mean((1, 2, 3)))).mean()
        if mode == "spatial":
            return (jax.nn.softplus(-a) + jax.nn.softplus(b)).mean()
        return (
            jax.nn.softplus(-a).mean()
            + (
                (jax.nn.softplus(b) * weights).sum((1, 2, 3))
                / jnp.maximum(weights.sum((1, 2, 3)), 1)
            ).mean()
        )

    @nnx.jit
    def step(model, optimizer, real, fake, weights):
        loss, gradient = nnx.value_and_grad(objective)(model, real, fake, weights)
        optimizer.update(model, gradient)
        return loss, optax.global_norm(gradient)

    return step


def run(parent, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = dict(
        steps=1000,
        batch_size=32,
        learning_rate=0.0001,
        train_variants=list(range(64, 80)),
        evaluation_variants=list(range(96, 104)),
        train_seed=2026100813,
        evaluation_seed=2026100913,
        methods=["scalar", "spatial", "localized"],
        initial="Same 10k critic from previous study",
        generator_updates=0,
        negatives="Equal sampling across halo, weak, cut, bridge; pure synthetic geometry",
        evaluation="Once after 1000 updates. New variants and milder defects; "
        "no best-step selection",
        next_phase_gate="Diagnostic only. Require >=80% clean-preference for cuts and bridges "
        "in both source conditions, and >=90% for halo, before considering a generator pilot.",
        limitations="No natural sketches or actual generator outputs in this pilot. Geometry "
        "comes from the same generator. Passing alone does not justify deployment.",
        source_sha256=sha(Path(__file__)),
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
        name: nnx.Optimizer(model, optax.adam(plan["learning_rate"]), wrt=nnx.Param)
        for name, model in models.items()
    }
    steps = {name: make_step(name) for name in models}
    train, records = dataset(plan["train_variants"], plan["train_seed"], False)
    groups = {
        name: [i for i, r in enumerate(records) if r["defect"] == name]
        for name in ["halo", "weak", "cut", "bridge"]
    }
    random = np.random.default_rng(plan["train_seed"])
    history = []
    for step in range(plan["steps"]):
        indices = np.concatenate(
            [random.choice(indices, plan["batch_size"] // 4) for indices in groups.values()]
        )
        batch = tuple(jax.device_put(x[indices]) for x in train)
        metrics = {}
        for name in models:
            loss, norm = steps[name](models[name], optimizers[name], *batch)
            assert np.isfinite(float(loss)) and np.isfinite(float(norm))
            metrics[name] = dict(loss=float(loss), gradient_norm=float(norm))
        if (step + 1) % 100 == 0:
            row = dict(step=step + 1, methods=metrics)
            history.append(row)
            write(output / "progress.json", row)
            print(row, flush=True)
    evaluation, metadata = dataset(plan["evaluation_variants"], plan["evaluation_seed"], True)
    predict = nnx.jit(lambda m, x: m(x))
    rows, summary = [], []
    for name, model in {"original": original, **models}.items():
        values = []
        for inputs in evaluation[:2]:
            values.append(
                np.concatenate(
                    [
                        np.asarray(predict(model, inputs[i : i + 64]))
                        for i in range(0, len(inputs), 64)
                    ]
                )
            )
        margins = values[0] - values[1]
        grouped = defaultdict(list)
        for record, margin in zip(metadata, margins, strict=True):
            rows.append(record | dict(method=name, margin=float(margin)))
            grouped[record["source"], record["defect"]].append(float(margin))
        for (source, defect), group in sorted(grouped.items()):
            summary.append(
                dict(
                    method=name,
                    source=source,
                    defect=defect,
                    pairs=len(group),
                    clean_preferred_fraction=float((np.asarray(group) > 0).mean()),
                    mean_margin=float(np.mean(group)),
                )
            )
    state = {
        name: dict(
            model=nnx.to_pure_dict(nnx.state(model)),
            optimizer=nnx.to_pure_dict(nnx.state(optimizers[name])),
        )
        for name, model in models.items()
    }
    with ocp.StandardCheckpointer() as saver:
        saver.save((output / "checkpoint").resolve(), state)
        saver.wait_until_finished()
    write(
        output / "results.json",
        dict(
            plan_sha256=sha(output / "plan.json"),
            history=history,
            summary=summary,
            rows=rows,
            parent_checkpoint_archive=archive,
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())
