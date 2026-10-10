"""Boundary-focused time sampling for the conditional transport bridge toy."""

import argparse
import fcntl
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from learned_bridge_toy import endpoint as parent_endpoint
from PIL import Image, ImageDraw
from scipy import ndimage
from transport_bridge_toy import batch as transport_batch

from lineartgen.config import (
    DataConfig,
    DiffusionConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    NoiseMatchingConfig,
    TrainConfig,
    load_config,
    save_config,
)
from lineartgen.diffusion import pyramid
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm

SEED = 2026101088
EVAL_SEED = 2026102088
STEPS = 6000
METHODS = ("boundary",)


def targets():
    a = np.zeros((64, 64, 1), np.float64)
    b = np.zeros_like(a)
    a[12:52, 27:37] = 1
    b[12:52, 27:30] = b[12:52, 34:37] = 1
    da = ndimage.maximum_filter(a, size=(5, 5, 1))
    db = ndimage.maximum_filter(b, size=(5, 5, 1))
    np.testing.assert_array_equal(da, db)
    source = 0.3 * ndimage.gaussian_filter(da, sigma=(1.5, 1.5, 0))
    return tuple(np.asarray(1 - 2 * x, np.float32) for x in (a, b, source))


def config():
    return ExperimentConfig(
        data=DataConfig(crop_size=64),
        diffusion=DiffusionConfig(scale_loss_multiplier=0),
        matching=NoiseMatchingConfig(method="independent"),
        loss=LossConfig(balance_tones=False, identity_weight=0, edge_weight=0, contrast_weight=0),
        model=ModelConfig(
            channels=(4, 10, 20, 40),
            detail_channels=16,
            output_mode="residual",
            source_conditioning=True,
            progress_conditioning=True,
        ),
        train=TrainConfig(
            steps=STEPS,
            batch_size=16,
            accumulate_steps=1,
            learning_rate=5e-5,
            minimum_learning_rate=1e-5,
            warmup_steps=100,
            seed=SEED,
            model_seed=42,
        ),
    )


def initialize(source, strength, key, depth):
    guides, levels = pyramid(source, depth), pyramid(strength, depth)
    keys = jax.random.split(key, depth)
    noisy = tuple(
        s + 0.25 * t * jax.random.normal(k, s.shape)
        for s, t, k in zip(guides, levels, keys, strict=True)
    )
    return noisy, levels, guides


def path_inputs(source, target, strength, key, tau, depth):
    z, levels, guides = initialize(source, strength, key, depth)
    clean = pyramid(jnp.where(strength > 0, target, source), depth)
    images = tuple(
        jnp.where(t > 0, (1 - tau) * x + tau * y, s)
        for x, y, t, s in zip(z, clean, levels, guides, strict=True)
    )
    progress = tuple(jnp.broadcast_to(1 - tau, x.shape) for x in images)
    return dict(images=images, levels=levels, guides=guides, progress=progress, target=clean[0])


def batch(step, depth, count=16):
    return redistribute_time(*transport_batch(step, depth, count))


@jax.jit
def redistribute_time(prepared, labels, old_tau):
    tau = jnp.square(jnp.sin(jnp.pi * old_tau / 2))
    direct = prepared["direct"]
    clean = pyramid(direct["target"], len(direct["images"]))
    bridge = dict(prepared["bridge"])
    bridge["images"] = tuple(
        jnp.where(t > 0, (1 - tau) * x + tau * y, s)
        for x, y, t, s in zip(
            direct["images"], clean, direct["levels"], direct["guides"], strict=True
        )
    )
    bridge["progress"] = tuple(jnp.broadcast_to(1 - tau, x.shape) for x in bridge["images"])
    return dict(direct=direct, bridge=bridge), labels, tau


@nnx.jit
def recursive_velocity(model, images, levels, guides, progress):
    smallest = images[-1]
    insert = jnp.zeros(
        (smallest.shape[0], smallest.shape[1] // 2, smallest.shape[2] // 2, 1), smallest.dtype
    )
    velocities, endpoints = [], []
    for x, t, s, u in zip(
        reversed(images), reversed(levels), reversed(guides), reversed(progress), strict=True
    ):
        _, raw = model(x, t, insert, s, u, return_preclip=True)
        v = jnp.where(t > 0, raw - x, 0)
        y = jnp.where(t > 0, x + u * v, s)
        insert = jnp.clip(y, -1, 1)
        velocities.append(v)
        endpoints.append(y)
    return tuple(reversed(velocities)), tuple(reversed(endpoints))


def endpoint(model, images, levels, guides, progress):
    return recursive_velocity(model, images, levels, guides, progress)[1]


def velocity(model, images, levels, guides, progress):
    return recursive_velocity(model, images, levels, guides, progress)[0]


def velocity_inputs(prepared):
    p = dict(prepared["bridge"])
    p["velocity_target"] = p["target"] - prepared["direct"]["images"][0]
    return p


def objective(model, prepared):
    prediction = velocity(
        model, prepared["images"], prepared["levels"], prepared["guides"], prepared["progress"]
    )[0]
    active = prepared["levels"][0] > 0
    numerator = (jnp.square(prediction - prepared["velocity_target"]) * active).sum((1, 2, 3))
    loss = (numerator / jnp.maximum(active.sum((1, 2, 3)), 1)).mean()
    return loss, dict(loss=loss, velocity_mse=loss)


def make_arms(root):
    initial, _ = load_inference(root / "initial")
    arms = {m: create_arm(load_config(root / f"configs/{m}.json"), initial) for m in METHODS}
    for arm in arms.values():
        assert_initial_parameters(initial, arm["model"])

        @nnx.jit
        def gradient(model, prepared):
            (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared)
            return metrics, grads

        arm["gradient"] = gradient
    return arms


def sample(predict, source, strength, key, *, depth, steps):
    """Teacher-free sampler. predict receives only evolving states and observations."""
    if steps < 1:
        raise ValueError("steps must be positive")
    images, levels, guides = initialize(source, strength, key, depth)
    for step in range(steps):
        remaining = 1 - step / steps
        progress = tuple(jnp.full_like(x, remaining) for x in images)
        estimates = predict(images, levels, guides, progress)
        images = tuple(
            jnp.where(t > 0, x + y / steps, s)
            for x, y, t, s in zip(images, estimates, levels, guides, strict=True)
        )
    return images[0]


def measure(prediction, a, b, source, strength):
    active = np.asarray(strength) > 0
    a, b = np.where(active, a, source), np.where(active, b, source)
    mean, half = (a + b) / 2, (a - b) / 2
    denom = float(np.sum(half[0].astype(np.float64) ** 2))
    assert denom > 0
    errors = [np.sum((prediction.astype(np.float64) - y) ** 2, axis=(1, 2, 3)) for y in (a, b)]
    nearest = np.sqrt(np.minimum(*errors) / denom)
    score = np.sum((prediction - mean) * half, axis=(1, 2, 3)) / denom
    outside = np.abs(prediction - source)[~active]
    return dict(
        mean_relative_endpoint_rms=float(nearest.mean()),
        max_relative_endpoint_rms=float(nearest.max()),
        fraction_within_020=float(np.mean(nearest <= 0.2)),
        nearest_A=int(np.sum(errors[0] < errors[1])),
        resolved_A=int(np.sum(score > 0.9)),
        resolved_B=int(np.sum(score < -0.9)),
        unresolved=int(np.sum(np.abs(score) <= 0.9)),
        teacher_averaged_mse=float(
            np.mean((errors[0] + errors[1]) / 2) / np.prod(prediction.shape[1:])
        ),
        protected_max_error=float(outside.max()) if outside.size else 0.0,
        finite=bool(np.isfinite(prediction).all()),
    )


def prepare(root):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    c = config()
    parent = Path(
        "../jax-transport-bridge-toy-20261010/runs/transport-bridge-toy-20261010"
    ).resolve()
    verify(parent)
    model, old_config = load_inference(parent / "initial")
    assert c == old_config
    initial = root / "initial"
    save_inference(
        model, c, initial, provenance="AW common random initialization; boundary time toy only"
    )
    assert read(initial / "metadata.json")["parameters"] == 130628
    for m in METHODS:
        save_config(c, root / f"configs/{m}.json")
    a, b, s = targets()
    np.savez_compressed(root / "inputs.npz", a=a, b=b, source=s)
    write(root / "train.json", dict(teachers=2, same_source=True, seed=SEED))
    write(root / "probe.json", dict(seed=EVAL_SEED, new_draws_only=True, new_shapes=False))
    context = Path(
        "../jax-transport-bridge-toy-20261010/runs/transport-bridge-toy-20261010"
    ).resolve()
    assert read(context / "final-audit.json")["stage"] == "complete_learned_toy_diagnostic"
    write(
        root / "plan.json",
        dict(
            arms=list(METHODS),
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir()},
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            steps=STEPS,
            seed=SEED,
            evaluation_seed=EVAL_SEED,
            depths=[1, 2, 3],
            adoption=False,
            objective="AW conditional OT velocity MSE; sin-squared time distribution",
            control_parent=str(parent),
            control_source_sha256=sha(parent / "source.json"),
            control_metrics_sha256=sha(parent / "evaluation/metrics.json"),
            plan_document_sha256=sha(Path("docs/boundary-time-bridge-toy-plan-20261010.md")),
            context=str(context),
            context_final_sha256=sha(context / "final-audit.json"),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert (
        sha(Path("docs/boundary-time-bridge-toy-plan-20261010.md")) == plan["plan_document_sha256"]
    )
    assert sha(Path(plan["context"]) / "final-audit.json") == plan["context_final_sha256"]
    parent = Path(plan["control_parent"])
    verify(parent)
    assert sha(parent / "source.json") == plan["control_source_sha256"]
    assert sha(parent / "evaluation/metrics.json") == plan["control_metrics_sha256"]
    return plan


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        if start == 0:
            snapshot(root, arms, 0)
            initial, _ = load_inference(root / "initial")
            errors = []
            for depth in (1, 2, 3):
                prepared, _, _ = batch(0, depth)
                arguments = {
                    k: prepared["direct"][k] for k in ("images", "levels", "guides", "progress")
                }
                expected = parent_endpoint(initial, **arguments)
                actual = endpoint(arms["boundary"]["model"], **arguments)
                error = max(
                    float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                    for a, b in zip(actual, expected, strict=True)
                )
                assert error <= 1e-5
                errors.append(dict(depth=depth, maximum_error=error))
            write(
                root / "initial-audit.json",
                dict(parameters_equal=True, outputs_within_tolerance=True, errors=errors),
            )
        for step in range(start, STEPS):
            depth = step % 3 + 1
            prepared, labels, tau = batch(step, depth)
            order = METHODS if step % 2 == 0 else METHODS[::-1]
            stats = {m: update(arms[m], velocity_inputs(prepared)) for m in order}
            complete = step + 1
            if complete <= 6 or complete % 100 == 0:
                row = dict(
                    step=complete,
                    depth=depth,
                    metrics=stats,
                    label_A=int(np.sum(np.asarray(labels) == 0)),
                    tau_mean=float(tau.mean()),
                )
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete in (6, 2000, STEPS):
                lineart_state.save(saved, arms, complete)
                restored = make_arms(root)
                assert lineart_state.restore(saved, restored, complete) == complete
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, complete),
                    lineart_state.payload(restored, complete),
                )
                write(out / f"restore-audit-{complete}.json", dict(all_state_exact=True))
                snapshot(root, arms, complete)
        bindings = {}
        for step in (6, 2000, STEPS):
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, step) == step
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.clone(arm["model"])
                    if mode == "ema":
                        nnx.update(expected, arm["ema"])
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    actual, c = load_inference(folder)
                    assert c == arm["config"]
                    jax.tree.map(
                        np.testing.assert_array_equal,
                        nnx.state(expected, nnx.Param),
                        nnx.state(actual, nnx.Param),
                    )
                    bindings.update({str(p): sha(p) for p in folder.iterdir()})
        write(root / "snapshot-audit.json", dict(exports=6, raw_ema_exact=True, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def evaluate(root):
    plan = check(root)
    parent = Path(plan["control_parent"])
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    models = {m: load_inference(root / f"snapshots/step-{STEPS}/{m}/raw")[0] for m in METHODS}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    a, b, s = targets()
    old = read(parent / "evaluation/metrics.json")
    rows = list(old["rows"])
    displays = {}
    bindings = dict(old["bindings"])
    for path, digest in bindings.items():
        assert sha(Path(path)) == digest
    for depth in (1, 2, 3):
        for partial in (False, True):
            for strength in (0.1, 0.5):
                source = jnp.broadcast_to(jnp.asarray(s), (64, *s.shape))
                t = jnp.full_like(source, strength)
                if partial:
                    t = t.at[:, :, :32].set(0)
                key = jax.random.fold_in(
                    jax.random.PRNGKey(EVAL_SEED),
                    depth * 4 + int(partial) * 2 + int(strength == 0.5),
                )
                group = f"d{depth}-p{int(partial)}-t{strength}"
                displays[group] = {
                    "A": np.where(np.asarray(t[0]) > 0, a, s),
                    "B": np.where(np.asarray(t[0]) > 0, b, s),
                    "rough": s,
                }
                for label, method, n in (
                    ("direct1", "direct", 1),
                    ("endpoint32", "bridge", 32),
                    ("velocity32", "velocity", 32),
                    ("transport32", "transport", 32),
                ):
                    filename = f"{group}-{method}-{n}.npz"
                    paths = [Path(p) for p in old["bindings"] if Path(p).name == filename]
                    assert len(paths) == 1
                    with np.load(paths[0]) as saved:
                        displays[group][label] = saved["prediction"][0]
                for method, model in models.items():
                    for steps in (1, 8, 32):
                        physical = np.asarray(
                            sample(
                                lambda x, t, s, u, m=model: velocity(m, x, t, s, u),
                                source,
                                t,
                                key,
                                depth=depth,
                                steps=steps,
                            )
                        )
                        prediction = np.clip(physical, -1, 1)
                        metric = measure(prediction, a, b, np.asarray(source), np.asarray(t))
                        assert metric["finite"] and np.isfinite(physical).all()
                        assert metric["protected_max_error"] == 0
                        metric.update(
                            unclipped_minimum=float(physical.min()),
                            unclipped_maximum=float(physical.max()),
                        )
                        rows.append(
                            dict(
                                method=method,
                                depth=depth,
                                partial=partial,
                                strength=strength,
                                steps=steps,
                                **metric,
                            )
                        )
                        path = out / f"predictions/{group}-{method}-{steps}.npz"
                        np.savez_compressed(path, prediction=prediction, physical=physical)
                        bindings[str(path)] = sha(path)
                        if steps == 1:
                            displays[group][method + "1"] = prediction[0]
                        if method == "boundary" and steps == 32:
                            for i in range(4):
                                displays[group][f"boundary32[{i}]"] = prediction[i]
                write(out / "progress.json", dict(groups=len(displays), total=12, rows=len(rows)))
    assert len(rows) == 180
    primary = [r for r in rows if r["method"] == "boundary" and r["steps"] == 32]
    checks = []
    for r in primary:
        for name, passed in dict(
            near_modes=r["mean_relative_endpoint_rms"] <= 0.15,
            mostly_resolved=r["fraction_within_020"] >= 0.8,
            both_modes=0.1 <= r["nearest_A"] / 64 <= 0.9,
        ).items():
            checks.append(
                dict(
                    depth=r["depth"],
                    partial=r["partial"],
                    strength=r["strength"],
                    guard=name,
                    passed=passed,
                )
            )
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            all_passed=all(c["passed"] for c in checks),
            finite=True,
            protected_exact=True,
            samples_per_group=64,
            adoption=False,
            bindings=bindings,
        ),
    )
    records = []
    for depth in (1, 2, 3):
        columns = (
            "A",
            "B",
            "rough",
            "direct1",
            "endpoint32",
            "velocity32",
            "transport32",
            "boundary1",
            *[f"boundary32[{i}]" for i in range(4)],
        )
        canvas = Image.new("L", (96 * len(columns), 116 * 4), 255)
        draw = ImageDraw.Draw(canvas)
        for row, (partial, strength) in enumerate(
            (p, t) for p in (False, True) for t in (0.1, 0.5)
        ):
            group = f"d{depth}-p{int(partial)}-t{strength}"
            draw.text((2, row * 116 + 2), group, fill=0)
            for col, label in enumerate(columns):
                pixels = np.rint(
                    np.clip((displays[group][label][..., 0] + 1) * 127.5, 0, 255)
                ).astype(np.uint8)
                draw.text((col * 96 + 2, row * 116 + 18), label, fill=0)
                canvas.paste(Image.fromarray(pixels), (col * 96 + 12, row * 116 + 40))
        path = out / f"depth-{depth}-native.png"
        canvas.save(path)
        records.append(dict(file=path.name, sha256=sha(path)))
    write(out / "render-manifest.json", dict(records=records, first_four_fixed=True))
    write(
        root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=STEPS)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "train", "evaluate"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="train6000"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        globals()[args.action](root)
