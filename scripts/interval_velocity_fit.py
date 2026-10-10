"""Interval velocity supervision from the mode-balanced30000-step model."""

import argparse
import fcntl
from pathlib import Path

import boundary_time_bridge_toy as boundary_parent
import bridge_duration as duration_parent
import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from PIL import Image, ImageDraw
from velocity_bridge_toy import (
    EVAL_SEED,
    SEED,
    measure,
    objective,
    sample,
    targets,
    velocity,
    velocity_inputs,
)

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm

START = 30000
STEPS = 36000
METHODS = ("mse", "interval")
SAVED = (START, START + 6, 32000, STEPS)


def batch_for(method, step, depth, count=16):
    assert method in METHODS
    return boundary_parent.batch(step, depth, count)


@nnx.jit
def gradient(model, prepared):
    (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared)
    return metrics, grads


def interval_error(prediction, target_velocity, clean):
    error = prediction - target_velocity
    return jnp.where(
        clean >= 1, jnp.minimum(error, 0), jnp.where(clean <= -1, jnp.maximum(error, 0), error)
    )


def interval_objective(model, prepared):
    prediction = velocity(
        model, prepared["images"], prepared["levels"], prepared["guides"], prepared["progress"]
    )[0]
    error = interval_error(prediction, prepared["velocity_target"], prepared["target"])
    active = prepared["levels"][0] > 0
    per_image = (jnp.square(error) * active).sum((1, 2, 3)) / jnp.maximum(active.sum((1, 2, 3)), 1)
    loss = per_image.mean()
    return loss, dict(loss=loss, interval_velocity_mse=loss)


@nnx.jit
def interval_gradient(model, prepared):
    (_, metrics), grads = nnx.value_and_grad(interval_objective, has_aux=True)(model, prepared)
    return metrics, grads


def split_arms(reference, configs):
    arms = {}
    for name in METHODS:
        arm = create_arm(configs[name], reference["model"])
        nnx.update(arm["model"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["model"])))
        nnx.update(
            arm["optimizer"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["optimizer"]))
        )
        arm["ema"] = jax.tree.map(lambda x: x.copy(), reference["ema"])
        arm["gradient"] = gradient if name == "mse" else interval_gradient
        arms[name] = arm
    return arms


def load_parent(root):
    plan = read(root / "plan.json")
    p = Path(plan["parents"]["mse"]["root"])
    arms = duration_parent.make_arms(p)
    with checkpoints.manager(p / "training/checkpoints", keep=4) as saved:
        assert lineart_state.restore(saved, arms, plan["start_step"]) == plan["start_step"]
    return arms["boundary_long"]


def make_arms(root):
    plan = read(root / "plan.json")
    reference = load_parent(root)
    arms = split_arms(
        reference, {name: load_config(root / f"configs/{name}.json") for name in METHODS}
    )
    for name, arm in arms.items():
        initial, _ = load_inference(root / "initial" / name)
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(arm["model"], nnx.Param),
            nnx.state(initial, nnx.Param),
        )
        assert int(arm["optimizer"].step[...]) == plan["start_step"]
    return arms


def prepare(root):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    p = Path("../jax-bridge-duration-20261010/runs/bridge-duration-20261010").resolve()
    verify(p)
    assert read(p / "final-audit.json")["stage"] == "complete_learned_toy_diagnostic"
    model, old = load_inference(p / f"snapshots/step-{START}/boundary_long/raw")
    c = old.model_copy(update={"train": old.train.model_copy(update={"steps": STEPS})})
    assert c.train.learning_rate_decay_steps == 6000
    parents = {}
    for name in METHODS:
        save_config(c, root / f"configs/{name}.json")
        save_inference(
            model,
            c,
            root / "initial" / name,
            provenance="Actual AY boundary30000; paired interval loss diagnostic",
        )
        parents[name] = dict(
            root=str(p),
            method="boundary_long",
            source_sha256=sha(p / "source.json"),
            final_sha256=sha(p / "final-audit.json"),
            checkpoint_files={
                str(f): sha(f)
                for f in (p / f"training/checkpoints/{START}").rglob("*")
                if f.is_file()
            },
        )
    initial = root / "initial"
    a, b, source = targets()
    np.savez_compressed(root / "inputs.npz", a=a, b=b, source=source)
    write(
        root / "train.json",
        dict(teachers=2, same_source=True, seed=SEED, updates_per_arm=STEPS - START),
    )
    write(root / "probe.json", dict(seed=EVAL_SEED, new_draws_only=True, new_shapes=False))
    write(
        root / "plan.json",
        dict(
            arms=list(METHODS),
            parents=parents,
            start_step=START,
            steps=STEPS,
            initial=str(initial),
            initial_files={
                str(f.relative_to(initial)): sha(f) for f in initial.rglob("*") if f.is_file()
            },
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            control_parent=str(p),
            control_metrics_sha256=sha(p / "evaluation/metrics.json"),
            plan_document_sha256=sha(Path("docs/interval-velocity-fit-plan-20261010.md")),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(Path("docs/interval-velocity-fit-plan-20261010.md")) == plan["plan_document_sha256"]
    for entry in plan["parents"].values():
        p = Path(entry["root"])
        verify(p)
        assert sha(p / "source.json") == entry["source_sha256"]
        assert sha(p / "final-audit.json") == entry["final_sha256"]
        for path, digest in entry["checkpoint_files"].items():
            assert sha(Path(path)) == digest
    assert (
        sha(Path(plan["control_parent"]) / "evaluation/metrics.json")
        == plan["control_metrics_sha256"]
    )
    return plan


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=4) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else START
        if saved.latest_step() is None:
            for name, arm in arms.items():
                original = load_parent(root)
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload({name: arm}, START),
                    lineart_state.payload({name: original}, START),
                )
                for n in (0, 100, 5999, 6000, 30000, 36000):
                    assert float(arm["schedule"](n)) == float(original["schedule"](n))
            write(
                root / "initial-audit.json",
                dict(all_parent_state_exact=True, schedules_exact=True, optimizer_step=START),
            )
            lineart_state.save(saved, arms, START)
            snapshot(root, arms, START)
            write(out / f"restore-audit-{START}.json", dict(all_state_exact=True))
        prefix = []
        for step in range(start, STEPS):
            depth = step % 3 + 1
            order = METHODS if step % 2 == 0 else METHODS[::-1]
            stats = {}
            for name in order:
                prepared, _, _ = batch_for(name, step, depth)
                stats[name] = update(arms[name], velocity_inputs(prepared))
            complete = step + 1
            if complete <= START + 6:
                prefix.append(stats)
            if complete <= START + 6 or complete % 100 == 0:
                row = dict(step=complete, depth=depth, metrics=stats)
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete == START + 6:
                replay = make_arms(root)
                for n in range(START, START + 6):
                    values = {
                        name: update(
                            replay[name], velocity_inputs(batch_for(name, n, n % 3 + 1)[0])
                        )
                        for name in METHODS
                    }
                    if start == START:
                        assert values == prefix[n - START]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, complete),
                    lineart_state.payload(replay, complete),
                )
                write(
                    root / "prefix-audit.json",
                    dict(updates=6, all_state_exact=True, metrics_exact=start == START),
                )
            if complete in SAVED:
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
        for step in SAVED:
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
        write(root / "snapshot-audit.json", dict(exports=16, raw_ema_exact=True, bindings=bindings))
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
                    ("boundary6000", "boundary", 32),
                    ("boundary30000", "boundary_long", 32),
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
                        if steps == 32:
                            for i in range(4):
                                displays[group][f"{method}[{i}]"] = prediction[i]
                write(out / "progress.json", dict(groups=len(displays), total=12, rows=len(rows)))
    assert len(rows) == 324
    primary = [r for r in rows if r["method"] in METHODS and r["steps"] == 32]
    checks = []
    for r in primary:
        for name, passed in dict(
            near_modes=r["mean_relative_endpoint_rms"] <= 0.15,
            mostly_resolved=r["fraction_within_020"] >= 0.8,
            both_modes=0.1 <= r["nearest_A"] / 64 <= 0.9,
        ).items():
            checks.append(
                dict(
                    method=r["method"],
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
            "boundary6000",
            "boundary30000",
            *[f"{method}[{i}]" for method in METHODS for i in range(4)],
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
            write(root / "workflow-status.json", dict(stage="train36000"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        globals()[args.action](root)
