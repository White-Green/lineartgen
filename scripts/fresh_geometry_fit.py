"""Fresh-parameter control for512px interval velocity fitting."""

import argparse
import fcntl
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bridge_time_grid import sample_grid
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from geometry_velocity_fit import prepared_batch
from interval_velocity_fit import interval_gradient
from paper_band_fit import PROBE, TRAIN, fit_checks, target_oracle
from PIL import Image, ImageDraw
from velocity_bridge_toy import velocity, velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm

METHODS = ("control", "fresh")
TRAINED = ("fresh",)
STEPS = 6000
SEED = 2026101093
EVAL_SEED = 2026102093
SAVED = (0, 6, 2000, STEPS)


def arrays(root, ids):
    with np.load(root / "inputs.npz") as data:
        return tuple(
            jnp.asarray(np.stack([data[f"{i}-{field}"] for i in ids]))
            for field in ("image", "source", "strength")
        )


def audit_data(root):
    rows = []
    with np.load(root / "inputs.npz") as data:
        for i in TRAIN + PROBE:
            a, b, t = [data[f"{i}-{field}"] for field in ("image", "source", "strength")]
            assert a.shape == b.shape == t.shape == (512, 512, 1)
            assert np.isfinite(a).all() and np.isfinite(b).all() and np.all(t == 0.5)
            rows.append(
                dict(
                    case=i,
                    split="train" if i in TRAIN else "probe",
                    antialias_pixels=int(np.sum((a > -1) & (a < 1))),
                    minimum=float(a.min()),
                    maximum=float(a.max()),
                )
            )
    assert set(TRAIN).isdisjoint(PROBE)
    assert all(r["antialias_pixels"] > 0 for r in rows)
    write(
        root / "data-audit.json",
        dict(
            train=list(TRAIN),
            probe=list(PROBE),
            disjoint=True,
            inputs_sha256=sha(root / "inputs.npz"),
            rows=rows,
        ),
    )


def make_arms(root):
    initial, _ = load_inference(root / "initial")
    arm = create_arm(load_config(root / "configs/fresh.json"), initial)
    assert_initial_parameters(initial, arm["model"])
    arm["gradient"] = interval_gradient
    return {"fresh": arm}


def prepare(root):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    data_parent = Path("../jax-paper-band-fit-20261010/runs/paper-band-fit-20261010").resolve()
    parent = Path(
        "../jax-geometry-velocity-fit-20261010/runs/geometry-velocity-fit-20261010"
    ).resolve()
    verify(data_parent)
    verify(parent)
    control, old = load_inference(parent / "snapshots/step-6000/interval/raw")
    model = create_model(old.model, rngs=nnx.Rngs(42))
    save_inference(control, old, root / "control", provenance="Frozen BC interval6000 control")
    config = old.model_copy(
        update={
            "data": old.data.model_copy(update={"crop_size": 512}),
            "train": old.train.model_copy(
                update={
                    "steps": STEPS,
                    "learning_rate_decay_steps": STEPS,
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "learning_rate": 5e-5,
                    "minimum_learning_rate": 1e-4,
                    "warmup_steps": 0,
                    "seed": SEED,
                }
            ),
        }
    )
    assert config == old
    assert config.train.peak_learning_rate == 1e-4
    save_inference(
        model,
        config,
        root / "initial",
        provenance="Fresh model seed42; same geometry batches and schedule as BC",
    )
    assert read(root / "initial/metadata.json")["parameters"] == 130628
    for method in TRAINED:
        save_config(config, root / f"configs/{method}.json")
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(data_parent / name, root / name)
    target_oracle(root)
    audit_data(root)
    initial = root / "initial"
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir()},
            data={
                "fixed_pairs": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))
            },
            data_parent=str(data_parent),
            data_source_sha256=sha(data_parent / "source.json"),
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            parent_final_sha256=sha(parent / "final-audit.json"),
            control_files={str(p): sha(p) for p in (root / "control").iterdir()},
            control_metrics_sha256=sha(parent / "evaluation/metrics.json"),
            crops=read(data_parent / "plan.json")["crops"],
            target_oracle_sha256=sha(root / "target-oracle.json"),
            steps=STEPS,
            seed=SEED,
            evaluation_seed=EVAL_SEED,
            depths=list(range(1, 7)),
            adoption=False,
            plan_document_sha256=sha(Path("docs/fresh-geometry-fit-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(Path("docs/fresh-geometry-fit-plan-20261011.md")) == plan["plan_document_sha256"]
    for prefix in ("parent", "data_parent"):
        verify(Path(plan[prefix]))
    assert sha(Path(plan["parent"]) / "source.json") == plan["parent_source_sha256"]
    assert sha(Path(plan["parent"]) / "final-audit.json") == plan["parent_final_sha256"]
    assert sha(Path(plan["data_parent"]) / "source.json") == plan["data_source_sha256"]
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    for p, digest in plan["control_files"].items():
        assert sha(Path(p)) == digest
    assert sha(Path(plan["parent"]) / "evaluation/metrics.json") == plan["control_metrics_sha256"]
    return plan


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    data = arrays(root, TRAIN)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=4) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        if saved.latest_step() is None:
            for arm in arms.values():
                assert int(arm["optimizer"].step[...]) == 0
                jax.tree.map(
                    np.testing.assert_array_equal, arm["ema"], nnx.state(arm["model"], nnx.Param)
                )
                for n in (0, 100, 5999):
                    assert abs(float(arm["schedule"](n)) - 1e-4) < 1e-10
            write(
                root / "initial-audit.json",
                dict(
                    initial_parameters_exact=True,
                    optimizer_zero=True,
                    ema_equals_model=True,
                    constant_lr=1e-4,
                    model_seed=42,
                ),
            )
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
            write(out / "restore-audit-0.json", dict(all_state_exact=True))
        prefix = []
        for step in range(start, STEPS):
            depth = step % 6 + 1
            batch, tau = prepared_batch(*data, step, depth)
            p = velocity_inputs(batch)
            order = TRAINED
            stats = {name: update(arms[name], p) for name in order}
            complete = step + 1
            if complete <= 6:
                prefix.append(stats)
            if complete <= 6 or complete % 50 == 0:
                row = dict(step=complete, depth=depth, metrics=stats, tau_mean=float(tau.mean()))
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete == 6:
                replay = make_arms(root)
                for n in range(6):
                    p = velocity_inputs(prepared_batch(*data, n, n % 6 + 1)[0])
                    values = {name: update(replay[name], p) for name in TRAINED}
                    if start == 0:
                        assert values == prefix[n]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, 6),
                    lineart_state.payload(replay, 6),
                )
                write(
                    root / "prefix-audit.json",
                    dict(
                        updates=6,
                        all_state_exact=True,
                        metrics_exact=start == 0,
                        depths=list(range(1, 7)),
                    ),
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
        write(root / "snapshot-audit.json", dict(exports=8, raw_ema_exact=True, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def run_sample(model, source, strength, key, depth):
    return np.asarray(
        sample_grid(
            lambda x, t, s, u: velocity(model, x, t, s, u),
            source,
            strength,
            key,
            depth=depth,
            steps=32,
        )
    )


def evaluate(root):
    plan = check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    models = {
        "fresh": load_inference(root / f"snapshots/step-{STEPS}/fresh/raw")[0],
        "control": load_inference(root / "control")[0],
    }
    initial = load_inference(root / "initial")[0]
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    bindings = {}
    replay = []
    displays = {}
    partial_displays = {}
    with np.load(root / "inputs.npz") as data:
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            clean, source, strength = arrays(root, ids)
            for depth in (1, 3, 6):
                for draw in range(4):
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(EVAL_SEED),
                        depth * 100 + int(split == "probe") * 10 + draw,
                    )
                    active_models = dict(models)
                    if depth == 3 and draw == 0:
                        active_models["initial"] = initial
                    for method, model in active_models.items():
                        physical = run_sample(model, source, strength, key, depth)
                        prediction = np.clip(physical, -1, 1)
                        assert np.isfinite(physical).all()
                        if method == "control":
                            reference = (
                                Path(plan["parent"])
                                / f"evaluation/predictions/{split}-d{depth}-r{draw}-interval.npz"
                            )
                            with np.load(reference) as a:
                                np.testing.assert_array_equal(physical, a["physical"])
                                np.testing.assert_array_equal(prediction, a["prediction"])
                            replay.append(str(reference))
                        path = out / f"predictions/{split}-d{depth}-r{draw}-{method}.npz"
                        np.savez_compressed(path, physical=physical, prediction=prediction)
                        bindings[str(path)] = sha(path)
                        for i, value in zip(ids, prediction, strict=True):
                            target, level = data[f"{i}-image"], data[f"{i}-strength"]
                            metrics = protected_metrics(value, target, level)
                            metrics |= halo_metrics(value, target, level)
                            metrics |= probe_metrics(
                                value, target, data[f"{i}-probes"], data[f"{i}-connections"]
                            )
                            metrics |= coverage_metrics(value, target, level)
                            assert all(v is None or np.isfinite(v) for v in metrics.values())
                            rows.append(
                                dict(
                                    method=method,
                                    case=i,
                                    split=split,
                                    depth=depth,
                                    draw=draw,
                                    **metrics,
                                )
                            )
                            if depth == 3 and method in METHODS:
                                displays[(method, i, draw)] = value
                    write(
                        out / "progress.json",
                        dict(split=split, depth=depth, draw=draw, rows=len(rows), total=200),
                    )
        clean, source, strength = arrays(root, TRAIN)
        strength = strength.at[:, :, :256].set(0)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 9000)
        for method, model in models.items():
            physical = run_sample(model, source, strength, key, 6)
            prediction = np.clip(physical, -1, 1)
            mask = np.asarray(strength) == 0
            assert np.isfinite(physical).all()
            np.testing.assert_array_equal(physical[mask], np.asarray(source)[mask])
            np.testing.assert_array_equal(prediction[mask], np.asarray(source)[mask])
            if method == "control":
                reference = Path(plan["parent"]) / "evaluation/predictions/partial-d6-interval.npz"
                with np.load(reference) as a:
                    np.testing.assert_array_equal(physical, a["physical"])
                    np.testing.assert_array_equal(prediction, a["prediction"])
                replay.append(str(reference))
            path = out / f"predictions/partial-d6-{method}.npz"
            np.savez_compressed(path, physical=physical, prediction=prediction)
            bindings[str(path)] = sha(path)
            for i, value in zip(TRAIN, prediction, strict=True):
                partial_displays[(method, i)] = value
        assert len(replay) == 25
        write(root / "replay-audit.json", dict(exact=True, files=replay))
        checks = []
        for row in rows:
            if row["depth"] == 3 and row["method"] in METHODS:
                for c in fit_checks([row]):
                    checks.append(dict(c, depth=3, draw=row["draw"]))
        guards = []
        for i in TRAIN + PROBE:
            for draw in range(4):
                values = {
                    r["method"]: r
                    for r in rows
                    if r["case"] == i and r["depth"] == 3 and r["draw"] == draw
                }
                for metric, limit, lower in (
                    ("ink_f1_alpha50", 0.005, False),
                    ("ink_deficit_fraction", 0.01, True),
                ):
                    a, b = values["fresh"][metric], values["control"][metric]
                    guards.append(
                        dict(
                            case=i,
                            draw=draw,
                            metric=metric,
                            passed=a <= b + limit if lower else a >= b - limit,
                            fresh=a,
                            control=b,
                        )
                    )
        assert len(rows) == 200 and len(checks) == 176 and len(guards) == 64 and len(bindings) == 52
        write(
            out / "metrics.json",
            dict(
                rows=rows,
                checks=checks,
                retention_checks=guards,
                fit_passed={
                    m: all(c["passed"] for c in checks if c["method"] == m) for m in METHODS
                },
                retention_passed=all(c["passed"] for c in guards),
                finite=True,
                protected_exact=True,
                bindings=bindings,
                adoption=False,
            ),
        )
        records = []
        columns = ("target", "source", *[f"{m}[{n}]" for m in METHODS for n in range(4)])
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            for native in (True, False):
                canvas = Image.new("L", (160 * len(columns), 184 * len(ids)), 255)
                for row, i in enumerate(ids):
                    values = [
                        data[f"{i}-image"],
                        data[f"{i}-source"],
                        *[displays[(m, i, n)] for m in METHODS for n in range(4)],
                    ]
                    for col, (label, value) in enumerate(zip(columns, values, strict=True)):
                        pixels = np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(
                            np.uint8
                        )
                        tile = Image.fromarray(pixels)
                        tile = (
                            tile.crop(plan["crops"][str(i)])
                            if native
                            else tile.resize((160, 160), Image.Resampling.LANCZOS)
                        )
                        canvas.paste(tile, (160 * col, 184 * row + 24))
                        ImageDraw.Draw(canvas).text(
                            (160 * col + 2, 184 * row + 4), f"{i} {label}", fill=0
                        )
                path = out / f"{split}-{'native' if native else 'overview'}.png"
                canvas.save(path)
                records.append(dict(file=path.name, sha256=sha(path)))
        canvas = Image.new("L", (160 * 4, 184 * 4), 255)
        for row, i in enumerate(TRAIN):
            target = np.where(
                np.asarray(strength[row]) > 0, data[f"{i}-image"], data[f"{i}-source"]
            )
            values = [target, data[f"{i}-source"], *[partial_displays[(m, i)] for m in METHODS]]
            for col, (label, value) in enumerate(
                zip(("masked_target", "source", *METHODS), values, strict=True)
            ):
                pixels = np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                canvas.paste(
                    Image.fromarray(pixels).crop(plan["crops"][str(i)]), (160 * col, 184 * row + 24)
                )
                ImageDraw.Draw(canvas).text((160 * col + 2, 184 * row + 4), f"{i} {label}", fill=0)
        path = out / "partial-native.png"
        canvas.save(path)
        records.append(dict(file=path.name, sha256=sha(path)))
        write(
            out / "render-manifest.json",
            dict(records=records, all_four_fixed_draws=True, crops_from_AD=True),
        )
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
        {"prepare": prepare, "evaluate": evaluate}[args.action](root)
