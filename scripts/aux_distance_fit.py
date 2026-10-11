"""Distance-dependent saturation bounds versus point-distance supervision."""

import argparse
import fcntl
import shutil
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, update, verify
from distance_supervision_fit import prepared_batch
from edge_energy_fit import forward, prepared_inputs
from flax import nnx
from paper_band_fit import PROBE, TRAIN, fit_checks, target_oracle
from PIL import Image, ImageDraw

from lineartgen.config import load_config, save_config
from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.auxiliary_distance import AuxiliaryDistance, objective
from lineartgen.training.steps import create_optimizer, make_steps

METHODS = ("reference", "control", "auxiliary")
TRAINED = ("control", "auxiliary")
STEPS = 1200
SAVED = (0, 6, 400, STEPS)
SEED = 2026101066
AUX_SEED = 2026101107


def arrays(root, ids):
    with np.load(root / "inputs.npz") as data:
        return tuple(
            np.stack([data[f"{i}-{field}"] for i in ids])
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
    arms = {}
    for method in TRAINED:
        config = load_config(root / f"configs/{method}.json")
        model = AuxiliaryDistance(initial, config.model, rngs=nnx.Rngs(AUX_SEED))
        optimizer, schedule = create_optimizer(model, config)
        arm = dict(
            config=config,
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            ema=jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param)),
            update=make_steps(config)[1],
        )
        arm["gradient"] = make_gradient(config, 0.1 if method == "auxiliary" else 0.0)
        arms[method] = arm
    return arms


def make_gradient(config, weight):
    @nnx.jit
    def gradient(model, prepared):
        (_, metrics), grads = nnx.value_and_grad(
            lambda m: objective(m, prepared, config, weight), has_aux=True
        )(model)
        return metrics, grads

    return gradient


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            model = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(model, arm["ema"])
            folder = root / f"snapshots/step-{step}/{name}/{mode}"
            save_inference(
                model.core, arm["config"], folder, provenance=f"BK core only {name} {step} {mode}"
            )
            assert read(folder / "metadata.json")["parameters"] == 130605
            actual, config = load_inference(folder)
            assert config == arm["config"]
            jax.tree.map(
                np.testing.assert_array_equal,
                nnx.state(model.core, nnx.Param),
                nnx.state(actual, nnx.Param),
            )


def prepare(root):
    assert not root.exists()
    parent = Path("../jax-distance-duration-20261011/runs/distance-duration-20261011").resolve()
    verify(parent)
    assert read(parent / "final-audit.json")["fit_passed"] == {
        "control": True,
        "point": True,
        "margin": True,
    }
    assert not any(read(parent / "final-audit.json")["methods_advance"].values())
    (root / "configs").mkdir(parents=True)
    initial, config = load_inference(parent / "control")
    save_inference(
        initial, config, root / "initial", provenance="Frozen AO7200 core from BJ exact adapter"
    )
    config = config.model_copy(update={"train": config.train.model_copy(update={"steps": STEPS})})
    assert config.train.learning_rate == 5e-6 and config.train.peak_learning_rate == 1e-5
    assert config.train.minimum_learning_rate == 1e-5 and config.train.warmup_steps == 0
    assert config.train.batch_size == 4 and config.train.accumulate_steps == 1
    for method in TRAINED:
        save_config(config, root / f"configs/{method}.json")
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(parent / name, root / name)
    target_oracle(root)
    audit_data(root)
    bound = [
        parent / "source.json",
        parent / "final-audit.json",
        parent / "evaluation/metrics.json",
    ]
    bound.extend((parent / "control").iterdir())
    bound.extend(
        parent / f"evaluation/predictions/{split}-d3-control.npz" for split in ("train", "probe")
    )
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={
                "fixed_pairs": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))
            },
            parent=str(parent),
            bound_files={str(p): sha(p) for p in bound},
            crops=read(parent / "plan.json")["crops"],
            target_oracle_sha256=sha(root / "target-oracle.json"),
            steps=STEPS,
            saved=list(SAVED),
            trained=list(TRAINED),
            seed=SEED,
            auxiliary_seed=AUX_SEED,
            fresh_optimizer_and_ema=True,
            primary="finalraw1200 depth3",
            adoption=False,
            plan_document_sha256=sha(Path("docs/aux-distance-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    verify(Path(plan["parent"]))
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest, path
    assert sha(Path("docs/aux-distance-plan-20261011.md")) == plan["plan_document_sha256"]
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    return plan


def assert_control(root, prediction, ids):
    parent = Path(read(root / "plan.json")["parent"])
    split = "train" if ids == TRAIN else "probe"
    with np.load(parent / f"evaluation/predictions/{split}-d3-control.npz") as reference:
        np.testing.assert_array_equal(prediction, reference["prediction"])


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    for arm in arms.values():
        assert sum(x.size for x in jax.tree.leaves(nnx.state(arm["model"], nnx.Param))) == 134382
    data = arrays(root, TRAIN)
    prepared = {d: prepared_batch(*data, d) for d in range(1, 7)}
    control = load_inference(root / "initial")[0]
    for ids in (TRAIN, PROBE):
        expected = run_sample(control, *arrays(root, ids), 3)
        assert_control(root, expected, ids)
        for arm in arms.values():
            np.testing.assert_array_equal(run_sample(arm["model"], *arrays(root, ids), 3), expected)
    write(root / "control-preflight.json", dict(eight_outputs_exact=True, both_wrappers_exact=True))
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
                for n in (0, 100, 1199):
                    assert abs(float(arm["schedule"](n)) - 1e-5) < 1e-10
            write(
                root / "initial-audit.json",
                dict(
                    initial_parameters_exact=True,
                    optimizer_zero=True,
                    ema_equals_model=True,
                    constant_lr=1e-5,
                    initial_AO7200_core=True,
                    training_parameters=134382,
                    inference_parameters=130605,
                ),
            )
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
            write(out / "restore-audit-0.json", dict(all_state_exact=True))
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(make_arms(root)["control"]["model"], nnx.Param),
            nnx.state(make_arms(root)["auxiliary"]["model"], nnx.Param),
        )
        prefix = []
        for step in range(start, STEPS):
            depth = step % 6 + 1
            p = prepared[depth]
            order = TRAINED
            stats = {name: update(arms[name], p) for name in order}
            complete = step + 1
            if complete <= 6:
                prefix.append(stats)
            if complete <= 6 or complete % 50 == 0:
                row = dict(step=complete, depth=depth, metrics=stats)
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete == 6:
                replay = make_arms(root)
                for n in range(6):
                    p = prepared[n % 6 + 1]
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
                        nnx.state(expected.core, nnx.Param),
                        nnx.state(actual, nnx.Param),
                    )
                    bindings.update({str(p): sha(p) for p in folder.iterdir()})
        write(root / "snapshot-audit.json", dict(exports=16, raw_ema_exact=True, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def run_sample(model, clean, source, strength, depth):
    return np.asarray(forward(model, prepared_inputs(clean, source, strength, depth)))


def evaluate(root):
    plan = check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    models = {m: load_inference(root / f"snapshots/step-{STEPS}/{m}/raw")[0] for m in TRAINED}
    models["reference"] = load_inference(root / "initial")[0]
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows, bindings, displays, partial_displays, replay = [], {}, {}, {}, []
    with np.load(root / "inputs.npz") as data:
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            clean, source, strength = arrays(root, ids)
            for depth in (1, 3, 6):
                active = dict(models)
                for method, model in active.items():
                    prediction = run_sample(model, clean, source, strength, depth)
                    assert np.isfinite(prediction).all()
                    if method == "reference" and depth == 3:
                        assert_control(root, prediction, ids)
                        replay.extend(ids)
                    path = out / f"predictions/{split}-d{depth}-{method}.npz"
                    np.savez_compressed(path, prediction=prediction)
                    bindings[str(path)] = sha(path)
                    for i, value in zip(ids, prediction, strict=True):
                        target, level = data[f"{i}-image"], data[f"{i}-strength"]
                        metrics = protected_metrics(value, target, level) | halo_metrics(
                            value, target, level
                        )
                        metrics |= probe_metrics(
                            value, target, data[f"{i}-probes"], data[f"{i}-connections"]
                        )
                        metrics |= coverage_metrics(value, target, level)
                        assert all(v is None or np.isfinite(v) for v in metrics.values())
                        rows.append(
                            dict(method=method, case=i, split=split, depth=depth, **metrics)
                        )
                        if depth == 3 and method in METHODS:
                            displays[(method, i)] = value
        clean, source, strength = arrays(root, TRAIN)
        strength = strength.copy()
        strength[:, :, :256] = 0
        for method, model in models.items():
            prediction = run_sample(model, clean, source, strength, 6)
            assert np.isfinite(prediction).all()
            mask = np.asarray(strength) == 0
            np.testing.assert_array_equal(prediction[mask], np.asarray(source)[mask])
            path = out / f"predictions/partial-d6-{method}.npz"
            np.savez_compressed(path, prediction=prediction)
            bindings[str(path)] = sha(path)
            for i, value in zip(TRAIN, prediction, strict=True):
                partial_displays[(method, i)] = value
        checks = [
            c
            for row in rows
            if row["depth"] == 3 and row["method"] in METHODS
            for c in fit_checks([row])
        ]
        guards = []
        for i in TRAIN + PROBE:
            lookup = {r["method"]: r for r in rows if r["case"] == i and r["depth"] == 3}
            for metric, limit, lower in (
                ("ink_f1_alpha50", 0.005, False),
                ("ink_deficit_fraction", 0.01, True),
            ):
                for method, baseline in (
                    ("control", "reference"),
                    ("auxiliary", "reference"),
                    ("auxiliary", "control"),
                ):
                    a, b = lookup[method][metric], lookup[baseline][metric]
                    guards.append(
                        dict(
                            method=method,
                            reference=baseline,
                            case=i,
                            metric=metric,
                            value=a,
                            baseline=b,
                            passed=a <= b + limit if lower else a >= b - limit,
                        )
                    )
        assert len(rows) == 72 and len(checks) == 66 and len(guards) == 48 and len(bindings) == 21
        assert set(replay) == set(TRAIN + PROBE) and len(replay) == 8
        write(root / "replay-audit.json", dict(exact=True, cases=replay))
        write(
            out / "metrics.json",
            dict(
                rows=rows,
                checks=checks,
                retention_checks=guards,
                bindings=bindings,
                finite=True,
                protected_exact=True,
                adoption=False,
                fit_passed={
                    m: all(c["passed"] for c in checks if c["method"] == m) for m in METHODS
                },
                retention_passed={
                    m: all(c["passed"] for c in guards if c["method"] == m) for m in TRAINED
                },
            ),
        )
        records = []
        columns = ("target", "source", *METHODS)
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            for native in (True, False):
                canvas = Image.new("L", (160 * len(columns), 184 * len(ids)), 255)
                for row, i in enumerate(ids):
                    values = [
                        data[f"{i}-image"],
                        data[f"{i}-source"],
                        *[displays[(m, i)] for m in METHODS],
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
        canvas = Image.new("L", (160 * len(columns), 184 * len(TRAIN)), 255)
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
            dict(records=records, deterministic_source_model=True, crops_from_AD=True),
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
            write(root / "workflow-status.json", dict(stage="train1200"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        {"prepare": prepare, "evaluate": evaluate}[args.action](root)
