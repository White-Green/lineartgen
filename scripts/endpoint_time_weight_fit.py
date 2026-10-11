"""Compare bounded endpoint risk with capped and absent late-time amplification."""

import argparse
import fcntl
import hashlib
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import velocity as endpoint_velocity
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from endpoint_time_weight import cap_gradient, endpoint_gradient
from flax import nnx
from geometry_velocity_fit import prepared_batch
from paper_band_fit import PROBE, TRAIN, fit_checks, target_oracle
from PIL import Image, ImageDraw
from velocity_bridge_toy import sample as uniform_sample
from velocity_bridge_toy import velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm

METHODS = ("control", "cap", "endpoint")
TRAINED = ("cap", "endpoint")
START = 6000
STEPS = 7200
SEED = 2026101093
EVAL_SEED = 2026102093
SAVED = (START, START + 6, 6600, STEPS)


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


def control_preflight(root):
    parent = Path(read(root / "plan.json")["parent"])
    initial = load_inference(root / "initial")[0]
    for split, ids in (("train", TRAIN), ("probe", PROBE)):
        _, source, strength = arrays(root, ids)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 300 + int(split == "probe") * 10)
        value = run_sample(initial, source, strength, key, 3)
        with np.load(parent / f"evaluation/predictions/{split}-d3-r0-modulated.npz") as old:
            np.testing.assert_array_equal(value, old["physical"])
    write(root / "control-preflight.json", dict(eight_CA_outputs_exact=True))


def split_arms(reference, configs):
    arms = {}
    for name in TRAINED:
        arm = create_arm(configs[name], reference["model"])
        nnx.update(arm["model"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["model"])))
        nnx.update(
            arm["optimizer"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["optimizer"]))
        )
        arm["ema"] = jax.tree.map(lambda x: x.copy(), reference["ema"])
        arm["gradient"] = cap_gradient if name == "cap" else endpoint_gradient
        arms[name] = arm
    return arms


def load_parent(root):
    from geometry_endpoint_fit import make_arms as parent_arms

    parent = Path(read(root / "plan.json")["parent"])
    arms = parent_arms(parent)
    with checkpoints.manager(root / "parent-checkpoints", keep=1) as saved:
        assert lineart_state.restore(saved, arms, START) == START
    return arms["modulated"]


def make_arms(root):
    reference = load_parent(root)
    arms = split_arms(reference, {m: load_config(root / f"configs/{m}.json") for m in TRAINED})
    initial, _ = load_inference(root / "initial")
    for arm in arms.values():
        assert_initial_parameters(initial, arm["model"])
        assert int(arm["optimizer"].step[...]) == START
    return arms


def prepare(root):
    assert not root.exists()
    parent = Path("../jax-geometry-endpoint-20261011/runs/geometry-endpoint-20261011").resolve()
    from geometry_endpoint_fit import check as parent_check

    parent_check(parent)
    assert (
        read(parent / "final-audit.json")["stage"]
        == "complete_geometry_endpoint_fitting_diagnostic"
    )
    model, old = load_inference(parent / f"snapshots/step-{START}/modulated/raw")
    c = old.model_copy(
        update={
            "train": old.train.model_copy(
                update={
                    "steps": STEPS,
                    "learning_rate": 5e-6,
                    "minimum_learning_rate": 1e-5,
                    "warmup_steps": 0,
                    "learning_rate_decay_steps": START,
                }
            )
        }
    )
    assert c.train.peak_learning_rate == 1e-5 and c.train.batch_size == 4
    (root / "configs").mkdir(parents=True)
    for name in TRAINED:
        save_config(c, root / f"configs/{name}.json")
    save_inference(
        model, c, root / "initial", provenance="CA modulated6000 actual Adam/EMA continuation"
    )
    assert read(root / "initial/metadata.json")["parameters"] == 134688
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(parent / name, root / name)
    target_oracle(root)
    audit_data(root)
    shutil.copytree(parent / f"training/checkpoints/{START}", root / f"parent-checkpoints/{START}")
    reference_parent = Path(
        "../jax-source-band-endpoint-20261011/runs/source-band-endpoint-20261011"
    ).resolve()
    from source_band_fit import check as reference_check

    reference_check(reference_parent)
    assert (
        read(reference_parent / "final-audit.json")["stage"]
        == "complete_source_band_endpoint_diagnostic"
    )
    shutil.copytree(reference_parent / "snapshots/step-7200/control/raw", root / "reference")
    bound = [parent / n for n in ("source.json", "final-audit.json", "evaluation/metrics.json")]
    bound += [
        reference_parent / n
        for n in (
            "source.json",
            "final-audit.json",
            "evaluation/metrics.json",
            "training/sampling.jsonl",
        )
    ]
    bound += list((reference_parent / "evaluation/predictions").glob("*-control.npz"))
    bound += list((reference_parent / "snapshots/step-7200/control/raw").iterdir())
    bound += list((root / "reference").iterdir())
    bound += list((parent / f"snapshots/step-{START}/modulated/raw").iterdir())
    bound += list((parent / "evaluation/predictions").glob("*-modulated.npz"))
    for path in (root / f"parent-checkpoints/{START}").rglob("*"):
        if path.is_file():
            original = (
                parent
                / f"training/checkpoints/{START}"
                / path.relative_to(root / f"parent-checkpoints/{START}")
            )
            assert sha(path) == sha(original)
            bound += [path, original]
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={
                "fixed_pairs": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))
            },
            parent=str(parent),
            reference_parent=str(reference_parent),
            loss_floors={"control": 1 / 32, "cap": 0.25, "endpoint": 1.0},
            bound_files={str(p): sha(p) for p in bound},
            crops=read(parent / "plan.json")["crops"],
            target_oracle_sha256=sha(root / "target-oracle.json"),
            start=START,
            steps=STEPS,
            saved=list(SAVED),
            seed=SEED,
            evaluation_seed=EVAL_SEED,
            depths=list(range(1, 7)),
            adoption=False,
            plan_document_sha256=sha(Path("docs/endpoint-time-weight-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    from geometry_endpoint_fit import check as parent_check

    parent_check(Path(plan["parent"]))
    from source_band_fit import check as reference_check

    reference_check(Path(plan["reference_parent"]))
    assert sha(Path("docs/endpoint-time-weight-plan-20261011.md")) == plan["plan_document_sha256"]
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    return plan


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    data = arrays(root, TRAIN)
    control_preflight(root)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=4) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else START
        if saved.latest_step() is None:
            reference = load_parent(root)
            for arm in arms.values():
                assert int(arm["optimizer"].step[...]) == START
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload({"x": reference}, START),
                    lineart_state.payload({"x": arm}, START),
                )
                for n in (0, START, STEPS - 1):
                    assert abs(float(arm["schedule"](n)) - 1e-5) < 1e-10
            write(
                root / "initial-audit.json",
                dict(
                    parent_full_state_exact=True,
                    initial_parameters_exact=True,
                    optimizer_step=START,
                    parameters=134688,
                ),
            )
            lineart_state.save(saved, arms, START)
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, START) == START
            jax.tree.map(
                np.testing.assert_array_equal,
                lineart_state.payload(arms, START),
                lineart_state.payload(restored, START),
            )
            snapshot(root, arms, START)
            write(out / f"restore-audit-{START}.json", dict(all_state_exact=True))
        journal = out / "sampling.jsonl"
        records = (
            [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        )
        if any(r["step"] > start for r in records):
            write(out / f"resume-discarded-{start}.json", [r for r in records if r["step"] > start])
            records = [r for r in records if r["step"] <= start]
            journal.write_text("".join(json.dumps(r) + "\n" for r in records))
        assert [r["step"] for r in records] == list(range(START + 1, start + 1))
        reference_records = {
            r["step"]: r
            for r in [
                json.loads(s)
                for s in (
                    Path(read(root / "plan.json")["reference_parent"]) / "training/sampling.jsonl"
                )
                .read_text()
                .splitlines()
            ]
        }
        prefix = []
        for step in range(start, STEPS):
            depth = step % 6 + 1
            batch, tau = prepared_batch(*data, step, depth)
            p = velocity_inputs(batch)
            order = TRAINED
            stats = {name: update(arms[name], p) for name in order}
            complete = step + 1
            digest = hashlib.sha256()
            for x in jax.tree.leaves(p):
                digest.update(np.asarray(x).tobytes())
            assert digest.hexdigest() == reference_records[complete]["inputs_sha256"]
            assert depth == reference_records[complete]["depth"]
            assert float(tau.mean()) == reference_records[complete]["tau_mean"]
            with journal.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=complete,
                            depth=depth,
                            inputs_sha256=digest.hexdigest(),
                            tau_mean=float(tau.mean()),
                        )
                    )
                    + "\n"
                )
            if complete <= START + 6:
                prefix.append(stats)
            if complete <= START + 6 or complete % 50 == 0:
                row = dict(step=complete, depth=depth, metrics=stats, tau_mean=float(tau.mean()))
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete == START + 6:
                replay = make_arms(root)
                for n in range(START, START + 6):
                    p = velocity_inputs(prepared_batch(*data, n, n % 6 + 1)[0])
                    values = {name: update(replay[name], p) for name in TRAINED}
                    if start == START:
                        assert values == prefix[n - START]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, START + 6),
                    lineart_state.payload(replay, START + 6),
                )
                write(
                    root / "prefix-audit.json",
                    dict(
                        updates=6,
                        all_state_exact=True,
                        metrics_exact=start == START,
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
        write(root / "snapshot-audit.json", dict(exports=16, raw_ema_exact=True, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def run_sample(model, source, strength, key, depth, method="initial"):
    def predict(x, t, s, u):
        return endpoint_velocity(model, x, t, s, u, False)

    return np.asarray(uniform_sample(predict, source, strength, key, depth=depth, steps=32))


def evaluate(root):
    plan = check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    models = {m: load_inference(root / f"snapshots/step-{STEPS}/{m}/raw")[0] for m in TRAINED}
    models["control"] = load_inference(root / "reference")[0]
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    old_metrics = read(Path(plan["reference_parent"]) / "evaluation/metrics.json")
    old_rows = {
        (r["case"], r["depth"], r["draw"]): r
        for r in old_metrics["rows"]
        if r["method"] == "control"
    }
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
                    for method, model in active_models.items():
                        physical = run_sample(model, source, strength, key, depth, method)
                        prediction = np.clip(physical, -1, 1)
                        assert np.isfinite(physical).all()
                        if method == "control":
                            reference = (
                                Path(plan["reference_parent"])
                                / f"evaluation/predictions/{split}-d{depth}-r{draw}-control.npz"
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
                            if method == "control":
                                assert {**rows[-1], "method": "control"} == old_rows[i, depth, draw]
                            if depth == 3 and method in METHODS:
                                displays[(method, i, draw)] = value
                    write(
                        out / "progress.json",
                        dict(split=split, depth=depth, draw=draw, rows=len(rows), total=288),
                    )
        clean, source, strength = arrays(root, TRAIN)
        strength = strength.at[:, :, :256].set(0)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 9000)
        for method, model in models.items():
            physical = run_sample(model, source, strength, key, 6, method)
            prediction = np.clip(physical, -1, 1)
            mask = np.asarray(strength) == 0
            assert np.isfinite(physical).all()
            np.testing.assert_array_equal(physical[mask], np.asarray(source)[mask])
            np.testing.assert_array_equal(prediction[mask], np.asarray(source)[mask])
            if method == "control":
                reference = (
                    Path(plan["reference_parent"]) / "evaluation/predictions/partial-d6-control.npz"
                )
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
        write(root / "replay-audit.json", dict(exact=True, metric_rows_exact=96, files=replay))
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
                for compared, reference in (
                    ("cap", "control"),
                    ("endpoint", "control"),
                    ("endpoint", "cap"),
                ):
                    for metric, limit, lower in (
                        ("ink_f1_alpha50", 0.005, False),
                        ("ink_deficit_fraction", 0.01, True),
                    ):
                        a, b = values[compared][metric], values[reference][metric]
                        guards.append(
                            dict(
                                case=i,
                                draw=draw,
                                metric=metric,
                                method=compared,
                                reference=reference,
                                passed=a <= b + limit if lower else a >= b - limit,
                                value=a,
                                reference_value=b,
                            )
                        )
        assert (
            len(rows) == 288 and len(checks) == 264 and len(guards) == 192 and len(bindings) == 75
        )
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
        canvas = Image.new("L", (160 * (2 + len(METHODS)), 184 * 4), 255)
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
            dict(
                records=records,
                all_four_fixed_draws=True,
                partial_one_draw_per_case=True,
                crops_from_AD=True,
            ),
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
            write(root / "workflow-status.json", dict(stage="continue6000to7200"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        {"prepare": prepare, "evaluate": evaluate}[args.action](root)
