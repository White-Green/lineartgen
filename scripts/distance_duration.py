"""Continue both distance objectives to the AO duration without optimizer resets."""

import argparse
import fcntl
import shutil
from pathlib import Path

import distance_margin_fit as margin_parent
import distance_supervision_fit as point_parent
import jax
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from edge_energy_fit import forward
from flax import nnx
from paper_band_fit import PROBE, TRAIN, fit_checks, target_oracle
from PIL import Image, ImageDraw

from lineartgen.config import ExperimentConfig, load_config, save_config
from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm

METHODS = ("control", "point", "margin")
TRAINED = ("point", "margin")
START, SWITCH, STEPS = 1200, 6000, 7200
SAVED = (START, START + 6, SWITCH, SWITCH + 6, STEPS)
SEED = 2026101066
PARENTS = {
    "point": Path(
        "../jax-distance-supervision-fit-replay-20261011/runs/distance-supervision-fit-20261011"
    ).resolve(),
    "margin": Path(
        "../jax-distance-margin-fit-20261011/runs/distance-margin-fit-20261011"
    ).resolve(),
}
MODULES = {"point": point_parent, "margin": margin_parent}
PARENT_KEYS = {"point": "distance", "margin": "margin"}

arrays = margin_parent.arrays
audit_data = margin_parent.audit_data
run_sample = margin_parent.run_sample


def phase_config(config, settled):
    rate = 5e-6 if settled else 5e-5
    return config.model_copy(
        update={
            "train": config.train.model_copy(
                update={
                    "steps": STEPS,
                    "learning_rate": rate,
                    "minimum_learning_rate": rate * 2,
                }
            )
        }
    )


def make_arms(root, step=START):
    initial, _ = load_inference(root / "initial")
    arms = {}
    for method in TRAINED:
        phase = "settle" if step > SWITCH else "constant"
        config = load_config(root / f"configs/{method}-{phase}.json")
        arm = create_arm(config, initial)
        arm["gradient"] = make_gradient(config, MODULES[method].objective)
        arms[method] = arm
    return arms


def make_gradient(config, objective):
    @nnx.jit
    def gradient(model, prepared):
        (_, metrics), grads = nnx.value_and_grad(
            lambda m: objective(m, prepared, config), has_aux=True
        )(model)
        return metrics, grads

    return gradient


def copy_numerical_state(source, destination, step):
    values = lineart_state.payload(source, step)["arms"]
    for method, arm in destination.items():
        checkpoints.restore_into(arm["model"], values[method]["model"])
        checkpoints.restore_into(arm["optimizer"], values[method]["optimizer"])
        nnx.replace_by_pure_dict(arm["ema"], jax.tree.map(jax.numpy.asarray, values[method]["ema"]))
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(source, step),
        lineart_state.payload(destination, step),
    )


def restore_parents(root, arms=None):
    plan = read(root / "plan.json")
    result = {}
    for method in TRAINED:
        parent = Path(plan["parents"][method])
        key = PARENT_KEYS[method]
        selected = MODULES[method].make_arms(parent) if arms is None else {key: arms[method]}
        with checkpoints.manager(parent / "training/checkpoints", keep=4) as saved:
            assert lineart_state.restore(saved, selected, START) == START
        result[method] = selected[key]
    return result


def prepare(root):
    assert not root.exists()
    control_parent = Path(
        "../jax-late-fit-settling-20261010/runs/late-fit-settling-20261010"
    ).resolve()
    for parent in (*PARENTS.values(), control_parent):
        verify(parent)
    assert read(control_parent / "final-audit.json")["methods_advance"]["settle"]
    data_parent = Path("../jax-paper-band-fit-20261010/runs/paper-band-fit-20261010").resolve()
    verify(data_parent)
    (root / "configs").mkdir(parents=True)
    common = None
    for method, parent in PARENTS.items():
        key = PARENT_KEYS[method]
        config = load_config(parent / f"configs/{key}.json")
        assert config.train.steps == START and config.train.peak_learning_rate == 1e-4
        assert config.train.warmup_steps == 0 and config.train.batch_size == 4
        assert config.train.accumulate_steps == 1
        if common is None:
            common = config
        else:
            assert config == common
        for settled in (False, True):
            save_config(
                phase_config(config, settled),
                root / f"configs/{method}-{'settle' if settled else 'constant'}.json",
            )
    model, config = load_inference(PARENTS["point"] / "initial")
    save_inference(
        model,
        config,
        root / "initial",
        provenance="R4000 template only; restore actual BH/BI full state",
    )
    original_control = control_parent / "snapshots/step-7200/settle/raw"
    old_config = read(original_control / "config.json")
    ignored = {
        "geometry": {"short_gap_probability": 0.0, "short_gap_maximum": 6.0},
        "loss": {"saturation_margin": 0.5, "saturation_margin_radius": 3, "ink_mass_weight": 0.0},
    }
    for section, fields in ignored.items():
        for name, value in fields.items():
            assert old_config[section].pop(name) == value
    config = ExperimentConfig.model_validate(old_config)
    assert config.model == common.model and config.loss == common.loss
    shutil.copytree(original_control, root / "control")
    save_config(config, root / "control/config.json")
    assert sha(root / "control/weights.npz") == sha(original_control / "weights.npz")
    load_inference(root / "control")
    write(
        root / "control-config-adapter.json",
        dict(
            removed_training_only_fields=ignored,
            model_config_unchanged=True,
            weights_bytes_exact=True,
            original_files={str(p): sha(p) for p in original_control.iterdir()},
        ),
    )
    assert read(root / "initial/metadata.json")["parameters"] == 130605
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(data_parent / name, root / name)
    target_oracle(root)
    audit_data(root)
    bound = []
    for method, parent in PARENTS.items():
        assert sha(parent / "inputs.npz") == sha(root / "inputs.npz")
        bound.extend(p for p in (parent / "training/checkpoints/1200").rglob("*") if p.is_file())
        bound.extend(
            p
            for p in (parent / f"snapshots/step-1200/{PARENT_KEYS[method]}/raw").iterdir()
            if p.is_file()
        )
    for parent in (*PARENTS.values(), control_parent, data_parent):
        bound.extend([parent / "source.json", parent / "final-audit.json"])
    bound.extend((root / "control").iterdir())
    bound.extend(original_control.iterdir())
    bound.append(root / "control-config-adapter.json")
    bound.append(control_parent / "evaluation/step-7200/predictions.npz")
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={
                "fixed_pairs": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))
            },
            parents={m: str(p) for m, p in PARENTS.items()},
            control_parent=str(control_parent),
            data_parent=str(data_parent),
            bound_files={str(p): sha(p) for p in bound},
            crops=read(data_parent / "plan.json")["crops"],
            target_oracle_sha256=sha(root / "target-oracle.json"),
            start_step=START,
            switch_step=SWITCH,
            steps=STEPS,
            additional_updates_per_arm=STEPS - START,
            trained=list(TRAINED),
            methods=list(METHODS),
            saved=list(SAVED),
            seed=SEED,
            depths=list(range(1, 7)),
            primary="final raw7200 depth3",
            adoption=False,
            plan_document_sha256=sha(Path("docs/distance-duration-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert plan["start_step"] == START and plan["switch_step"] == SWITCH and plan["steps"] == STEPS
    assert sha(Path("docs/distance-duration-plan-20261011.md")) == plan["plan_document_sha256"]
    for parent in (*plan["parents"].values(), plan["control_parent"], plan["data_parent"]):
        verify(Path(parent))
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest, path
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    return plan


def assert_control(root, prediction, ids):
    parent = Path(read(root / "plan.json")["control_parent"])
    with np.load(parent / "evaluation/step-7200/predictions.npz") as reference:
        for i, value in zip(ids, prediction, strict=True):
            np.testing.assert_array_equal(value, reference[f"settle-{i}"])


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    data = arrays(root, TRAIN)
    prepared = {m: {d: MODULES[m].prepared_batch(*data, d) for d in range(1, 7)} for m in TRAINED}
    control = load_inference(root / "control")[0]
    for ids in (TRAIN, PROBE):
        assert_control(root, run_sample(control, *arrays(root, ids), 3), ids)
    write(root / "control-preflight.json", dict(eight_outputs_exact=True))
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = saved.latest_step() if saved.latest_step() is not None else START
        arms = make_arms(root, start)
        if saved.latest_step() is None:
            restore_parents(root, arms)
            original = restore_parents(root)
            jax.tree.map(
                np.testing.assert_array_equal,
                lineart_state.payload(arms, START),
                lineart_state.payload(original, START),
            )
            for method, arm in arms.items():
                assert int(arm["optimizer"].step[...]) == START
                for depth in range(1, 7):
                    p = prepared[method][depth]
                    np.testing.assert_array_equal(
                        forward(arm["model"], p), forward(original[method]["model"], p)
                    )
                np.testing.assert_allclose(
                    [arm["schedule"](n) for n in (1200, 3000, 5999)], 1e-4, rtol=1e-6
                )
            write(
                root / "initial-audit.json",
                dict(
                    model_optimizer_ema_exact=True,
                    optimizer_step=START,
                    all_depth_outputs_exact=True,
                    constant_lr=1e-4,
                ),
            )
            lineart_state.save(saved, arms, START)
            snapshot(root, arms, START)
            write(out / f"restore-audit-{START}.json", dict(all_state_exact=True))
        else:
            assert lineart_state.restore(saved, arms, start) == start
        prefixes = {}
        for step in range(start, STEPS):
            if step == SWITCH:
                previous = arms
                arms = make_arms(root, SWITCH + 1)
                copy_numerical_state(previous, arms, SWITCH)
                for arm in arms.values():
                    np.testing.assert_allclose(
                        [arm["schedule"](n) for n in (6000, 6600, 7199)], 1e-5, rtol=1e-6
                    )
                    assert int(arm["optimizer"].step[...]) == SWITCH
                write(
                    root / "phase-switch-audit.json",
                    dict(
                        all_state_exact=True, optimizer_step=SWITCH, lr_before=1e-4, lr_after=1e-5
                    ),
                )
            depth = step % 6 + 1
            stats = {m: update(arms[m], prepared[m][depth]) for m in TRAINED}
            complete = step + 1
            if START < complete <= START + 6 or SWITCH < complete <= SWITCH + 6:
                prefixes[complete] = stats
            if complete in prefixes or complete % 50 == 0:
                row = dict(step=complete, depth=depth, metrics=stats)
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete in (START + 6, SWITCH + 6):
                origin = complete - 6
                replay = restore_parents(root) if origin == START else make_arms(root, SWITCH + 1)
                if origin == SWITCH:
                    assert lineart_state.restore(saved, replay, SWITCH) == SWITCH
                for n in range(origin, complete):
                    values = {m: update(replay[m], prepared[m][n % 6 + 1]) for m in TRAINED}
                    assert values == prefixes[n + 1]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, complete),
                    lineart_state.payload(replay, complete),
                )
                write(
                    root / f"prefix-audit-{origin}.json",
                    dict(
                        updates=6,
                        all_state_exact=True,
                        metrics_exact=True,
                        depths=list(range(1, 7)),
                    ),
                )
            if complete in SAVED:
                lineart_state.save(saved, arms, complete)
                restored = make_arms(root, complete)
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
            restored = make_arms(root, step)
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
        write(root / "snapshot-audit.json", dict(exports=20, raw_ema_exact=True, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def evaluate(root):
    plan = check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    models = {
        "point": load_inference(root / f"snapshots/step-{STEPS}/point/raw")[0],
        "margin": load_inference(root / f"snapshots/step-{STEPS}/margin/raw")[0],
        "control": load_inference(root / "control")[0],
    }
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
                    if method == "control" and depth == 3:
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
                for method in TRAINED:
                    a, b = lookup[method][metric], lookup["control"][metric]
                    guards.append(
                        dict(
                            method=method,
                            case=i,
                            metric=metric,
                            value=a,
                            control=b,
                            passed=a <= b + limit if lower else a >= b - limit,
                        )
                    )
        assert len(rows) == 72 and len(checks) == 66 and len(guards) == 32 and len(bindings) == 21
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
            write(root / "workflow-status.json", dict(stage="train7200"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        {"prepare": prepare, "evaluate": evaluate}[args.action](root)
