"""Point versus interval velocity on fresh, entirely procedural teachers."""

import argparse
import fcntl
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from geometry_velocity_fit import prepared_batch as bridge_batch
from interval_velocity_fit import gradient, interval_gradient
from paper_band_fit import PROBE, TRAIN
from train_procedural_structure import prepare_batch
from velocity_bridge_toy import velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm

SEED = 2026101111
STEPS = 4000
METHODS = ("mse", "interval")
SAVED = (0, 6, 1000, STEPS)
AUDIT_STEPS = sorted(set(range(12)) | set(range(0, STEPS, 25)))


def teachers(config, step, size=512):
    white = np.ones((config.train.batch_size, size, size, 1), np.float32)
    clean, records = mix_geometry(
        white, config.geometry, seed=SEED, step=step, image_scale=config.data.image_scale
    )
    assert len(records) == len(clean) and {r["item"] for r in records} == set(range(len(clean)))
    return clean, records


def prepared_batch(config, prepare, step, size=512):
    clean, records = teachers(config, step, size)
    depth = step % 6 + 1
    p = prepare(jax.numpy.asarray(clean), jax.random.fold_in(jax.random.PRNGKey(SEED), step), depth)
    result = velocity_inputs(
        bridge_batch(p.targets[0], p.guides[0], p.noise_levels[0], step, depth)[0]
    )
    active = np.asarray(p.noise_levels[0]) > config.diffusion.min_noise
    np.testing.assert_array_equal(np.asarray(p.targets[0])[active], clean[active])
    np.testing.assert_array_equal(
        np.asarray(p.targets[0])[~active], np.asarray(p.guides[0])[~active]
    )
    counts = Counter(r["kinds"][0] if r.get("coverage") else "geometry" for r in records)
    row = dict(
        step=step + 1,
        depth=depth,
        teacher_counts=dict(counts),
        clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
    )
    return result, row


def make_arms(root):
    initial, _ = load_inference(root / "initial")
    arms = {}
    for name in METHODS:
        arm = create_arm(load_config(root / f"configs/{name}.json"), initial)
        assert_initial_parameters(initial, arm["model"])
        arm["gradient"] = gradient if name == "mse" else interval_gradient
        arms[name] = arm
    return arms


def prepare(root):
    assert not root.exists()
    parent = Path(
        "../jax-aux-distance-generalization-20261011/runs/aux-distance-generalization-20261011"
    ).resolve()
    broad = Path(
        "../jax-paper-band-generalization-20261010/runs/paper-band-generalization-20261010"
    ).resolve()
    velocity_parent = Path(
        "../jax-geometry-velocity-fit-20261010/runs/geometry-velocity-fit-20261010"
    ).resolve()
    for p in (parent, broad, velocity_parent):
        verify(p)
        assert (p / "final-audit.json").exists()
    old = read(broad / "plan.json")
    source_config = load_config(parent / "configs/control.json")
    initial, velocity_config = load_inference(velocity_parent / "initial")
    config = source_config.model_copy(
        update={
            "model": velocity_config.model,
            "loss": velocity_config.loss,
            "train": source_config.train.model_copy(update={"learning_rate_decay_steps": STEPS}),
        }
    )
    assert config.train.steps == STEPS and config.train.seed == SEED
    assert config.train.peak_learning_rate == 1e-4 and config.diffusion.scale_loss_multiplier == 0
    assert config.geometry.probability == 0.9 and config.geometry.coverage_probability == 1
    assert config.sketch.mode == "mixed" and config.sketch.probability == 0.8
    (root / "configs").mkdir(parents=True)
    save_config(source_config, root / "configs/source.json")
    for name in METHODS:
        save_config(config, root / f"configs/{name}.json")
    save_inference(
        initial,
        config,
        root / "initial",
        provenance="BC initial AZ interval36000; broad procedural flow training",
    )
    assert read(root / "initial/metadata.json")["parameters"] == 130628
    shutil.copytree(parent / "initial", root / "reference-initial")
    shutil.copytree(parent / "pixel", root / "pixel")
    shutil.copyfile(broad / "reference.json", root / "reference.json")
    write(root / "train.json", dict(kind="procedural_only", seed=SEED, teachers=STEPS * 4))
    write(root / "probe.json", dict(kind="reused_development", reference=str(broad), count=1566))
    bound = [
        p / n for p in (parent, broad, velocity_parent) for n in ("source.json", "final-audit.json")
    ]
    bound += list((root / "pixel").iterdir()) + list((root / "reference-initial").iterdir())
    bound += list((velocity_parent / "initial").iterdir())
    bound += [
        parent / "configs/control.json",
        parent / "data-audit.json",
        parent / "training/sampling.jsonl",
        broad / "development/metrics.json",
        broad / "development/native-review/manifest.json",
    ]
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={
                "reference": dict(
                    path=str(root / "reference.json"), sha256=sha(root / "reference.json")
                )
            },
            parent=str(parent),
            broad_parent=str(broad),
            velocity_parent=str(velocity_parent),
            bound_files={str(p): sha(p) for p in bound},
            panel_groups=old["panel_groups"],
            crops=old["crops"],
            base_cases=1562,
            blank_cases=4,
            steps=STEPS,
            seed=SEED,
            saved=list(SAVED),
            audit_steps=AUDIT_STEPS,
            methods=list(METHODS),
            procedural_teachers_only=True,
            adoption=False,
            plan_document_sha256=sha(Path("docs/procedural-velocity-plan-20261011.md")),
        ),
    )
    audit_data(root)
    plan = read(root / "plan.json")
    plan["data_audit_sha256"] = sha(root / "data-audit.json")
    write(root / "plan.json", plan)
    assert read(root / "data-audit.json") == read(parent / "data-audit.json")


def audit_data(root):
    config = load_config(root / "configs/source.json")
    rows = []
    for step in AUDIT_STEPS:
        clean, records = teachers(config, step)
        rows.append(
            dict(
                step=step + 1,
                clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                teacher_counts=dict(
                    Counter(r["kinds"][0] if r.get("coverage") else "geometry" for r in records)
                ),
            )
        )
    write(
        root / "data-audit.json",
        dict(rows=rows, procedural_only=True, independently_regenerated=True),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    for name in ("parent", "broad_parent", "velocity_parent"):
        verify(Path(plan[name]))
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest, path
    assert sha(root / "data-audit.json") == plan["data_audit_sha256"]
    assert sha(Path("docs/procedural-velocity-plan-20261011.md")) == plan["plan_document_sha256"]
    return plan


def preflight(root, arms):
    from evaluate_procedural_velocity import load_case

    from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs

    plan, ref = check(root), read(root / "reference.json")
    forward = nnx.jit(lambda m, x, t, s: inference_with_source(m, x, t, s, steps=1))
    models = {
        "initial": load_inference(root / "reference-initial")[0],
        "pixel": load_inference(root / "pixel")[0],
    }
    for index in TRAIN + PROBE:
        case, data = load_case(plan, ref, index)
        source = data["source"][None]
        level = data["strength"][None]
        inputs = jax.device_put(runtime_inputs(source, level, case["seed"], 3))
        source = jax.device_put(source)
        with np.load(
            Path(plan["broad_parent"]) / f"development/predictions/case-{index:04d}.npz"
        ) as old:
            for name, model in models.items():
                actual = np.asarray(forward(model, *inputs, source))[0]
                np.testing.assert_array_equal(actual, old[name])
    from aux_distance_generalization import prepared_batch as original_preparation
    from geometry_velocity_fit import EVAL_SEED, run_sample
    from geometry_velocity_fit import arrays as fixed_arrays

    source_config = load_config(root / "configs/source.json")
    source_prepare = prepare_batch(source_config)
    for step in range(6):
        old, _ = original_preparation(source_config, source_prepare, step)
        actual, _ = prepared_batch(source_config, source_prepare, step)
        for a, b in (
            (old["target"], actual["target"]),
            (old["levels"], actual["levels"]),
            (old["guides"], actual["guides"]),
        ):
            jax.tree.map(np.testing.assert_array_equal, a, b)
    for split, ids in (("train", TRAIN), ("probe", PROBE)):
        _, source, level = fixed_arrays(Path(plan["velocity_parent"]), ids)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 300 + int(split == "probe") * 10)
        value = run_sample(arms["mse"]["model"], source, level, key, 3)
        with np.load(
            Path(plan["velocity_parent"]) / f"evaluation/predictions/{split}-d3-r0-initial.npz"
        ) as old:
            np.testing.assert_array_equal(value, old["physical"])
    write(
        root / "control-preflight.json",
        dict(reference_outputs_exact=16, source_batches_exact=6, velocity_initial_outputs_exact=8),
    )


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"] and jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    config = load_config(root / "configs/source.json")
    prepare = prepare_batch(config)
    preflight(root, arms)
    data_audit = {r["step"]: r for r in read(root / "data-audit.json")["rows"]}
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        if any(r["step"] > start for r in rows):
            write(out / f"resume-discarded-{start}.json", [r for r in rows if r["step"] > start])
            rows = [r for r in rows if r["step"] <= start]
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert [r["step"] for r in rows] == list(range(1, start + 1))
        if saved.latest_step() is None:
            jax.tree.map(
                np.testing.assert_array_equal,
                nnx.state(arms["mse"]["model"]),
                nnx.state(arms["interval"]["model"]),
            )
            for arm in arms.values():
                assert int(arm["optimizer"].step[...]) == 0
                jax.tree.map(
                    np.testing.assert_array_equal, arm["ema"], nnx.state(arm["model"], nnx.Param)
                )
            write(
                root / "initial-audit.json",
                dict(
                    full_parameters_identical=True,
                    optimizer_zero=True,
                    ema_equals_model=True,
                    training_parameters=130628,
                    inference_parameters=130628,
                ),
            )
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
            write(out / "restore-audit-0.json", dict(all_state_exact=True))
        prefix = []
        for step in range(start, STEPS):
            p, row = prepared_batch(config, prepare, step)
            complete = step + 1
            if complete in data_audit:
                assert row["clean_sha256"] == data_audit[complete]["clean_sha256"]
                assert row["teacher_counts"] == data_audit[complete]["teacher_counts"]
            stats = {m: update(a, p) for m, a in arms.items()}
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if complete <= 6:
                prefix.append(stats)
            if complete <= 6 or complete % 50 == 0:
                write(out / "progress.json", dict(**row, metrics=stats))
                print(complete, stats, flush=True)
            if complete == 6:
                replay = make_arms(root)
                for n in range(6):
                    p, _ = prepared_batch(config, prepare, n)
                    values = {m: update(a, p) for m, a in replay.items()}
                    assert values == prefix[n]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, 6),
                    lineart_state.payload(replay, 6),
                )
                write(
                    root / "prefix-audit.json",
                    dict(
                        updates=6, all_state_exact=True, metrics_exact=True, inputs_regenerated=True
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=16, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "train"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="train4000"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        prepare(root)
