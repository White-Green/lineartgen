"""Continue the local CNN without additional Gaussian noise, matched to CO control."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from local_edge_energy import prepared_batch as control_batch
from local_recursive_duration import run_sample
from local_recursive_fit import make_arm
from procedural_velocity import SEED, teachers
from train_procedural_structure import prepare_batch
from zero_noise_rollout import without_noise

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters

START = 20000
STEPS = 24000
DATA_START = 14000
AUDIT_STEPS = sorted(set(range(DATA_START, DATA_START + 6)) | set(range(DATA_START, 18000, 100)))
TRAINED = ("local",)
SAVED = (START, START + 6, 22000, STEPS)
EVAL_SEED = 2026102093
PLAN_DOCUMENT = Path("docs/local-zero-noise-plan-20261012.md")


def split_local(reference, config):
    arm = make_arm(config, reference["model"])
    nnx.update(
        arm["optimizer"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["optimizer"]))
    )
    arm["ema"] = jax.tree.map(lambda x: x.copy(), reference["ema"])
    return arm


def load_parent(root):
    from local_procedural_duration import make_arms as parent_arms

    references = parent_arms(Path(read(root / "plan.json")["parent"]))
    with checkpoints.manager(root / "parent-checkpoints", keep=1) as saved:
        assert lineart_state.restore(saved, references, START) == START
    return references["local"]


def make_arms(root):
    reference = load_parent(root)
    config = load_config(root / "configs/local.json")
    arm = split_local(reference, config)
    initial, saved = load_inference(root / "initials/local")
    assert config == saved
    assert_initial_parameters(initial, arm["model"])
    assert int(arm["optimizer"].step[...]) == START
    return {"local": arm}


def prepared_batch(config, preparation, index, size=512):
    direct, row = control_batch(config, preparation, index, size=size)
    digest = hashlib.sha256()
    for x in jax.tree.leaves(direct):
        digest.update(np.asarray(x).tobytes())
    return without_noise(direct), {**row, "control_inputs_sha256": digest.hexdigest()}


def audit_teachers(config, root):
    from collections import Counter

    rows = []
    for index in AUDIT_STEPS:
        clean, records = teachers(config, index)
        rows.append(
            dict(
                step=index + 1,
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


def prepare(root):
    assert not root.exists()
    parent = Path(
        "../jax-local-procedural-duration-20261012/runs/local-procedural-duration-20261012"
    ).resolve()
    verify(parent)
    assert (parent / "final-audit.json").exists()
    parent_plan = read(parent / "plan.json")
    source_parent = Path(parent_plan["source_parent"])
    model, old = load_inference(parent / f"snapshots/step-{START}/local/raw")
    source = load_config(parent / "configs/source.json")
    config = old.model_copy(
        update={
            "train": old.train.model_copy(
                update={
                    "steps": STEPS,
                    "learning_rate_decay_steps": STEPS,
                    "learning_rate": 5e-6,
                    "minimum_learning_rate": 1e-5,
                }
            )
        }
    )
    assert config.train.peak_learning_rate == 1e-5 and config.train.minimum_learning_rate == 1e-5
    assert config.train.warmup_steps == 0 and config.train.batch_size == 4
    assert config.model == old.model and config.loss == old.loss
    assert config.diffusion == old.diffusion and config.sketch == old.sketch
    assert config.geometry == old.geometry and config.data == old.data
    (root / "configs").mkdir(parents=True)
    save_config(config, root / "configs/local.json")
    save_config(source, root / "configs/source.json")
    save_inference(
        model,
        config,
        root / "initials/local",
        provenance="CM actual raw20000; zero additional Gaussian continuation",
    )
    shutil.copytree(parent / f"training/checkpoints/{START}", root / f"parent-checkpoints/{START}")
    for name in ("reference-initial", "pixel"):
        shutil.copytree(parent / name, root / name)
    for name in ("inputs.npz", "reference.json"):
        shutil.copyfile(parent / name, root / name)
    bound = [p / n for p in (parent, source_parent) for n in ("source.json", "final-audit.json")]
    bound += [p for p in (root / "parent-checkpoints").rglob("*") if p.is_file()]
    bound += list((root / "reference-initial").iterdir()) + list((root / "pixel").iterdir())
    control = Path("../jax-local-edge-energy-20261012/runs/local-edge-energy-20261012").resolve()
    verify(control)
    assert (control / "final-audit.json").exists()
    shutil.copytree(control / "snapshots/step-24000/control/raw", root / "fixed-control")
    bound += [
        control / n
        for n in ("source.json", "final-audit.json", "training/sampling.jsonl", "data-audit.json")
    ]
    bound += list((root / "fixed-control").iterdir())
    for baseline in (parent, control):
        for directory in ("development", "development32"):
            bound += [baseline / directory / "metrics.json", baseline / directory / "summary.json"]
            bound += list((baseline / directory / "predictions").glob("*.npz"))
    noise_parent = Path(
        "../jax-local-noise-sensitivity-20261012/runs/local-noise-sensitivity-20261012"
    ).resolve()
    verify(noise_parent)
    assert read(noise_parent / "final-audit.json")["adoption"] is False
    bound += [noise_parent / n for n in ("source.json", "final-audit.json")]
    for directory in ("development", "development32"):
        bound += list((noise_parent / directory / "predictions").glob("*.npz"))
    audit_teachers(config, root)
    write(
        root / "train.json",
        dict(
            kind="procedural_only",
            seed=SEED,
            teachers=16000,
            data_steps=[14000, 17999],
            optimizer_steps=[20000, 23999],
        ),
    )
    write(
        root / "probe.json",
        dict(kind="reused_development", conditions=1566, counts=[2, 32], confirmation=False),
    )
    write(
        root / "plan.json",
        dict(
            parent=str(parent),
            source_parent=str(source_parent),
            broad_parent=str(control),
            noise_parent=str(noise_parent),
            additional_gaussian=False,
            initial=str(root / "initials/local"),
            initial_files={p.name: sha(p) for p in (root / "initials/local").iterdir()},
            data={
                "reference": dict(
                    path=str(root / "reference.json"), sha256=sha(root / "reference.json")
                )
            },
            bound_files={str(p): sha(p) for p in bound},
            panel_groups=parent_plan["panel_groups"],
            crops=parent_plan["crops"],
            base_cases=1562,
            blank_cases=4,
            start=START,
            steps=STEPS,
            data_start=DATA_START,
            seed=SEED,
            saved=list(SAVED),
            audit_steps=AUDIT_STEPS,
            trained=list(TRAINED),
            procedural_teachers_only=True,
            adoption=False,
            plan_document_sha256=sha(PLAN_DOCUMENT),
            data_audit_sha256=sha(root / "data-audit.json"),
            inputs_sha256=sha(root / "inputs.npz"),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    for name in ("parent", "source_parent", "broad_parent", "noise_parent"):
        verify(Path(plan[name]))
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest, path
    assert sha(PLAN_DOCUMENT) == plan["plan_document_sha256"]
    assert sha(root / "data-audit.json") == plan["data_audit_sha256"]
    assert sha(root / "inputs.npz") == plan["inputs_sha256"]
    return plan


def control_preflight(root):
    from evaluate_procedural_velocity import load_case

    from lineartgen.evaluation_sampling import inference_with_source, runtime_inputs

    plan, ref = check(root), read(root / "reference.json")
    parent = Path(plan["parent"])
    local = load_inference(root / "initials/local")[0]
    selected = set(sum(plan["panel_groups"].values(), [])) | set(range(1562, 1566))
    assert len(selected) == 38
    blocks = 0
    for steps, directory in ((2, "development"), (32, "development32")):
        for index in sorted(selected):
            case, data = load_case(plan, ref, index)
            source = data.get("source", data["image"])[None]
            actual = np.clip(
                np.asarray(
                    run_sample(
                        local,
                        jax.device_put(source),
                        jax.device_put(data["strength"][None]),
                        jax.random.PRNGKey(case["seed"]),
                        3,
                        steps=steps,
                    )
                )[0],
                -1,
                1,
            )
            with np.load(parent / directory / f"predictions/case-{index:04d}.npz") as old:
                np.testing.assert_array_equal(actual, old["local"])
            blocks += 1
    forward = nnx.jit(lambda m, x, t, s: inference_with_source(m, x, t, s, steps=1))
    reference_models = {
        "initial": load_inference(root / "reference-initial")[0],
        "pixel": load_inference(root / "pixel")[0],
    }
    reference_blocks = 0
    for index in (807, 903, 999, 1095, 1191, 1287, 1383, 1479):
        case, data = load_case(plan, ref, index)
        source = data["source"][None]
        initial = jax.device_put(runtime_inputs(source, data["strength"][None], case["seed"], 3))
        with np.load(parent / f"development/predictions/case-{index:04d}.npz") as old:
            for name, model in reference_models.items():
                np.testing.assert_array_equal(
                    np.asarray(forward(model, *initial, jax.device_put(source)))[0], old[name]
                )
                reference_blocks += 1
    write(
        root / "control-preflight.json",
        dict(CM_clipped_blocks_exact=blocks, reference_outputs_exact=reference_blocks),
    )


def zero_preflight(root):
    from evaluate_procedural_velocity import load_case
    from zero_noise_rollout import run_sample as zero_sample

    plan, ref = check(root), read(root / "reference.json")
    model = load_inference(root / "initials/local")[0]
    selected = set(sum(plan["panel_groups"].values(), [])) | set(range(1562, 1566))
    assert len(selected) == 38
    blocks = 0
    for steps, directory in ((2, "development"), (32, "development32")):
        for index in sorted(selected):
            case, data = load_case(plan, ref, index)
            source = data.get("source", data["image"])[None]
            value = zero_sample(
                model,
                jax.device_put(source),
                jax.device_put(data["strength"][None]),
                jax.random.PRNGKey(case["seed"]),
                3,
                steps=steps,
            )
            actual = np.clip(np.asarray(value)[0], -1, 1)
            with np.load(
                Path(plan["noise_parent"]) / directory / f"predictions/case-{index:04d}.npz"
            ) as old:
                np.testing.assert_array_equal(actual, old["zero"][0])
            blocks += 1
    write(root / "zero-preflight.json", dict(CN_zero_clipped_blocks_exact=blocks))


def control_update_preflight(root, preparation):
    from local_edge_energy import make_arms as control_arms
    from local_edge_energy import prepared_batch as control_batch

    parent = Path(read(root / "plan.json")["broad_parent"])
    arms, expected = control_arms(parent), control_arms(parent)
    with checkpoints.manager(parent / "training/checkpoints", keep=4) as saved:
        assert lineart_state.restore(saved, arms, START) == START
        assert lineart_state.restore(saved, expected, START + 6) == START + 6
    for index in range(DATA_START, DATA_START + 6):
        p, _ = control_batch(arms["control"]["config"], preparation, index)
        update(arms["control"], p)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"x": arms["control"]}, START + 6),
        lineart_state.payload({"x": expected["control"]}, START + 6),
    )
    write(root / "control-update-preflight.json", dict(CO_six_update_full_state_exact=True))


def train(root):
    assert os.environ.get("XLA_FLAGS", "") == "", "This run requires the audited default XLA flags"
    check(root)
    write(
        root / "backend-audit.json",
        dict(
            flags=os.environ.get("XLA_FLAGS", ""),
            device_kind=jax.devices()[0].device_kind,
            platform_version=jax.devices()[0].client.platform_version,
            nearest_double_jvp="broadcast transpose with fixed four-term reduction",
        ),
    )
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    preparation = prepare_batch(load_config(root / "configs/source.json"))
    control_preflight(root)
    zero_preflight(root)
    control_update_preflight(root, preparation)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=4) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else START
        if saved.latest_step() is None:
            references = {"local": load_parent(root)}
            for name, arm in arms.items():
                assert int(arm["optimizer"].step[...]) == START
                assert_initial_parameters(
                    load_inference(root / f"initials/{name}")[0], arm["model"]
                )
                for n, expected in ((START, 1e-5), (STEPS, 1e-5)):
                    assert abs(float(arm["schedule"](n)) - expected) < 1e-10
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload({"x": references[name]}, START),
                    lineart_state.payload({"x": arm}, START),
                )
            write(
                root / "initial-audit.json",
                dict(
                    parent_full_state_exact=True,
                    initial_parameters_exact=True,
                    optimizer_step=START,
                    parameters={"local": 15809},
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
        prefix = []
        for step in range(start, STEPS):
            depth = step % 6 + 1
            p, data_row = prepared_batch(
                arms["local"]["config"], preparation, step - START + DATA_START
            )
            order = TRAINED
            stats, elapsed = {}, {}
            for name in order:
                tick = time.perf_counter()
                stats[name] = update(arms[name], p)
                elapsed[name] = time.perf_counter() - tick
            complete = step + 1
            digest = hashlib.sha256()
            for x in jax.tree.leaves(p):
                digest.update(np.asarray(x).tobytes())
            with journal.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=complete,
                            depth=depth,
                            inputs_sha256=digest.hexdigest(),
                            elapsed_seconds=elapsed,
                            **data_row,
                        )
                    )
                    + "\n"
                )
            if complete <= START + 6:
                prefix.append(stats)
            if complete <= START + 6 or complete % 50 == 0:
                row = dict(step=complete, depth=depth, metrics=stats)
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete == START + 6:
                replay = make_arms(root)
                for n in range(START, START + 6):
                    p, _ = prepared_batch(
                        replay["local"]["config"], preparation, n - START + DATA_START
                    )
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
                        depths=[(DATA_START + n) % 6 + 1 for n in range(6)],
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "train"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="local_zero_noise20000to24000"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        prepare(root)


if __name__ == "__main__":
    main()
