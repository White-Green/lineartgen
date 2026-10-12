"""Continue the small local CNN on varied synthetic strokes and editing masks."""

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
from geometry_velocity_fit import prepared_batch as bridge_batch
from local_recursive_duration import arrays, run_sample
from local_recursive_fit import make_arm
from paper_band_fit import PROBE, TRAIN
from procedural_velocity import AUDIT_STEPS, SEED, teachers
from rollout_endpoint import inputs
from train_procedural_structure import prepare_batch

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters

START = 6000
STEPS = 10000
TRAINED = ("local",)
SAVED = (START, START + 6, 8000, STEPS)
EVAL_SEED = 2026102093
PLAN_DOCUMENT = Path("docs/local-procedural-rollout-plan-20261012.md")


def split_local(reference, config):
    arm = make_arm(config, reference["model"])
    nnx.update(
        arm["optimizer"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["optimizer"]))
    )
    arm["ema"] = jax.tree.map(lambda x: x.copy(), reference["ema"])
    return arm


def load_parent(root):
    from local_recursive_duration import make_arms as parent_arms

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
    clean, records = teachers(config, index, size)
    depth = index % 6 + 1
    p = preparation(
        jax.numpy.asarray(clean), jax.random.fold_in(jax.random.PRNGKey(SEED), index), depth
    )
    result = inputs(bridge_batch(p.targets[0], p.guides[0], p.noise_levels[0], index, depth)[0])
    active = np.asarray(p.noise_levels[0]) > config.diffusion.min_noise
    np.testing.assert_array_equal(np.asarray(p.targets[0])[active], clean[active])
    np.testing.assert_array_equal(
        np.asarray(p.targets[0])[~active], np.asarray(p.guides[0])[~active]
    )
    from collections import Counter

    return result, dict(
        data_step=index + 1,
        teacher_counts=dict(
            Counter(r["kinds"][0] if r.get("coverage") else "geometry" for r in records)
        ),
        clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
    )


def prepare(root):
    assert not root.exists()
    parent = Path(
        "../jax-local-recursive-stable-20261011/runs/local-recursive-duration-20261011"
    ).resolve()
    source_parent = Path(
        "../jax-aux-distance-generalization-20261011/runs/aux-distance-generalization-20261011"
    ).resolve()
    broad = Path(
        "../jax-paper-band-generalization-20261010/runs/paper-band-generalization-20261010"
    ).resolve()
    for p in (parent, source_parent, broad):
        verify(p)
        assert (p / "final-audit.json").exists()
    model, old = load_inference(parent / f"snapshots/step-{START}/local/raw")
    source = load_config(source_parent / "configs/control.json")
    config = source.model_copy(
        update=dict(
            model=old.model,
            loss=old.loss,
            train=old.train.model_copy(update=dict(steps=STEPS, seed=SEED)),
        )
    )
    assert config.train.peak_learning_rate == config.train.minimum_learning_rate == 1e-4
    assert config.train.batch_size == 4 and config.train.learning_rate_decay_steps == 1200
    assert config.diffusion.scale_loss_multiplier == 0 and config.matching.method == "independent"
    assert config.geometry.probability == 0.9 and config.geometry.coverage_probability == 1
    assert config.sketch.mode == "mixed" and config.sketch.probability == 0.8
    (root / "configs").mkdir(parents=True)
    save_config(config, root / "configs/local.json")
    save_config(source, root / "configs/source.json")
    save_inference(
        model,
        config,
        root / "initials/local",
        provenance="CJ actual raw6000, broad synthetic continuation",
    )
    shutil.copytree(parent / f"training/checkpoints/{START}", root / f"parent-checkpoints/{START}")
    shutil.copytree(source_parent / "initial", root / "reference-initial")
    shutil.copytree(source_parent / "pixel", root / "pixel")
    shutil.copyfile(parent / "inputs.npz", root / "inputs.npz")
    shutil.copyfile(broad / "reference.json", root / "reference.json")
    shutil.copyfile(source_parent / "data-audit.json", root / "data-audit.json")
    bound = [
        p / n for p in (parent, source_parent, broad) for n in ("source.json", "final-audit.json")
    ]
    bound += [source_parent / "training/sampling.jsonl", source_parent / "configs/control.json"]
    bound += list((root / "reference-initial").iterdir()) + list((root / "pixel").iterdir())
    bound += [p for p in (root / "parent-checkpoints").rglob("*") if p.is_file()]
    for directory in ("evaluation", "evaluation32"):
        bound += [p for p in (parent / directory / "predictions").glob("*-local.npz")]
    broad_plan = read(broad / "plan.json")
    write(
        root / "train.json",
        dict(
            kind="procedural_only",
            seed=SEED,
            teachers=16000,
            data_steps=[0, 3999],
            optimizer_steps=[6000, 9999],
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
            broad_parent=str(broad),
            initial=str(root / "initials/local"),
            initial_files={p.name: sha(p) for p in (root / "initials/local").iterdir()},
            data={
                "reference": dict(
                    path=str(root / "reference.json"), sha256=sha(root / "reference.json")
                )
            },
            bound_files={str(p): sha(p) for p in bound},
            panel_groups=broad_plan["panel_groups"],
            crops=broad_plan["crops"],
            base_cases=1562,
            blank_cases=4,
            start=START,
            steps=STEPS,
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
    for name in ("parent", "source_parent", "broad_parent"):
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

    plan = check(root)
    parent = Path(plan["parent"])
    local = load_inference(root / "initials/local")[0]
    blocks = 0
    for steps, directory in ((2, "evaluation"), (32, "evaluation32")):
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            _, source, strength = arrays(root, ids)
            for depth in (1, 3, 6):
                for draw in range(4):
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(EVAL_SEED),
                        depth * 100 + int(split == "probe") * 10 + draw,
                    )
                    actual = run_sample(local, source, strength, key, depth, steps=steps)
                    with np.load(
                        parent / directory / f"predictions/{split}-d{depth}-r{draw}-local.npz"
                    ) as old:
                        np.testing.assert_array_equal(actual, old["physical"])
                    blocks += 1
        _, source, strength = arrays(root, TRAIN)
        strength = strength.at[:, :, :256].set(0)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 9000)
        actual = run_sample(local, source, strength, key, 6, steps=steps)
        with np.load(parent / directory / "predictions/partial-d6-local.npz") as old:
            np.testing.assert_array_equal(actual, old["physical"])
        blocks += 1
    assert blocks == 50
    forward = nnx.jit(lambda m, x, t, s: inference_with_source(m, x, t, s, steps=1))
    ref = read(root / "reference.json")
    reference_models = {
        "initial": load_inference(root / "reference-initial")[0],
        "pixel": load_inference(root / "pixel")[0],
    }
    reference_blocks = 0
    for index in (807, 903, 999, 1095, 1191, 1287, 1383, 1479):
        case, data = load_case(plan, ref, index)
        source = data["source"][None]
        initial = jax.device_put(runtime_inputs(source, data["strength"][None], case["seed"], 3))
        with np.load(
            Path(plan["broad_parent"]) / f"development/predictions/case-{index:04d}.npz"
        ) as old:
            for name, model in reference_models.items():
                actual = np.asarray(forward(model, *initial, jax.device_put(source)))[0]
                np.testing.assert_array_equal(actual, old[name])
                reference_blocks += 1
    write(
        root / "control-preflight.json",
        dict(CJ_output_blocks_exact=blocks, reference_outputs_exact=reference_blocks),
    )


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
                for n in (0, START, STEPS - 1):
                    assert abs(float(arm["schedule"](n)) - 1e-4) < 1e-10
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
            p, data_row = prepared_batch(arms["local"]["config"], preparation, step - START)
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
                    p, _ = prepared_batch(replay["local"]["config"], preparation, n - START)
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
            write(root / "workflow-status.json", dict(stage="continue_local_procedural6000to10000"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        prepare(root)


if __name__ == "__main__":
    main()
