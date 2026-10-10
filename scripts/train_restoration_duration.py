"""Restore R/S model, optimizer, EMA and data clock and continue at the old LR floor."""

import argparse
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from train_connection_queries import evaluate as evaluate_primary
from train_lineart_loss_study import restore_journal
from train_procedural_structure import prepare_batch

from lineartgen.config import load_config, save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import recursive_forward_with_preclip
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm
from lineartgen.training.losses import image_loss

WORKTREES = Path("/home/whitegreen/projects/lineartgen/.worktrees")
SEED, START, STEPS = 2026100961, 4000, 16000
METHODS = ("interval", "core50")
MILESTONES = (4000, 8000, 16000)


def prepare(root):
    assert not root.exists()
    parents = dict(
        interval=WORKTREES / "jax-source-interval-l1-20261010/runs/source-interval-l1-20261010",
        core50=WORKTREES / "jax-core-retention-data-20261010/runs/core-retention-data-20261010",
    )
    for parent in parents.values():
        verify(parent)
        assert read(parent / "final-audit.json")["stage"] == "complete_diagnostic_insufficient"
    common = read(parents["interval"] / "plan.json")
    initial = Path(common["initial"])
    (root / "configs").mkdir(parents=True)
    bindings, controls, references = {}, {}, {}
    matched = dict(interval="pixel", core50="core_control")
    parent_records = {}
    for name, parent in parents.items():
        old = load_config(parent / f"configs/{name}.json")
        assert old.train.steps == START and old.train.learning_rate_decay_steps is None
        config = old.model_copy(
            update={
                "train": old.train.model_copy(
                    update={"steps": STEPS, "learning_rate_decay_steps": START}
                )
            }
        )
        save_config(config, root / f"configs/{name}.json")
        checkpoint = parent / f"milestone-checkpoints/{START}"
        bindings.update({str(p): sha(p) for p in checkpoint.rglob("*") if p.is_file()})
        parent_records[name] = dict(
            root=str(parent),
            method=name,
            step=START,
            source_sha256=sha(parent / "source.json"),
            final_audit_sha256=sha(parent / "final-audit.json"),
        )
        control = matched[name]
        references[control] = dict(root=str(parent), method=name, step=START)
        for step in MILESTONES:
            destination = root / f"snapshots/step-{step}/{control}"
            shutil.copytree(parent / f"snapshots/step-{START}/{name}", destination)
            controls.update({str(p): sha(p) for p in destination.rglob("*") if p.is_file()})
    for filename in (
        "train.json",
        "probe.json",
        "inputs.npz",
        "queries.json",
        "dataset.json",
        "reference.json",
    ):
        shutil.copyfile(parents["interval"] / filename, root / filename)
        assert sha(parents["interval"] / filename) == sha(parents["core50"] / filename)
    write(
        root / "plan.json",
        dict(
            arms=list(METHODS),
            warm_start_methods=list(METHODS),
            comparison_methods=["pixel", "core_control", *METHODS],
            control_methods=["pixel", "core_control"],
            control_parents=references,
            matched_controls=matched,
            parents=parent_records,
            parent_checkpoint_files=bindings,
            reused_control_files=controls,
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir() if p.is_file()},
            data=common["data"],
            seed=SEED,
            start=START,
            steps=STEPS,
            new_updates_per_arm=STEPS - START,
            milestones=list(MILESTONES),
            retained_full_states=[START, START + 6, 8000, STEPS],
            constant_extension_lr=1e-5,
            panel_groups=common["panel_groups"],
            crops=common["crops"],
            base_cases=1562,
            blank_cases=4,
            **{
                k: common[k]
                for k in [
                    "inputs_sha256",
                    "queries_sha256",
                    "dataset_sha256",
                    "reference_sha256",
                    "evaluation_source_counts",
                ]
            },
            plan_document_sha256=sha(Path("docs/restoration-duration-plan-20261010.md")),
            diagnostic_only=True,
            adoption=False,
            sealed_confirmation_generated=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    for name in ["dataset", "reference", "inputs", "queries"]:
        suffix = ".npz" if name == "inputs" else ".json"
        assert sha(root / (name + suffix)) == plan[name + "_sha256"]
    assert sha(root / "source.tar.gz") == read(root / "source.json")["archive_sha256"]
    for key in ["reused_control_files", "parent_checkpoint_files"]:
        for path, digest in plan[key].items():
            assert sha(Path(path)) == digest
    for entry in plan["parents"].values():
        parent = Path(entry["root"])
        assert sha(parent / "source.json") == entry["source_sha256"]
        assert sha(parent / "final-audit.json") == entry["final_audit_sha256"]
    return plan


def configure_arm(config, initial):
    arm = create_arm(config, initial)

    def objective(model, prepared):
        predictions, physical = recursive_forward_with_preclip(
            model, prepared.noisy, prepared.noise_levels, guides=prepared.guides
        )
        loss, metrics = image_loss(
            predictions[0],
            prepared.targets[0],
            prepared.noise_levels[0],
            config.loss,
            config.diffusion.min_noise,
            preclip=physical[0],
        )
        return loss, dict(metrics, loss=loss)

    @nnx.jit
    def gradient(model, prepared):
        (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared)
        return metrics, grads

    arm["gradient"] = gradient
    return arm


def make_arms(root):
    initial, _ = load_inference(read(root / "plan.json")["initial"])
    return {
        name: configure_arm(load_config(root / f"configs/{name}.json"), initial) for name in METHODS
    }


def restore_parents(root, arms):
    plan = read(root / "plan.json")
    initial, _ = load_inference(plan["initial"])
    for name, entry in plan["parents"].items():
        parent = Path(entry["root"])
        step = entry["step"]
        method = entry["method"]
        old = configure_arm(load_config(parent / f"configs/{method}.json"), initial)
        with checkpoints.manager(parent / "milestone-checkpoints", keep=3) as saved:
            assert lineart_state.restore(saved, {method: old}, step) == step
            assert lineart_state.restore(saved, {method: arms[name]}, step) == step
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({name: old}, step),
            lineart_state.payload({name: arms[name]}, step),
        )
        assert int(arms[name]["optimizer"].step[...]) == step
    write(
        root / "parent-state-audit.json",
        dict(
            model_optimizer_ema_exact=True,
            global_step=plan["start"],
            learning_rates={
                name: float(arm["schedule"](plan["start"])) for name, arm in arms.items()
            },
        ),
    )


def restore_audit(root, saved, arms, step):
    restored = make_arms(root)
    assert lineart_state.restore(saved, restored, step) == step
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, step),
        lineart_state.payload(restored, step),
    )
    assert all(int(arm["optimizer"].step[...]) == step for arm in restored.values())
    write(root / f"training/restore-audit-{step}.json", dict(all_state_exact=True, step=step))


def evaluate(root, arms, step):
    plan = check(root)
    controls = {
        name: dict(model=load_inference(root / f"snapshots/step-{step}/{name}/raw")[0])
        for name in plan["control_methods"]
    }
    evaluate_primary(root, dict(**controls, **arms), step)
    with np.load(root / f"evaluation/step-{step}/predictions.npz") as actual:
        rows = read(root / f"evaluation/step-{step}/metrics.json")["rows"]
        for name, ref in plan["control_parents"].items():
            parent = Path(ref["root"])
            old_step = ref["step"]
            with np.load(parent / f"evaluation/step-{old_step}/predictions.npz") as expected:
                keys = [k for k in actual.files if k.startswith(name + "-")]
                assert len(keys) == 64
                for key in keys:
                    np.testing.assert_array_equal(
                        actual[key], expected[key.replace(name + "-", ref["method"] + "-", 1)]
                    )
                    if step == START:
                        candidate = next(
                            m for m, c in plan["matched_controls"].items() if c == name
                        )
                        np.testing.assert_array_equal(
                            actual[key], actual[key.replace(name + "-", candidate + "-", 1)]
                        )
            old_rows = read(parent / f"evaluation/step-{old_step}/metrics.json")["rows"]
            assert [r for r in rows if r["method"] == name] == [
                dict(r, method=name) for r in old_rows if r["method"] == ref["method"]
            ]
    write(
        root / f"evaluation/step-{step}/control-audit.json",
        dict(all_exact=True, conditions_per_control=64, controls=list(controls)),
    )


def audit_broad(root, step):
    for name, ref in check(root)["control_parents"].items():
        parent = Path(ref["root"])
        old_step = ref["step"]
        for split in ["train", "probe"]:
            with (
                np.load(root / f"broad/step-{step}/{name}-{split}.npz") as a,
                np.load(parent / f"broad/step-{old_step}/{ref['method']}-{split}.npz") as b,
            ):
                np.testing.assert_array_equal(a["images"], b["images"])
        actual = [
            r for r in read(root / f"broad/step-{step}/metrics.json")["rows"] if r["method"] == name
        ]
        expected = [
            dict(r, method=name)
            for r in read(parent / f"broad/step-{old_step}/metrics.json")["rows"]
            if r["method"] == ref["method"]
        ]
        assert actual == expected and len(actual) == 456
    write(
        root / f"broad/step-{step}/control-audit.json",
        dict(all_exact=True, conditions_per_control=456),
    )


def train(root, end):
    check(root)
    assert end in (4006, 8000, STEPS) and jax.devices()[0].platform != "cpu"
    assert (
        read(root / "data-audit.json")["crop_resume_exact"]
        and read(root / "test-results.json")["passed"]
    )
    arms = make_arms(root)
    config = arms["interval"]["config"]
    sources = discover(config.data)
    training, validation, final = split_experiment(sources, config.data, SEED)
    assert dataset_description(sources, training, validation, final) == read(root / "dataset.json")
    preparers = {n: prepare_batch(a["config"]) for n, a in arms.items()}
    destination = root / "training"
    destination.mkdir(exist_ok=True)
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        if saved.latest_step() is None:
            restore_parents(root, arms)
            lineart_state.save(saved, arms, START)
            restore_audit(root, saved, arms, START)
            shutil.copytree(
                destination / f"checkpoints/{START}", root / f"milestone-checkpoints/{START}"
            )
            snapshot(root, arms, START)
            evaluate(root, arms, START)
            start = START
        else:
            start = lineart_state.restore(saved, arms)
            restore_journal(destination, start)
        assert start <= end
        if end > 4006:
            assert read(destination / "restore-audit-4006.json")["all_state_exact"]
        stream = batches(
            training, config.data, batch_size=4, count=end - start, seed=SEED, start=start
        )
        key = jax.random.PRNGKey(SEED)
        counts = {d: 0 for d in range(1, 7)}
        teachers = {}
        try:
            for step, batch in enumerate(stream, start=start):
                begin = time.perf_counter()
                clean, records = mix_geometry(
                    batch.images,
                    config.geometry,
                    seed=SEED,
                    step=step,
                    image_scale=config.data.image_scale,
                )
                for record in records:
                    kind = record["kinds"][0] if record.get("coverage") else "original_geometry"
                    teachers[kind] = teachers.get(kind, 0) + 1
                teachers["native_corpus"] = teachers.get("native_corpus", 0) + 4 - len(records)
                depth = config.diffusion.levels_for_scale(
                    512, config.model.minimum_size, batch.scale_level
                )
                counts[depth] += 1
                prepared = {
                    n: f(jnp.asarray(clean), jax.random.fold_in(key, step), depth)
                    for n, f in preparers.items()
                }
                if step < START + 6:
                    assert (
                        hashlib.sha256(clean.tobytes()).hexdigest()
                        == read(root / "data-audit.json")["first_12_extension_target_hashes"][
                            step - START
                        ]
                    )
                    for p in prepared.values():
                        active = np.asarray(p.noise_levels[0]) > 0
                        np.testing.assert_array_equal(
                            np.asarray(p.targets[0])[active], clean[active]
                        )
                        np.testing.assert_array_equal(
                            np.asarray(p.targets[0])[~active], np.asarray(p.guides[0])[~active]
                        )
                order = METHODS if step % 2 == 0 else METHODS[::-1]
                stats = {n: update(arms[n], prepared[n]) for n in order}
                complete = step + 1
                row = dict(
                    step=complete,
                    depth=depth,
                    scale_level=batch.scale_level,
                    generated=len(records),
                    coverage_generated=sum(bool(r.get("coverage")) for r in records),
                    arms=stats,
                    learning_rates={n: float(a["schedule"](step)) for n, a in arms.items()},
                    seconds=time.perf_counter() - begin,
                    clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                )
                if step < START + 6 or complete % 25 == 0:
                    with (destination / "training.jsonl").open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                    write(destination / "progress.json", row)
                    print(json.dumps(row), flush=True)
                if complete % 100 == 0 or complete == 4006:
                    lineart_state.save(saved, arms, complete)
                if complete in (4006, 8000, STEPS):
                    restore_audit(root, saved, arms, complete)
                    shutil.copytree(
                        destination / f"checkpoints/{complete}",
                        root / f"milestone-checkpoints/{complete}",
                    )
                    if complete != 4006:
                        snapshot(root, arms, complete)
                        evaluate(root, arms, complete)
        finally:
            stream.close()
        write(
            destination / f"segment-{start}-{end}.json",
            dict(
                start=start,
                end=end,
                depth_counts=counts,
                teacher_counts=teachers,
                original_global_data_clock=True,
                optimizer_clock_not_reset=True,
                clean_targets_protected_correct=True,
            ),
        )


def audit_snapshots(root):
    check(root)
    bindings = {}
    for step in (8000, STEPS):
        arms = make_arms(root)
        with checkpoints.manager(root / "milestone-checkpoints", keep=4) as saved:
            lineart_state.restore(saved, arms, step)
        for name, arm in arms.items():
            for mode in ("raw", "ema"):
                expected = nnx.clone(arm["model"])
                if mode == "ema":
                    nnx.update(expected, arm["ema"])
                folder = root / f"snapshots/step-{step}/{name}/{mode}"
                actual, config = load_inference(folder)
                assert config == arm["config"]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    nnx.state(expected, nnx.Param),
                    nnx.state(actual, nnx.Param),
                )
                bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
    write(
        root / "snapshot-audit.json",
        dict(full_state_restore_exact=True, raw_ema_exact=True, models=8, bindings=bindings),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "freeze", "train", "audit", "audit-broad"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--end", type=int, default=STEPS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "freeze":
        freeze(root)
    elif args.action == "audit":
        audit_snapshots(root)
    elif args.action == "audit-broad":
        audit_broad(root, args.end)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, args.end)
