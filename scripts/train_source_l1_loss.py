"""Paired L1 pixel-loss study with original and uniform strength weighting."""

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
from diagnose_learnability import arms_from_root, freeze, snapshot, update, verify
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
from lineartgen.training.losses import image_loss

WORKTREES = Path("/home/whitegreen/projects/lineartgen/.worktrees")
SEED = 2026100961
STEPS = 4000
METHODS = ("l1", "flat_l1")


def prepare(root):
    assert not root.exists()
    parent = WORKTREES / "jax-ink-area-coverage-20261010/runs/ink-area-coverage-20261010"
    verify(parent)
    assert read(parent / "workflow-status.json")["stage"].startswith(
        ("numerical_complete", "complete_")
    )
    old = read(parent / "plan.json")
    initial = Path(old["initial"])
    base = load_config(parent / "configs/coverage.json")
    assert base.geometry.probability == 0.2 and base.geometry.coverage_probability == 0.125
    assert base.data.native_training_targets and base.loss.noise_weight_floor == 0.05
    assert base.train.peak_learning_rate == 1e-4 and base.diffusion.scale_loss_multiplier == 0
    assert base.model.sampling_mode == "one_pass" and base.train.seed == SEED
    (root / "configs").mkdir(parents=True)
    for name, original in (("l1", "coverage"), ("flat_l1", "flat_coverage")):
        config = load_config(parent / f"configs/{original}.json")
        assert config.loss.pixel_loss == "clipped_mse"
        config = config.model_copy(
            update={"loss": config.loss.model_copy(update={"pixel_loss": "clipped_l1"})}
        )
        save_config(config, root / f"configs/{name}.json")
    controls, control_parents = {}, {}
    for name, method in (("pixel", "coverage"), ("flat", "flat_coverage")):
        control_parents[name] = dict(root=str(parent), method=method)
        for step in (0, 1000, STEPS):
            source = parent / f"snapshots/step-{step}/{method}"
            destination = root / f"snapshots/step-{step}/{name}"
            shutil.copytree(source, destination)
            controls.update({str(p): sha(p) for p in destination.rglob("*") if p.is_file()})
    for name in ("train.json", "probe.json", "inputs.npz", "queries.json"):
        shutil.copyfile(parent / name, root / name)
    sources = discover(base.data)
    training, validation, final = split_experiment(sources, base.data, SEED)
    description = dataset_description(sources, training, validation, final)
    expected = read(
        WORKTREES / "jax-detail-study-20261008/runs/detail-mixed-20261008-screen/dataset.json"
    )
    assert description == expected
    write(root / "dataset.json", description)
    reference = read(WORKTREES / "jax-rewrite/runs/lineart-loss-20261006/evaluation.json")
    manifest = read(Path(reference["root"]) / "cases.json")
    real_ids = {r["source"] for r in manifest["cases"] if r["kind"] == "real"}
    assert not real_ids & set(description["train"])
    assert real_ids <= set(description["validation"])
    write(root / "reference.json", reference)
    panels = read(
        WORKTREES / "jax-source-feature-broad-20261009/runs/source-feature-broad-20261009/plan.json"
    )
    groups = dict(panels["panel_groups"], partial=[717, 727])
    crops = dict(panels["crops"], **{"717": [160, 96, 320, 256], "727": [160, 96, 320, 256]})
    old = read(parent / "plan.json")
    write(
        root / "plan.json",
        dict(
            arms=list(METHODS),
            warm_start_methods=list(METHODS),
            comparison_methods=["pixel", "flat", "l1", "flat_l1"],
            control_parents=control_parents,
            reused_control_files=controls,
            control_methods=["pixel", "flat"],
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir() if p.is_file()},
            data=old["data"],
            parent=str(parent),
            seed=SEED,
            steps=STEPS,
            milestones=[0, 1000, STEPS],
            initial_outputs_equal=True,
            inputs_sha256=sha(root / "inputs.npz"),
            queries_sha256=sha(root / "queries.json"),
            dataset_sha256=sha(root / "dataset.json"),
            reference_sha256=sha(root / "reference.json"),
            panel_groups=groups,
            crops=crops,
            base_cases=1562,
            blank_cases=4,
            plan_document_sha256=sha(Path("docs/source-l1-loss-plan-20261010.md")),
            diagnostic_only=True,
            adoption=False,
            sealed_confirmation_generated=False,
            evaluation_source_counts={"train": 0, "validation": len(real_ids), "final_test": 0},
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    for name, key in [
        ("dataset.json", "dataset_sha256"),
        ("reference.json", "reference_sha256"),
        ("inputs.npz", "inputs_sha256"),
        ("queries.json", "queries_sha256"),
    ]:
        assert sha(root / name) == plan[key]
    assert sha(root / "source.tar.gz") == read(root / "source.json")["archive_sha256"]
    for path, digest in plan["reused_control_files"].items():
        assert sha(Path(path)) == digest
    return plan


def make_arms(root):
    arms = arms_from_root(root)
    for arm in arms.values():
        config = arm["config"]

        def objective(model, prepared, config=config):
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

        def make_gradient(objective):
            @nnx.jit
            def gradient(model, prepared):
                (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model, prepared)
                return metrics, grads

            return gradient

        arm["gradient"] = make_gradient(objective)
    return arms


def restore_audit(root, saved, arms, step):
    restored = make_arms(root)
    assert lineart_state.restore(saved, restored, step) == step
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, step),
        lineart_state.payload(restored, step),
    )
    write(root / f"training/restore-audit-{step}.json", dict(all_state_exact=True, step=step))


def evaluate(root, arms, step):
    plan = check(root)
    controls = {
        name: dict(model=load_inference(root / f"snapshots/step-{step}/{name}/raw")[0])
        for name in plan["control_methods"]
    }
    evaluate_primary(root, dict(**controls, **arms), step)
    actual = np.load(root / f"evaluation/step-{step}/predictions.npz")
    rows = read(root / f"evaluation/step-{step}/metrics.json")["rows"]
    for name, ref in plan["control_parents"].items():
        parent = Path(ref["root"])
        with np.load(parent / f"evaluation/step-{step}/predictions.npz") as expected:
            keys = [key for key in actual.files if key.startswith(name + "-")]
            assert len(keys) == 64
            for key in keys:
                other = key.replace(name + "-", ref["method"] + "-", 1)
                np.testing.assert_array_equal(actual[key], expected[other])
                if step == 0:
                    for candidate in METHODS:
                        np.testing.assert_array_equal(
                            actual[key], actual[key.replace(name + "-", candidate + "-", 1)]
                        )
        expected_rows = [
            dict(r, method=name)
            for r in read(parent / f"evaluation/step-{step}/metrics.json")["rows"]
            if r["method"] == ref["method"]
        ]
        assert [r for r in rows if r["method"] == name] == expected_rows
    write(
        root / f"evaluation/step-{step}/control-audit.json",
        dict(all_exact=True, conditions_per_control=64, controls=list(controls)),
    )


def audit_broad(root, step):
    plan = check(root)
    for name, ref in plan["control_parents"].items():
        parent = Path(ref["root"])
        for split in ("train", "probe"):
            actual = np.load(root / f"broad/step-{step}/{name}-{split}.npz")["images"]
            expected = np.load(parent / f"broad/step-{step}/{ref['method']}-{split}.npz")["images"]
            np.testing.assert_array_equal(actual, expected)
        actual = [
            r for r in read(root / f"broad/step-{step}/metrics.json")["rows"] if r["method"] == name
        ]
        expected = [
            dict(r, method=name)
            for r in read(parent / f"broad/step-{step}/metrics.json")["rows"]
            if r["method"] == ref["method"]
        ]
        assert actual == expected and len(actual) == 456
    write(
        root / f"broad/step-{step}/control-audit.json",
        dict(all_exact=True, conditions_per_control=456, controls=plan["control_methods"]),
    )


def train(root, end):
    check(root)
    assert end in (6, 1000, STEPS) and jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    config = arms["l1"]["config"]
    sources = discover(config.data)
    training, validation, final = split_experiment(sources, config.data, SEED)
    assert dataset_description(sources, training, validation, final) == read(root / "dataset.json")
    preparers = {name: prepare_batch(arm["config"]) for name, arm in arms.items()}
    destination = root / "training"
    destination.mkdir(exist_ok=True)
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(destination, start)
        if start == 0:
            snapshot(root, arms, 0)
            evaluate(root, arms, 0)
        elif end > 6:
            assert read(root / "training/restore-audit-6.json")["all_state_exact"]
        stream = batches(
            training, config.data, batch_size=4, count=end - start, seed=SEED, start=start
        )
        key = jax.random.PRNGKey(SEED)
        counts = {d: 0 for d in range(1, 7)}
        coverage_counts = {}
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
                    coverage_counts[kind] = coverage_counts.get(kind, 0) + 1
                coverage_counts["native_corpus"] = (
                    coverage_counts.get("native_corpus", 0) + 4 - len(records)
                )
                depth = config.diffusion.levels_for_scale(
                    512, config.model.minimum_size, batch.scale_level
                )
                counts[depth] += 1
                prepared = {
                    n: f(jnp.asarray(clean), jax.random.fold_in(key, step), depth)
                    for n, f in preparers.items()
                }
                if step < 6:
                    assert (
                        hashlib.sha256(clean.tobytes()).hexdigest()
                        == read(root / "data-audit.json")["first_12_target_hashes"][step]
                    )
                    p = prepared["l1"]
                    jax.tree.map(np.testing.assert_array_equal, p, prepared["flat_l1"])
                    active = np.asarray(p.noise_levels[0]) > 0
                    np.testing.assert_array_equal(np.asarray(p.targets[0])[active], clean[active])
                    np.testing.assert_array_equal(
                        np.asarray(p.targets[0])[~active], np.asarray(p.guides[0])[~active]
                    )
                order = METHODS[step % 2 :] + METHODS[: step % 2]
                stats = {n: update(arms[n], prepared[n]) for n in order}
                complete = step + 1
                row = dict(
                    step=complete,
                    depth=depth,
                    scale_level=batch.scale_level,
                    generated=len(records),
                    coverage_generated=sum(bool(r.get("coverage")) for r in records),
                    arms=stats,
                    seconds=time.perf_counter() - begin,
                    clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                )
                if step < 6 or complete % 25 == 0:
                    with (destination / "training.jsonl").open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                    write(destination / "progress.json", row)
                    print(json.dumps(row), flush=True)
                if complete % 100 == 0 or complete == 6:
                    lineart_state.save(saved, arms, complete)
                if complete in (6, 1000, STEPS):
                    restore_audit(root, saved, arms, complete)
                    shutil.copytree(
                        destination / f"checkpoints/{complete}",
                        root / f"milestone-checkpoints/{complete}",
                    )
                    if complete != 6:
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
                teacher_counts=coverage_counts,
                virtual_depth_unchanged=True,
                clean_targets_protected_correct=True,
            ),
        )


def audit_snapshots(root):
    check(root)
    bindings = {}
    for step in (1000, STEPS):
        arms = make_arms(root)
        with checkpoints.manager(root / "milestone-checkpoints", keep=3) as saved:
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "train", "audit", "audit-broad"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--end", type=int, default=STEPS)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "freeze":
        freeze(root)
    elif args.action == "audit-broad":
        audit_broad(root, args.end)
    elif args.action == "audit":
        audit_snapshots(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, args.end)
