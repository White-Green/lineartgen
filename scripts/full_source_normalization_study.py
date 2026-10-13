"""Compare mandatory full-source normalization against a frozen, matched raw-input control."""

import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from cue_capacity import forward, make_arm, prepare_inputs
from cue_preserving_data import MODES, strength_map
from diagnose_learnability import freeze, snapshot, verify
from flax import nnx
from plausible_study import digest, exact, replay_comparison
from rough_family_data import BASE, augmentation, rough_source
from style_family_objective import style_gradient

from lineartgen.config import ModelConfig, load_config, save_config
from lineartgen.diffusion import source_guides
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state

METHODS = ("normalized",)
REPORT_METHODS = ("raw", "normalized")
STEPS = 18000
SAVED = (0, 6, 6000, STEPS)
DOCUMENT = Path("docs/full-source-normalization-plan-20261013.md")
PARENT = Path("../jax-fresh-context-family-replay-20261013/runs/fresh-context-family-20261013")
SEED = 2026101334
EVAL_SEED = 2026101335
MODEL_SEED = 2026101336


def make_arms(root):
    result = {}
    for method in METHODS:
        model, config = load_inference(root / f"initials/{method}")
        assert config == load_config(root / f"configs/{method}.json")
        result[method] = make_arm(config, model)
        result[method]["gradient"] = style_gradient()
    return result


def selection(step):
    rng = np.random.default_rng([SEED, step])
    return dict(
        indices=rng.integers(64, size=4).tolist(),
        variants=rng.integers(8, size=4).tolist(),
        modes=rng.integers(len(MODES), size=4).tolist(),
        depth=step % 6 + 1,
        augmentation=[augmentation(SEED, step, item) for item in range(4)],
    )


def training_batch(data, step, method):
    selected = selection(step)
    indices = np.asarray(selected["indices"]) * 8
    indices += np.asarray(selected["variants"])
    candidates = jnp.asarray(data["variants"][indices])
    assert method in METHODS
    options = selected["augmentation"]
    source = jnp.asarray(
        np.stack(
            [
                data["source"][index]
                if opts == BASE
                else rough_source(data["variants"][index, [1, 25]].mean(0), opts)
                for index, opts in zip(indices, options, strict=True)
            ]
        )
    )
    strength = np.stack([strength_map(MODES[m]) for m in selected["modes"]])
    prepared = prepare_inputs(
        candidates[:, 13],
        source,
        strength,
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        depth=selected["depth"],
    )
    return (
        prepared,
        jnp.asarray(data["canonical"][indices]),
        {k: jnp.asarray(data[k][indices]) for k in ("probes", "valid")}
        | dict(normalizers=candidates),
    )


def update(arm, prepared, candidates, annotation):
    parts, grads = arm["gradient"](arm["model"], prepared, candidates, annotation)
    parts["gradient_norm"] = optax.global_norm(grads)
    metrics = {k: float(v) for k, v in parts.items()}
    assert all(np.isfinite(x) for x in metrics.values())
    _, arm["ema"] = arm["update"](arm["model"], arm["optimizer"], grads, arm["ema"])
    jax.block_until_ready(arm["ema"])
    return metrics


def prepare(root):
    parent = PARENT.resolve()
    assert read(parent / "completion.json")["stage"] == "complete_fresh_context_family"
    root.mkdir(parents=True, exist_ok=False)
    raw, config = load_inference(parent / "initials/plain")
    assert config.train.steps == STEPS and config.train.seed == SEED
    assert config.train.model_seed == MODEL_SEED and config.train.peak_learning_rate == 1e-4
    (root / "configs").mkdir()
    shutil.copytree(parent / "initials/plain", root / "initials/raw")
    mc = ModelConfig.model_validate(
        config.model.model_dump() | dict(source_preprocess="local_peak_full")
    )
    cfg = config.model_copy(update={"model": mc})
    save_config(cfg, root / "configs/normalized.json")
    fresh = create_model(mc, rngs=nnx.Rngs(MODEL_SEED))
    exact(nnx.state(raw, nnx.Param), nnx.state(fresh, nnx.Param))
    save_inference(
        fresh,
        cfg,
        root / "initials/normalized",
        provenance="Fresh shared DI seed; full source normalization",
    )
    shutil.copytree(
        parent / "snapshots/step-18000/plain/raw", root / "snapshots/step-18000/raw/raw"
    )
    shutil.copytree(parent / "snapshots/step-18000/dilated/raw", root / "reference")
    for name in ("data", "stress-data"):
        shutil.copytree(parent / name, root / name)
    for name in (
        "train.npz",
        "data.json",
        "canonical.npz",
        "path-annotations.npz",
        "stress-plan.json",
    ):
        shutil.copy2(parent / name, root / name)
    shutil.copy2(parent / "training/sampling.jsonl", root / "control-journal.jsonl")
    write(root / "sampling-plan.json", [dict(step=k, **selection(k)) for k in range(STEPS)])
    write(
        root / "train.json",
        dict(cases=512, seed=SEED, steps=STEPS, profile_coefficient=0.25, reused_control=True),
    )
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    bindings = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    for name in (
        "completion.json",
        "source.json",
        "snapshot-audit.json",
        "prefix-audit.json",
        "training/sampling.jsonl",
    ):
        bindings[str(parent / name)] = sha(parent / name)
    for folder in (
        parent / "initials/plain",
        parent / "snapshots/step-6/plain",
        parent / "snapshots/step-18000/plain",
        parent / "snapshots/step-18000/dilated/raw",
    ):
        for path in folder.rglob("*"):
            if path.is_file():
                bindings[str(path)] = sha(path)
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initials/normalized"),
            initial_files={p.name: sha(p) for p in (root / "initials/normalized").iterdir()},
            data={},
            bound_files=bindings,
            parent=str(parent),
            methods=METHODS,
            report_methods=REPORT_METHODS,
            steps=STEPS,
            saved=SAVED,
            reused_control="DI plain raw18000; same seed, batches, optimizer and schedule",
            plan_document_sha256=sha(DOCUMENT),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(DOCUMENT) == plan["plan_document_sha256"]
    for file, h in plan["bound_files"].items():
        assert sha(Path(file)) == h, file
    return plan


def train(root):
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    arms = make_arms(root)
    with np.load(root / "train.npz") as z:
        data = {k: z[k] for k in z.files}
    with np.load(root / "path-annotations.npz") as z:
        data.update({k: z[k] for k in z.files})
    with np.load(root / "canonical.npz") as z:
        data["canonical"] = z["canonical"]
    control_model, control_config = load_inference(root / "initials/raw")
    exact(nnx.state(control_model, nnx.Param), nnx.state(arms["normalized"]["model"], nnx.Param))
    assert sum(x.size for x in jax.tree.leaves(nnx.state(control_model, nnx.Param))) == 15809
    for step in range(6):
        p, _, _ = training_batch(data, step, "normalized")
        plain = forward(control_model, p)
        expected = tuple(
            jnp.where(t > 0, g, x)
            for t, g, x in zip(
                p.noise_levels,
                source_guides(arms["normalized"]["model"], p.guides),
                p.noisy,
                strict=True,
            )
        )
        actual = forward(arms["normalized"]["model"], p)
        exact(plain[0], p.guides)
        exact(actual[0], expected)
    control = make_arm(control_config, control_model)
    control["gradient"] = style_gradient()
    control_records = [
        json.loads(s) for s in (root / "control-journal.jsonl").read_text().splitlines()
    ]
    control_metrics = []
    for step in range(6):
        actual = update(control, *training_batch(data, step, "normalized"))
        control_metrics.append(
            replay_comparison(actual, control_records[step]["metrics"]["plain"], "prior")
        )
    control_exports = {}
    for mode in ("raw", "ema"):
        expected, _ = load_inference(PARENT / f"snapshots/step-6/plain/{mode}")
        actual = nnx.state(control["model"], nnx.Param) if mode == "raw" else control["ema"]
        control_exports[mode] = replay_comparison(actual, nnx.state(expected, nnx.Param), "prior")
    write(
        root / "control-replay.json",
        dict(passed=True, metrics=control_metrics, exports=control_exports),
    )
    write(
        root / "initial-audit.json",
        dict(
            six_depth_expected_outputs_exact=True,
            initial_parameters_exact=True,
            initial_functions_different_by_design=True,
            generator_parameters=15809,
            fresh_initialization_seed=MODEL_SEED,
            control_reused=True,
        ),
    )
    write(
        root / "backend.json",
        dict(flags=os.environ.get("XLA_FLAGS", ""), device=jax.devices()[0].device_kind),
    )
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            records = [json.loads(s) for s in journal.read_text().splitlines()]
            records = [x for x in records if x["step"] <= start]
            assert [x["step"] for x in records] == list(range(1, start + 1))
            journal.write_text("".join(json.dumps(x) + "\n" for x in records))
        if start == 0:
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        prefix = []
        for step in range(start, STEPS):
            stats, elapsed, inputs, shared = {}, {}, {}, {}
            for name in METHODS[step % 2 :] + METHODS[: step % 2]:
                p, candidates, annotation = training_batch(data, step, name)
                inputs[name] = digest((p, candidates, annotation))
                shared[name] = digest((p, annotation))
                # Both arms receive the same original inputs and canonical candidate family.
                tick = time.perf_counter()
                stats[name] = update(arms[name], p, candidates, annotation)
                elapsed[name] = time.perf_counter() - tick
            assert len(set(shared.values())) == 1
            assert len(set(inputs.values())) == 1
            reference = control_records[step]
            assert inputs["normalized"] == reference["per_arm_input_sha256"]["plain"]
            assert shared["normalized"] == reference["shared_geometry_sha256"]["plain"]
            row = dict(
                step=step + 1,
                **selection(step),
                metrics=stats,
                seconds=elapsed,
                input_sha256=digest(tuple(inputs[m] for m in METHODS)),
                per_arm_input_sha256=inputs,
                shared_geometry_sha256=shared,
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step < 6:
                prefix.append(stats)
            if step < 6 or (step + 1) % 100 == 0:
                write(out / "progress.json", dict(step=step + 1, metrics=stats))
                print(dict(step=step + 1), flush=True)
            if step == 5:
                replay = make_arms(root)
                metric_audit = []
                for k in range(6):
                    actual = {m: update(replay[m], *training_batch(data, k, m)) for m in METHODS}
                    metric_audit.append(
                        {m: replay_comparison(actual[m], prefix[k][m], "prior") for m in METHODS}
                    )
                expected = lineart_state.payload(arms, 6)["arms"]
                actual = lineart_state.payload(replay, 6)["arms"]
                write(
                    root / "prefix-audit.json",
                    dict(
                        passed=True,
                        metrics=metric_audit,
                        states={
                            m: replay_comparison(actual[m], expected[m], "prior") for m in METHODS
                        },
                    ),
                )
            if step + 1 in SAVED:
                lineart_state.save(saved, arms, step + 1)
                restored = make_arms(root)
                assert lineart_state.restore(saved, restored, step + 1) == step + 1
                exact(
                    lineart_state.payload(arms, step + 1), lineart_state.payload(restored, step + 1)
                )
                snapshot(root, arms, step + 1)
                write(out / f"restore-{step + 1}.json", dict(full_state_exact=True))
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
                    actual, config = load_inference(folder)
                    assert config == arm["config"]
                    exact(nnx.state(expected, nnx.Param), nnx.state(actual, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=8, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "train"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="training"))
            train(root)
    elif a.action == "freeze":
        freeze(root)
    else:
        prepare(root)
