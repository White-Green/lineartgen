"""Train the consolidated physical/L4/support objective from fresh initialization."""

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
from campaign_protocol import read, sha, write
from context_dropout_objective import context_mask
from cue_capacity import make_arm
from cue_preserving_data import MODES
from diagnose_learnability import freeze, verify
from flax import nnx
from forced_context_objective import forced_gradient
from joint_geometry_study import batch as parent_batch
from joint_geometry_study import load_data as parent_load_data
from joint_geometry_study import select as parent_select
from physical_field_model import convert, infer, load_field, save_field
from plausible_study import digest, exact
from scale_dilation_model import infer_dilation
from scale_style_balance_study import update
from support_merge_data import merge_geometry as support_geometry

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = ("fresh",)
REPORT_METHODS = ("control", "fresh")
CONTROL = Path("../jax-forced-geometry-20261015/runs/forced-geometry-20261015")
SUPPORT = Path("../jax-support-geometry-20261015/runs/support-geometry-20261015")
AUDIT = Path("../jax-scale-dilation-20261015/runs/scale-dilation-20261015")
CACHE_SOURCE = Path("../jax-merge-geometry-20261015/runs/merge-geometry-20261015")
DIAGNOSTIC = Path("../jax-width-response-20261015/runs/width-response-20261015")
PARAMETERS = {m: 16402 for m in METHODS}
STEPS = 24000
SAVED = (0, 6, *range(2000, STEPS + 1, 2000))
SEED = 2026101440
EVAL_SEED = 2026101503
PARENT = Path("../jax-joint-geometry-20261015/runs/joint-geometry-20261015")
DOCUMENT = Path("docs/consolidated-geometry-plan-20261015.md")


def replay_state(actual, expected):
    """Warm ROCm replay envelope established before this fresh diagnostic run."""
    aa, sa = jax.tree.flatten(actual)
    bb, sb = jax.tree.flatten(expected)
    assert sa == sb
    maximum = 0.0
    for a, b in zip(aa, bb, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        if np.issubdtype(a.dtype, np.inexact):
            np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-4, equal_nan=False)
            maximum = max(maximum, float(np.max(abs(a - b), initial=0)))
        else:
            np.testing.assert_array_equal(a, b)
    return dict(
        passed=True,
        max_abs_difference=maximum,
        atol=5e-6,
        rtol=1e-4,
        exact=all(np.array_equal(a, b) for a, b in zip(aa, bb, strict=True)),
    )


def select(step):
    return parent_select(step)


def batch(data, step, method):
    prepared, targets, directions = parent_batch(data, step, "mixed")
    keep = context_mask(step % 18000, len(prepared.guides), 4, enabled=True)
    return prepared, (*targets, jnp.asarray(keep)), directions


def load_data(root):
    data = parent_load_data(root)
    data["merge_geometry"] = [support_geometry(r) for r in data["records"]]
    return data


def arms_from_root(root):
    arms = {}
    for method in METHODS:
        model, cfg = load_field(root / f"initials/{method}")
        assert cfg == load_config(root / f"configs/{method}.json")
        arms[method] = make_arm(cfg, model)
        arms[method]["gradient"] = forced_gradient()
    return arms


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            model = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(model, arm["ema"])
            save_field(
                model,
                arm["config"],
                root / f"snapshots/step-{step}/{name}/{mode}",
                provenance=f"regional geometry {name} step{step} {mode}; training-only diagnostic",
            )


def prepare(root):
    parent = PARENT.resolve()
    assert read(DIAGNOSTIC / "completion.json")["stage"] == "complete_width_response"
    assert read(parent / "completion.json")["stage"] == "complete_joint_geometry"
    assert read(CACHE_SOURCE / "completion.json")["stage"] == "complete_merge_geometry"
    assert read(CONTROL / "completion.json")["stage"] == "complete_forced_geometry"
    assert read(AUDIT / "completion.json")["stage"] == "complete_scale_dilation"
    assert read(SUPPORT / "completion.json")["stage"] == "complete_support_geometry"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copytree(parent / "snapshots/step-18000/mixed/raw", root / "reference")
    model, cfg = load_field(parent / "initials/mixed")
    train = cfg.train.model_copy(
        update=dict(
            steps=STEPS,
            learning_rate_decay_steps=STEPS,
            seed=SEED,
            learning_rate=5e-4,
            minimum_learning_rate=1e-4,
            warmup_steps=0,
        )
    )
    cfg = cfg.model_copy(update=dict(train=train))
    assert cfg.train.peak_learning_rate == 1e-3
    model = convert(model, cfg)
    for method in METHODS:
        save_field(
            model,
            cfg,
            root / f"initials/{method}",
            provenance="Fresh EI/EQ group6; physical dilation; L4; supported teacher",
        )
        loaded, cc = load_field(root / f"initials/{method}")
        assert cc == cfg
        exact(nnx.state(model, nnx.Param), nnx.state(loaded, nnx.Param))
        save_config(cfg, root / f"configs/{method}.json")
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "fields.npy",
        "directions.npy",
        "cache-audit.json",
    ):
        (root / name).symlink_to((parent / name).resolve())
    for name in ("merged-fields.npy", "merge-cache.json"):
        (root / name).symlink_to((SUPPORT / name).resolve())
    dest = root / f"snapshots/step-{STEPS}"
    dest.mkdir(parents=True)
    (dest / "control").symlink_to(
        (CONTROL / "snapshots/step-6000/supported").resolve(), target_is_directory=True
    )
    bound = {}
    for name in ("data", "stress-data"):
        (root / name).symlink_to((parent / name).resolve(), target_is_directory=True)
        for p in (parent / name).rglob("*"):
            if p.is_file():
                bound[str(p.resolve())] = sha(p)
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(parent / "scene512" / name, root / "scene512" / name)
    write(root / "sampling-plan.json", [dict(step=k, **select(k)) for k in range(STEPS)])
    write(
        root / "train.json",
        dict(
            cases=512,
            steps=STEPS,
            seed=SEED,
            parameters=PARAMETERS,
            augmentation="D4 + 25/75 base/rough; wide sources 75 percent loguniform4..32px",
            modes=MODES,
            fresh_initialization=True,
            same_original_initial_parameters=True,
            same_inputs_original_targets=True,
            late_context_masks_exact=True,
            only_change="consolidated rules throughout 24k; one fresh optimizer schedule",
            reused_control=str(CONTROL),
            objective="merged objective; union of opaque component fields",
            reference="EQ fresh initial; same 24k input trajectory; EZ supported final control",
            optimizer_initialized_fresh=True,
        ),
    )
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    bound.update({str(p): sha(p) for p in root.rglob("*") if p.is_file()})
    for p in (
        CACHE_SOURCE / "plan.json",
        CACHE_SOURCE / "source.json",
        CACHE_SOURCE / "sampling-plan.json",
        CACHE_SOURCE / "configs/merge.json",
        DIAGNOSTIC / "completion.json",
        DIAGNOSTIC / "evaluation/metrics.json",
    ):
        bound[str(p.resolve())] = sha(p)
    for p in (parent / "completion.json", parent / "source.json", DOCUMENT):
        bound[str(p.resolve())] = sha(p)
    bound.update(read(DIAGNOSTIC / "evaluation/metrics.json")["bindings"])
    for folder in (CONTROL, AUDIT, SUPPORT):
        for name in (
            "completion.json",
            "source.json",
            "plan.json",
            "training/sampling.jsonl",
            "snapshot-audit.json",
            "prefix-audit.json",
        ):
            p = folder / name
            if p.exists():
                bound[str(p.resolve())] = sha(p)
    for p, h in read(CONTROL / "snapshot-audit.json")["bindings"].items():
        assert sha(Path(p)) == h
        bound[str(Path(p).resolve())] = h
    for p in (parent / "initials/mixed").iterdir():
        if p.is_file():
            bound[str(p.resolve())] = sha(p)
    initial = root / "initials/fresh"
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir()},
            data={},
            bound_files=bound,
            methods=METHODS,
            steps=STEPS,
            saved=SAVED,
            plan_document_sha256=sha(DOCUMENT),
            adoption=False,
            parent=str(parent),
            seed=SEED,
            outputs=9516,
            native_pages=34,
            fresh_confirmation=False,
        ),
    )


def check(root):
    verify(root)
    p = read(root / "plan.json")
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f


def train(root, until=STEPS):
    assert until in SAVED[2:]
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    assert read(root / "preflight.json")["passed"]
    data, arms = load_data(root), arms_from_root(root)
    early = [json.loads(s) for s in (PARENT / "training/sampling.jsonl").read_text().splitlines()]
    late = [json.loads(s) for s in (CONTROL / "training/sampling.jsonl").read_text().splitlines()]
    assert len(early) == 18000 and len(late) == 6000
    reference, _ = load_field(PARENT / "initials/mixed")
    for arm in arms.values():
        exact(nnx.state(reference, nnx.Param), nnx.state(arm["model"], nnx.Param))
    initial_predictions = {}
    for k in range(6):
        prepared, _, _ = batch(data, k, "fresh")
        a = np.asarray(infer(arms["fresh"]["model"], prepared))
        b = np.asarray(infer_dilation(reference, prepared, mode="physical"))
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
        initial_predictions[str(k + 1)] = digest(a)
    write(
        root / "initial-audit.json",
        dict(
            fresh_parameters_exact=True,
            six_depth_diagnostic_functions_exact=True,
            prediction_sha256=initial_predictions,
        ),
    )
    for method, arm in arms.items():
        assert (
            sum(x.size for x in jax.tree.leaves(nnx.state(arm["model"], nnx.Param)))
            == PARAMETERS[method]
        )
    write(
        root / "backend.json",
        dict(
            device=jax.devices()[0].device_kind,
            parameters=PARAMETERS,
            flags=os.environ.get("XLA_FLAGS", ""),
        ),
    )
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            rows = [json.loads(s) for s in journal.read_text().splitlines()]
            rows = [r for r in rows if r["step"] <= start]
            assert [r["step"] for r in rows] == list(range(1, start + 1))
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if start == 0:
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        prefix = []
        assert start < until <= STEPS
        for step in range(start, until):
            stats, inputs, elapsed, shared, payloads, flags = {}, {}, {}, {}, {}, {}
            for method in METHODS[:: 1 if step % 2 == 0 else -1]:
                p, c, a = batch(data, step, method)
                inputs[method] = digest((p, c[0], a))
                shared[method] = digest((c[0], a, p.noise_levels))
                payloads[method] = digest(c[:3])
                flags[method] = np.asarray(c[2]).tolist()
                tick = time.perf_counter()
                stats[method] = update(arms[method], p, c, a)
                elapsed[method] = time.perf_counter() - tick
            old = early[step] if step < 18000 else late[step - 18000]
            original_method = "mixed" if step < 18000 else "supported"
            for values, key in ((inputs, "input_sha256"), (shared, "shared_target_sha256")):
                assert values["fresh"] == old[key][original_method]
            assert stats["fresh"]["merged_fraction"] == sum(flags["fresh"]) / 4
            keep = context_mask(step % 18000, select(step)["depth"], 4, enabled=True).tolist()
            if step >= 18000:
                assert payloads["fresh"] == old["candidate_payload_sha256"]["supported"]
                assert flags["fresh"] == old["merge_eligible"]["supported"]
                assert keep == old["context_keep"]
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
                merge_eligible=flags,
                context_keep=keep,
                reference_phase="EQ" if step < 18000 else "EZ",
                candidate_payload_sha256=payloads,
                input_sha256=inputs,
                shared_target_sha256=shared,
                seconds=elapsed,
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step < 6:
                prefix.append(stats)
            if step < 6 or (step + 1) % 100 == 0:
                write(out / "progress.json", dict(step=step + 1, metrics=stats))
                print(dict(step=step + 1), flush=True)
            if step == 5:
                replay = arms_from_root(root)
                audits = []
                for k in range(6):
                    actual = {m: update(replay[m], *batch(data, k, m)) for m in METHODS}
                    audits.append({m: replay_state(actual[m], prefix[k][m]) for m in METHODS})
                expected, actual = (lineart_state.payload(x, 6)["arms"] for x in (arms, replay))
                write(
                    root / "prefix-audit.json",
                    dict(
                        passed=True,
                        metrics=audits,
                        states={m: replay_state(actual[m], expected[m]) for m in METHODS},
                    ),
                )
            if step + 1 in SAVED:
                lineart_state.save(saved, arms, step + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(saved, restored, step + 1) == step + 1
                exact(
                    lineart_state.payload(arms, step + 1), lineart_state.payload(restored, step + 1)
                )
                snapshot(root, arms, step + 1)
                write(out / f"restore-{step + 1}.json", dict(full_state_exact=True))
        if until < STEPS:
            write(
                root / "workflow-status.json", dict(stage="training_segment_complete", step=until)
            )
            return
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(saved, restored, step) == step
            for method, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                    folder = root / f"snapshots/step-{step}/{method}/{mode}"
                    actual, cfg = load_field(folder)
                    assert cfg == arm["config"]
                    exact(expected, nnx.state(actual, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(
            root / "snapshot-audit.json",
            dict(raw_ema_exact=True, exports=2 * len(SAVED), bindings=bindings),
        )
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--until", type=int, default=STEPS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root)
    elif args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, until=args.until)
    else:
        globals()[args.action](root)
