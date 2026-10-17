"""Exact GM procedural batches with GA's zero-initialized broad spatial cue."""

import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import make_arm
from flax import nnx
from opening_prior_objective import opening_gradient
from plausible_study import digest, exact
from pooled_context_model import add_pool, load_field, save_field
from procedural_shape_study import AMBIGUITY as AMBIGUITY
from procedural_shape_study import DIAGNOSTIC as DIAGNOSTIC
from procedural_shape_study import EVAL_SEED as EVAL_SEED
from procedural_shape_study import GK as GK
from procedural_shape_study import MODELREF as MODELREF
from procedural_shape_study import batch as gm_batch
from procedural_shape_study import load_data as gm_load_data
from procedural_shape_study import select
from scale_style_balance_study import update

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

GM = Path(
    "../jax-procedural-shape-training-20261017/runs/procedural-shape-training-20261017"
).resolve()
GO = Path(
    "../jax-wide-context-identifiability-20261017/runs/wide-context-identifiability-20261017"
).resolve()
GA = Path("../jax-pooled-context-20261016").resolve()
DOC = Path("docs/pooled-procedural-plan-20261017.md")
METHODS = REPORT_METHODS = ("procedural", "pooled")
TRAINED = ("pooled",)
STEPS = 6000
SAVED = (0, 6, 24, *range(1000, 6001, 1000))


def load_data(root=None):
    data = gm_load_data(GM)
    data["gm_rows"] = [
        json.loads(s) for s in (GM / "training/sampling.jsonl").read_text().splitlines()
    ]
    return data


def batch(data, step):
    p, c, d = gm_batch(data, step, "procedural")
    expected = data["gm_rows"][step]
    assert digest((p, c[0], d)) == expected["input_sha256"]["procedural"]
    assert digest(c[:3]) == expected["candidate_payload_sha256"]["procedural"]
    assert np.asarray(c[3]).tolist() == expected["context_keep"]
    return p, c, d


def arms_from_root(root):
    model, cfg = load_field(root / "initials/pooled")
    assert cfg == load_config(root / "config.json")
    arm = make_arm(cfg, model)
    arm["gradient"] = opening_gradient()
    return dict(pooled=arm)


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(GM / "completion.json")["stage"] == "complete_procedural_shape_training"
    assert read(GO / "completion.json")["stage"] == "complete_wide_context_identifiability"
    root.mkdir(parents=True, exist_ok=False)
    (root / "diagnostics").mkdir()
    model, cfg = load_field(GM / "initials/procedural")
    assert sha(Path("scripts/pooled_context_model.py")) == sha(
        GA / "scripts/pooled_context_model.py"
    )
    pooled = add_pool(model, cfg)
    state = nnx.to_pure_dict(nnx.state(pooled, nnx.Param))
    added = state.pop("pooled_stem")
    exact(state, nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    assert all(np.count_nonzero(np.asarray(v)) == 0 for v in jax.tree.leaves(added))
    assert sum(v.size for v in jax.tree.leaves(nnx.state(pooled, nnx.Param))) == 21778
    assert (
        cfg.train.steps == 6000
        and cfg.train.peak_learning_rate == 2e-4
        and cfg.train.batch_size == 4
    )
    save_field(
        pooled,
        cfg,
        root / "initials/pooled",
        provenance="Untouched FW mixed6000 plus zero GA pooled cue; research only",
    )
    restored, cc = load_field(root / "initials/pooled")
    exact(nnx.state(pooled, nnx.Param), nnx.state(restored, nnx.Param))
    assert cc == cfg
    save_config(cfg, root / "config.json")
    shutil.copytree(GM / "reference", root / "reference")
    shutil.copytree(
        GM / "snapshots/step-6000/procedural/raw", root / "snapshots/step-6000/procedural/raw"
    )
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "data",
        "stress-data",
        "teacher-preview",
        "source-preview",
        "width-range-audit.json",
        "width-plan.json",
        "probe.json",
    ):
        (root / name).symlink_to((GM / name).resolve(), target_is_directory=(GM / name).is_dir())
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(GM / "scene512" / name, root / "scene512" / name)
    files = [p for p in root.rglob("*") if p.is_file()]
    files += [Path(f) for f in read(GM / "plan.json")["bound_files"]]
    files += [
        GM / f
        for f in (
            "completion.json",
            "diagnostics/supplemental-binding.json",
            "training/sampling.jsonl",
            "snapshot-audit.json",
        )
    ]
    files += [GO / f for f in ("completion.json", "decision.json", "inputs.npz", "metrics.json")]
    files += [p for p in (GM / "initials/procedural").iterdir() if p.is_file()]
    files += [
        GA / "scripts/pooled_context_model.py",
        GA / "tests/test_pooled_context_model.py",
        DOC.resolve(),
    ]
    files += [
        Path(f).resolve()
        for f in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if f.endswith((".py", ".toml")) or f == "uv.lock"
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            bindings={str(p.resolve()): sha(p) for p in files},
            archive_sha256=sha(root / "source.tar.gz"),
            steps=STEPS,
            saved=SAVED,
            parameters=dict(procedural=21718, pooled=21778),
            trained_arms=1,
            reused_control="GM procedural raw6000",
            adoption=False,
            fresh_confirmation=False,
        ),
    )


def check(root):
    for f, h in read(root / "plan.json")["bindings"].items():
        assert sha(Path(f)) == h, f
    assert sha(root / "source.tar.gz") == read(root / "plan.json")["archive_sha256"]


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
                provenance=f"Pooled procedural {step} {mode}; research only",
            )


def train(root, until):
    check(root)
    assert jax.devices()[0].platform != "cpu" and until in SAVED[3:]
    assert read(root / "test-results.json")["passed"] and read(root / "preflight.json")["passed"]
    data, arms = load_data(), arms_from_root(root)
    out = root / "training"
    out.mkdir(exist_ok=True)
    rows = []
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as manager:
        start = lineart_state.restore(manager, arms) if manager.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            rows = [json.loads(s) for s in journal.read_text().splitlines()]
            if any(r["step"] > start for r in rows):
                write(out / f"interrupted-{time.time_ns()}.json", rows)
            rows = [r for r in rows if r["step"] <= start]
            assert [r["step"] for r in rows] == list(range(1, start + 1))
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if start == 0:
            lineart_state.save(manager, arms, 0)
            snapshot(root, arms, 0)
        assert start < until
        for k in range(start, until):
            p, c, d = batch(data, k)
            tick = time.perf_counter()
            metrics = update(arms["pooled"], p, c, d)
            seconds = time.perf_counter() - tick
            assert all(np.isfinite(v) for v in metrics.values())
            row = dict(
                step=k + 1,
                **select(k),
                metrics=dict(pooled=metrics),
                seconds=dict(pooled=seconds),
                input_sha256=digest((p, c[0], d)),
                candidate_payload_sha256=digest(c[:3]),
                context_keep=np.asarray(c[3]).tolist(),
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k == 23:
                replay = arms_from_root(root)
                for j in range(24):
                    exact(update(replay["pooled"], *batch(data, j)), rows[j]["metrics"]["pooled"])
                exact(lineart_state.payload(replay, 24), lineart_state.payload(arms, 24))
                write(
                    root / "prefix-audit.json",
                    dict(passed=True, metrics_exact=True, full_state_exact=True, steps=24),
                )
            if k + 1 in SAVED:
                lineart_state.save(manager, arms, k + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(manager, restored, k + 1) == k + 1
                exact(lineart_state.payload(restored, k + 1), lineart_state.payload(arms, k + 1))
                snapshot(root, arms, k + 1)
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
            if k < 24 or (k + 1) % 100 == 0:
                write(out / "progress.json", dict(step=k + 1, metrics=metrics))
                print(dict(step=k + 1), flush=True)
        if until < STEPS:
            return
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(manager, restored, step) == step
            arm = restored["pooled"]
            for mode in ("raw", "ema"):
                expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                folder = root / f"snapshots/step-{step}/pooled/{mode}"
                model, cfg = load_field(folder)
                assert cfg == arm["config"]
                exact(expected, nnx.state(model, nnx.Param))
                bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=18, bindings=bindings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--until", type=int, default=STEPS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, args.until)
