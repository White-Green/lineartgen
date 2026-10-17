"""Equal-size pooled residual mean/choice models with frequent permitted axes."""

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
from plausible_study import digest, exact
from pooled_axis_batches import GV, batch, load_data
from pooled_axis_model import add_readout, load_field, save_field
from pooled_axis_objective import axis_gradient
from pooled_procedural_study import GM, GO
from procedural_shape_study import AMBIGUITY as AMBIGUITY
from procedural_shape_study import DIAGNOSTIC as DIAGNOSTIC
from procedural_shape_study import EVAL_SEED as EVAL_SEED
from procedural_shape_study import GK as GK
from procedural_shape_study import MODELREF as MODELREF
from procedural_shape_study import select
from scale_style_balance_study import update

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

GP = Path("../jax-pooled-procedural-20261017/runs/pooled-procedural-20261017-retry1").resolve()
GT = Path("../jax-pooled-noopening-20261017/runs/pooled-noopening-20261017").resolve()
GU = Path("../jax-one-sided-opening-20261017/runs/one-sided-opening-20261017").resolve()
DOC = Path("docs/pooled-axis-plan-20261017.md")
METHODS = REPORT_METHODS = ("mean", "choice")
STEPS = 6000
SAVED = (0, 6, 36, *range(1000, 6001, 1000))


def arms_from_root(root):
    arms = {}
    for name in METHODS:
        model, cfg = load_field(root / f"initials/{name}")
        assert cfg == load_config(root / "config.json") and model.head.mode == name
        arms[name] = make_arm(cfg, model)
        arms[name]["gradient"] = axis_gradient(prior_mode="none")
    return arms


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    for folder, stage in (
        (GT, "complete_pooled_noopening"),
        (GU, "complete_one_sided_opening_fit"),
        (GV, "complete_ambiguous_shape_data_preview"),
    ):
        assert read(folder / "completion.json")["stage"] == stage
    root.mkdir(parents=True, exist_ok=False)
    (root / "diagnostics").mkdir()
    base, cfg = load_field(GP / "initials/pooled")
    assert (
        cfg.train.steps == 6000
        and cfg.train.peak_learning_rate == 2e-4
        and cfg.train.batch_size == 4
    )
    for name in METHODS:
        model = add_readout(base, cfg, mode=name)
        assert sum(v.size for v in jax.tree.leaves(nnx.state(model, nnx.Param))) == 21890
        state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
        residual = state["head"].pop("residual")
        assert all(np.count_nonzero(np.asarray(v)) == 0 for v in jax.tree.leaves(residual))
        exact(state, nnx.to_pure_dict(nnx.state(base, nnx.Param)))
        save_field(
            model,
            cfg,
            root / f"initials/{name}",
            provenance="Untouched FW mixed6000 + zero pooled cue + zero residual; research only",
        )
        restored, cc = load_field(root / f"initials/{name}")
        exact(nnx.state(restored, nnx.Param), nnx.state(model, nnx.Param))
        assert cc == cfg
    save_config(cfg, root / "config.json")
    arms = arms_from_root(root)
    exact(
        nnx.state(arms["mean"]["model"], nnx.Param), nnx.state(arms["choice"]["model"], nnx.Param)
    )
    shutil.copytree(GM / "reference", root / "reference")
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
    (root / "diagnostics/orientation").mkdir()
    shutil.copy2(
        Path(
            "../jax-residual-axis-20261016/runs/residual-axis-20261016/diagnostics/orientation/square48-input.npz"
        ),
        root / "diagnostics/orientation/square48-input.npz",
    )
    files = [p for p in root.rglob("*") if p.is_file()]
    files += [Path(f) for f in read(GP / "plan.json")["bindings"]]
    for folder in (GP, GT, GU, GV):
        files += [folder / "completion.json"]
        if folder != GV:
            files += [folder / "diagnostics/supplemental-binding.json"]
    files += [
        GT / "diagnostics/decision.json",
        GU / "decision.json",
        GV / "decision.json",
        GV / "recipes.json",
        GT / "training/sampling.jsonl",
        GP / "training/sampling.jsonl",
    ]
    files += list((GP / "initials/pooled").iterdir()) + [GO / "inputs.npz", DOC.resolve()]
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
            parameters={m: 21890 for m in METHODS},
            trained_arms=2,
            prior_mode="none",
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
                provenance=f"GW pooled axis {name} {step} {mode}; research only",
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
            assert len(rows) == start, (
                "Preserve interruption evidence and explicitly audit replay before retry"
            )
            assert [r["step"] for r in rows] == list(range(1, start + 1))
        if start == 0:
            lineart_state.save(manager, arms, 0)
            snapshot(root, arms, 0)
        assert start < until
        for k in range(start, until):
            p, c, d = batch(data, k)
            metrics, seconds = {}, {}
            for name in METHODS:
                tick = time.perf_counter()
                metrics[name] = update(arms[name], p, c, d)
                seconds[name] = time.perf_counter() - tick
                assert all(np.isfinite(v) for v in metrics[name].values())
            row = dict(
                step=k + 1,
                **select(k),
                metrics=metrics,
                seconds=seconds,
                input_sha256=digest((p, c[0], d)),
                candidate_payload_sha256=digest(c),
                context_keep=np.asarray(c[3]).tolist(),
                audit=data["last_audit"],
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k == 35:
                replay = arms_from_root(root)
                for j in range(36):
                    for name in METHODS:
                        exact(update(replay[name], *batch(data, j)), rows[j]["metrics"][name])
                exact(lineart_state.payload(replay, 36), lineart_state.payload(arms, 36))
                write(
                    root / "prefix-audit.json",
                    dict(passed=True, metrics_exact=True, full_state_exact=True, steps=36),
                )
            if k + 1 in SAVED:
                lineart_state.save(manager, arms, k + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(manager, restored, k + 1) == k + 1
                exact(lineart_state.payload(restored, k + 1), lineart_state.payload(arms, k + 1))
                snapshot(root, arms, k + 1)
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
            if k < 36 or (k + 1) % 100 == 0:
                write(out / "progress.json", dict(step=k + 1, metrics=metrics))
                print(dict(step=k + 1), flush=True)
        if until < STEPS:
            return
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(manager, restored, step) == step
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    model, cfg = load_field(folder)
                    assert cfg == arm["config"]
                    exact(expected, nnx.state(model, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=36, bindings=bindings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--until", type=int, default=STEPS)
    a = parser.parse_args()
    root = a.root.resolve()
    if a.action == "prepare":
        prepare(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, a.until)
