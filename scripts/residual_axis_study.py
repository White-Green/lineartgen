"""Matched residual readouts on identical whole-axis permissive teacher batches."""

import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import jax
import numpy as np
from axis_family_data import axis_payload
from axis_family_objective import axis_gradient
from campaign_protocol import read, sha, write
from choice_readout_study import AMBIGUITY as AMBIGUITY
from choice_readout_study import (
    CONTROL,
    LOCALCONTROL,
    REUSED,
    SEED,
    SOURCE,
    select,
)
from choice_readout_study import DIAGNOSTIC as DIAGNOSTIC
from choice_readout_study import EVAL_SEED as EVAL_SEED
from choice_readout_study import batch as old_batch
from choice_readout_study import load_data as old_load_data
from consolidated_geometry_study import replay_state
from cue_capacity import make_arm
from diagnose_learnability import freeze, verify
from flax import nnx
from plausible_study import digest, exact
from residual_axis_model import add_readout, load_field, save_field
from scale_style_balance_study import update

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = ("mean", "choice")
REPORT_METHODS = METHODS
PARAMETERS = {m: 21830 for m in METHODS}
STEPS = 6000
SAVED = (0, 6, *range(1000, STEPS + 1, 1000))
DOCUMENT = Path("docs/residual-axis-plan-20261016.md")
MODELREF = LOCALCONTROL
GC = Path("../jax-choice-readout-20261016/runs/choice-readout-20261016-retry2")
GF = Path("../jax-residual-choice-fit-20261016/runs/residual-choice-fit-20261016-retry1")
GG = Path("../jax-whole-axis-family-20261016/runs/whole-axis-family-preview-20261016")


def load_data(root):
    data = old_load_data(root)
    data["axis_widths"] = read(root / "width-plan.json")
    return data


def batch(data, step, method):
    assert method in METHODS
    if data.get("cached_axis_step") == step:
        data["last_audit"] = data["cached_axis_audit"]
        return data["cached_axis_batch"]
    p, c, d = old_batch(data, step, "choice")
    a, records = axis_payload(data, select(step), data["axis_widths"][step]["records"], c, d)
    assert all(not r["allowed"] or select(step)["modes"][r["item"]] != 2 for r in records)
    data["last_audit"] = dict(data["last_audit"], axis_records=records)
    result = p, (*c, *a), d
    data["cached_axis_step"] = step
    data["cached_axis_batch"] = result
    data["cached_axis_audit"] = data["last_audit"]
    return result


def arms_from_root(root):
    arms = {}
    for m in METHODS:
        model, cfg = load_field(root / f"initials/{m}")
        assert cfg == load_config(root / f"configs/{m}.json")
        assert model.head.mode == m
        arms[m] = make_arm(cfg, model)
        arms[m]["gradient"] = axis_gradient()
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
                provenance=f"residual-axis {name} step{step} {mode}; research only",
            )


def prepare(root):
    assert read(GF / "completion.json")["stage"] == "complete_residual_choice_fit"
    assert read(GF / "decision.json")["mechanism_passed"]
    assert read(GG / "completion.json")["stage"] == "complete_whole_axis_family_preview"
    assert read(GG / "decision.json")["teacher_preview_passed"]
    assert read(GC / "completion.json")["stage"] == "complete_choice_readout"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    source = LOCALCONTROL / "snapshots/step-6000/mixed/raw"
    shutil.copytree(source, root / "reference")
    base, cfg = load_field(source)
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=STEPS,
                    learning_rate_decay_steps=STEPS,
                    seed=SEED,
                    learning_rate=1e-4,
                    minimum_learning_rate=2e-5,
                    warmup_steps=0,
                )
            )
        )
    )
    assert cfg.train.peak_learning_rate == 2e-4 and cfg.train.batch_size == 4
    for name in METHODS:
        model = add_readout(base, cfg, mode=name)
        save_field(
            model,
            cfg,
            root / f"initials/{name}",
            provenance="Untouched FW mixed6000 plus zero residual; research only",
        )
        restored, cc = load_field(root / f"initials/{name}")
        exact(nnx.state(model, nnx.Param), nnx.state(restored, nnx.Param))
        assert (
            cc == cfg
            and sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
            == PARAMETERS[name]
        )
        state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
        residual = state["head"].pop("residual")
        assert all(np.count_nonzero(x) == 0 for x in jax.tree.leaves(residual))
        exact(state, nnx.to_pure_dict(nnx.state(base, nnx.Param)))
        save_config(cfg, root / f"configs/{name}.json")
    arms = arms_from_root(root)
    exact(
        nnx.state(arms["mean"]["model"], nnx.Param), nnx.state(arms["choice"]["model"], nnx.Param)
    )
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "fields.npy",
        "directions.npy",
        "cache-audit.json",
        "merged-fields.npy",
        "merge-cache.json",
        "data",
        "stress-data",
    ):
        (root / name).symlink_to(
            (CONTROL / name).resolve(), target_is_directory=(CONTROL / name).is_dir()
        )
    for folder in ("source-preview", "teacher-preview"):
        shutil.copytree(GC / folder, root / folder)
    shutil.copytree(GG, root / "axis-preview", ignore=shutil.ignore_patterns("source.tar.gz"))
    for name in (
        "sampling-plan.json",
        "width-plan.json",
        "frequency-plan.json",
        "width-range-audit.json",
        "probe.json",
    ):
        shutil.copy2(GC / name, root / name)
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(GC / "scene512" / name, root / "scene512" / name)
    (root / "diagnostics/orientation").mkdir(parents=True)
    shutil.copy2(
        GC / "diagnostics/orientation/square48.npz",
        root / "diagnostics/orientation/square48-input.npz",
    )
    write(
        root / "train.json",
        dict(
            cases=512,
            steps=STEPS,
            seed=SEED,
            parameters=PARAMETERS,
            reference="FW mixed6000 raw",
            initial="FW mixed6000 plus zero112-parameter residual",
            optimizer_initialized_fresh=True,
            teacher_policy="GG whole-only alternative axis",
            eligible_examples=120,
            shared_batch=True,
            adoption=False,
        ),
    )
    files = [p for p in root.rglob("*") if p.is_file()]
    for folder in (GF, GG, GC, LOCALCONTROL, REUSED, SOURCE):
        files += [folder / "completion.json"]
    files += [
        GF / "decision.json",
        GF / "observations.json",
        GG / "decision.json",
        GG / "active-support.json",
        GC / "diagnostics/supplemental-binding.json",
    ]
    for folder in (LOCALCONTROL, REUSED, SOURCE):
        files += [folder / "training/sampling.jsonl", folder / "snapshot-audit.json"]
    files += [p for p in source.iterdir() if p.is_file()]
    files += [Path(f) for f in read(GC / "plan.json")["bound_files"]]
    files += [DOCUMENT]
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initials/mean"),
            initial_files={p.name: sha(p) for p in (root / "initials/mean").iterdir()},
            data={},
            bound_files={str(p.resolve()): sha(p) for p in files},
            methods=METHODS,
            steps=STEPS,
            saved=SAVED,
            outputs=9516,
            native_pages=34,
            adoption=False,
            fresh_confirmation=False,
            plan_document_sha256=sha(DOCUMENT),
        ),
    )


def check(root):
    verify(root)
    assert sha(root / "source.tar.gz") == read(root / "source.json")["archive_sha256"]
    p = read(root / "plan.json")
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f
    assert sha(DOCUMENT) == p["plan_document_sha256"]


def train(root, until=STEPS):
    assert until in SAVED[2:]
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"] and read(root / "preflight.json")["passed"]
    source_review = read(root / "source-preview/visual-review.json")
    assert source_review["reviewed"]
    for f, h in source_review["files"].items():
        assert sha(root / "source-preview" / f) == h
    review = read(root / "teacher-preview/visual-review.json")
    assert review["reviewed"] and len(review["files"]) == 2
    for f, h in review["files"].items():
        assert sha(root / "teacher-preview" / f) == h
    data, arms = load_data(root), arms_from_root(root)
    write(root / "backend.json", dict(device=jax.devices()[0].device_kind, parameters=PARAMETERS))
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            rows = [json.loads(x) for x in journal.read_text().splitlines()]
            if any(r["step"] > start for r in rows):
                write(out / f"interrupted-{time.time_ns()}.json", rows)
            rows = [r for r in rows if r["step"] <= start]
            assert [r["step"] for r in rows] == list(range(1, start + 1))
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if start == 0:
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        prefix = []
        assert start < until <= STEPS
        for step in range(start, until):
            stats, inputs, shared, payloads, flags, elapsed = {}, {}, {}, {}, {}, {}
            p, c, d = batch(data, step, METHODS[0])
            for m in METHODS[:: 1 if step % 2 == 0 else -1]:
                inputs[m] = digest((p, c[0], d))
                shared[m] = digest((c[0], d, p.noise_levels))
                payloads[m] = digest(c[:3])
                flags[m] = np.asarray(c[2]).tolist()
                tick = time.perf_counter()
                stats[m] = update(arms[m], p, c, d)
                elapsed[m] = time.perf_counter() - tick
            assert len(set(inputs.values())) == 1 and len(set(payloads.values())) == 1
            for m in METHODS:
                assert stats[m]["merged_fraction"] == sum(flags[m]) / 4
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
                merge_eligible=flags,
                context_keep=np.asarray(c[3]).tolist(),
                candidate_payload_sha256=payloads,
                input_sha256=inputs,
                shared_target_sha256=shared,
                seconds=elapsed,
                teacher_audit=data["last_audit"],
                axis_allowed=np.asarray(c[6]).tolist(),
                axis_payload_sha256=digest(c[4:]),
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
                a, b = (lineart_state.payload(x, 6)["arms"] for x in (replay, arms))
                write(
                    root / "prefix-audit.json",
                    dict(
                        passed=True,
                        metrics=audits,
                        states={m: replay_state(a[m], b[m]) for m in METHODS},
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
            for m, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                    folder = root / f"snapshots/step-{step}/{m}/{mode}"
                    actual, cfg = load_field(folder)
                    assert cfg == arm["config"]
                    exact(expected, nnx.state(actual, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=32, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "train"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--until", type=int, default=STEPS)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root)
    elif args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, args.until)
    else:
        prepare(root)
