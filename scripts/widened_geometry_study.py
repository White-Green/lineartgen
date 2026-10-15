"""Compare fixed-output warm width6 and width12 using identical next training batches."""

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
from consolidated_geometry_study import replay_state, snapshot
from context_dropout_objective import context_mask
from cue_capacity import make_arm
from diagnose_learnability import freeze, verify
from flax import nnx
from forced_context_objective import forced_gradient
from joint_geometry_study import batch as parent_batch
from joint_geometry_study import load_data as parent_load_data
from joint_geometry_study import select as parent_select
from physical_field_model import load_field, save_field
from plausible_study import digest, exact
from scale_style_balance_study import update
from support_merge_data import merge_geometry
from widen_physical_model import widen

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = REPORT_METHODS = ("control", "wide")
PARAMETERS = dict(control=16402, wide=63892)
PARENT = Path("../jax-joint-geometry-20261015/runs/joint-geometry-20261015")
CONTROL = Path("../jax-forced-geometry-20261015/runs/forced-geometry-20261015")
SUPPORT = Path("../jax-support-geometry-20261015/runs/support-geometry-20261015")
DIAGNOSTIC = Path("../jax-width-response-20261015/runs/width-response-20261015")
DOCUMENT = Path("docs/widened-geometry-plan-20261015.md")
SEED = 2026101440
EVAL_SEED = 2026101503
STEPS = 3000
SAVED = (0, 6, 1000, 2000, STEPS)
OFFSET = 24000


def select(step):
    return parent_select(step + OFFSET)


def batch(data, step, method):
    p, t, d = parent_batch(data, step + OFFSET, "mixed")
    keep = context_mask(step, len(p.guides), 4, enabled=True)
    return p, (*t, jnp.asarray(keep)), d


def load_data(root):
    d = parent_load_data(root)
    d["merge_geometry"] = [merge_geometry(r) for r in d["records"]]
    return d


def arms_from_root(root):
    arms = {}
    for m in METHODS:
        model, cfg = load_field(root / f"initials/{m}")
        assert cfg == load_config(root / f"configs/{m}.json")
        arms[m] = make_arm(cfg, model)
        arms[m]["gradient"] = forced_gradient()
    return arms


def prepare(root):
    assert read(CONTROL / "completion.json")["stage"] == "complete_forced_geometry"
    assert read(PARENT / "completion.json")["stage"] == "complete_joint_geometry"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    source = CONTROL / "snapshots/step-6000/supported/raw"
    shutil.copytree(source, root / "reference")
    model, cfg = load_field(source)
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
    assert cfg.train.peak_learning_rate == 2e-4
    wide, wc = widen(model, cfg)
    for name, m, c in [("control", model, cfg), ("wide", wide, wc)]:
        save_field(
            m,
            c,
            root / f"initials/{name}",
            provenance="EZ supported24k; embedded width12 or width6; research only",
        )
        loaded, cc = load_field(root / f"initials/{name}")
        assert cc == c
        exact(nnx.state(m, nnx.Param), nnx.state(loaded, nnx.Param))
        assert sum(x.size for x in jax.tree.leaves(nnx.state(m, nnx.Param))) == PARAMETERS[name]
        save_config(c, root / f"configs/{name}.json")
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "fields.npy",
        "directions.npy",
        "cache-audit.json",
    ):
        (root / name).symlink_to((PARENT / name).resolve())
    for name in ("merged-fields.npy", "merge-cache.json"):
        (root / name).symlink_to((SUPPORT / name).resolve())
    bound = {}
    for folder in ("data", "stress-data"):
        (root / folder).symlink_to((PARENT / folder).resolve(), target_is_directory=True)
        for p in (PARENT / folder).rglob("*"):
            if p.is_file():
                bound[str(p.resolve())] = sha(p)
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(PARENT / "scene512" / name, root / "scene512" / name)
    write(root / "sampling-plan.json", [dict(step=k, **select(k)) for k in range(STEPS)])
    write(
        root / "train.json",
        dict(
            cases=512,
            steps=STEPS,
            seed=SEED,
            parameters=PARAMETERS,
            offset=OFFSET,
            reference="EZ supported raw6000, total24k",
            same_inputs_targets_masks=True,
            optimizer_initialized_fresh=True,
            only_change="embed width6 function in width12, new outgoing weights zero",
            objective="physical dilation + forced support teacher + L4 + context dropout",
            adoption=False,
        ),
    )
    write(
        root / "probe.json",
        dict(
            cases=256,
            seed=EVAL_SEED,
            modes=("whole01", "whole05", "partial05"),
            draws=2,
            depth=3,
            confirmation=False,
        ),
    )
    bound.update({str(p): sha(p) for p in root.rglob("*") if p.is_file()})
    for folder in (PARENT, CONTROL, SUPPORT):
        for name in ("plan.json", "source.json", "completion.json"):
            p = folder / name
            bound[str(p.resolve())] = sha(p)
    for p in source.iterdir():
        if p.is_file():
            bound[str(p.resolve())] = sha(p)
    bound.update(read(DIAGNOSTIC / "evaluation/metrics.json")["bindings"])
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initials/control"),
            initial_files={p.name: sha(p) for p in (root / "initials/control").iterdir()},
            data={},
            bound_files=bound,
            methods=METHODS,
            steps=STEPS,
            saved=SAVED,
            plan_document_sha256=sha(DOCUMENT),
            adoption=False,
            parent=str(PARENT.resolve()),
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
    assert read(root / "test-results.json")["passed"] and read(root / "preflight.json")["passed"]
    data, arms = load_data(root), arms_from_root(root)
    write(root / "backend.json", dict(device=jax.devices()[0].device_kind, parameters=PARAMETERS))
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            rows = [json.loads(x) for x in journal.read_text().splitlines()]
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
            for m in METHODS[:: 1 if step % 2 == 0 else -1]:
                p, c, d = batch(data, step, m)
                inputs[m] = digest((p, c[0], d))
                shared[m] = digest((c[0], d, p.noise_levels))
                payloads[m] = digest(c[:3])
                flags[m] = np.asarray(c[2]).tolist()
                tick = time.perf_counter()
                stats[m] = update(arms[m], p, c, d)
                elapsed[m] = time.perf_counter() - tick
            for values in (inputs, shared, payloads, flags):
                assert values["control"] == values["wide"]
            for m in METHODS:
                assert stats[m]["merged_fraction"] == sum(flags[m]) / 4
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
                merge_eligible=flags,
                context_keep=context_mask(step, select(step)["depth"], 4, enabled=True).tolist(),
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=20, bindings=bindings))
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
