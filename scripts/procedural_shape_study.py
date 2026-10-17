"""Matched unchanged-model continuation with a frozen procedural data intervention."""

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
from choice_readout_study import AMBIGUITY as AMBIGUITY
from choice_readout_study import CONTROL, LOCALCONTROL, SEED, select
from choice_readout_study import DIAGNOSTIC as DIAGNOSTIC
from choice_readout_study import EVAL_SEED as EVAL_SEED
from choice_readout_study import batch as old_batch
from choice_readout_study import load_data as old_load_data
from consolidated_geometry_study import replay_state
from cue_capacity import make_arm
from diagnose_learnability import freeze, verify
from flax import nnx
from mixed_context_model import load_field, save_field
from opening_prior_objective import opening_gradient
from plausible_study import digest, exact
from procedural_shape_data import record, render, selected
from scale_style_balance_study import update

from lineartgen.config import load_config, save_config
from lineartgen.diffusion import ProgressTrainingInputs, pyramid
from lineartgen.training import checkpoints, lineart_state

METHODS = ("control", "procedural")
REPORT_METHODS = METHODS
PARAMETERS = {m: 21718 for m in METHODS}
STEPS = 6000
SAVED = (0, 6, 24, *range(1000, STEPS + 1, 1000))
DOCUMENT = Path("docs/procedural-shape-training-plan-20261017.md")
MODELREF = LOCALCONTROL
GH = Path("../jax-residual-axis-20261016/runs/residual-axis-20261016")
GJ = Path("../jax-bar-fill-fit-20261017/runs/bar-fill-fit-20261017")
GK = Path("../jax-bar-fill-transfer-20261017/runs/bar-fill-transfer-20261017")
GL = Path("../jax-procedural-shape-data-20261017/runs/procedural-shape-data-20261017")


def load_data(root):
    data = old_load_data(root)
    data["procedural_recipes"] = {(r["step"], r["item"]): r for r in read(GL / "recipes.json")}
    return data


@jax.jit
def procedural_inputs(clean, source, strengths, progress, key):
    depth = len(strengths)
    guides, targets = pyramid(source, depth), pyramid(clean, depth)
    keys = jax.random.split(key, depth)
    noisy = tuple(
        s + 0.25 * t * jax.random.normal(k, s.shape)
        for s, t, k in zip(guides, strengths, keys, strict=True)
    )
    targets = tuple(
        jnp.where(t > 0, y, s) for y, t, s in zip(targets, strengths, guides, strict=True)
    )
    return ProgressTrainingInputs(targets, strengths, noisy, guides, progress)


def batch(data, step, method):
    assert method in METHODS
    if data.get("cached_procedural_step") != step:
        previous = old_batch(data, step, "choice")
        data["last_audit"] = dict(data["last_audit"], procedural_selected=selected(step))
        pair = dict(control=previous, procedural=previous)
        if selected(step):
            p, c, d = previous
            records = [record(step, i) for i in range(4)]
            assert records == [data["procedural_recipes"][step, i] for i in range(4)]
            arrays = [render(r) for r in records]
            source, clean, base, directions = (
                jnp.asarray(np.stack([a[k] for a in arrays]))
                for k in ("source", "clean", "base", "direction")
            )
            prepared = procedural_inputs(
                clean,
                source,
                p.noise_levels,
                p.progress,
                jax.random.fold_in(jax.random.PRNGKey(2026101703), step),
            )
            exact(prepared.noise_levels, p.noise_levels)
            exact(prepared.progress, p.progress)
            payload = (base, base, jnp.zeros(4, dtype=bool), c[3])
            pair["procedural"] = (prepared, payload, directions)
            data["last_audit"] |= dict(
                procedural_records=records,
                procedural_arrays_sha256=digest((source, clean, base, directions)),
            )
        data["cached_procedural_step"] = step
        data["cached_procedural_batches"] = pair
        data["cached_procedural_audit"] = data["last_audit"]
    data["last_audit"] = data["cached_procedural_audit"]
    return data["cached_procedural_batches"][method]


def arms_from_root(root):
    arms = {}
    for m in METHODS:
        model, cfg = load_field(root / f"initials/{m}")
        assert cfg == load_config(root / f"configs/{m}.json")
        arms[m] = make_arm(cfg, model)
        arms[m]["gradient"] = opening_gradient()
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
                provenance=f"procedural-shape {name} step{step} {mode}; research only",
            )


def prepare(root):
    assert read(GL / "completion.json")["stage"] == "complete_procedural_shape_data_preview"
    assert read(GL / "decision.json")["teacher_preview_passed"]
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
    assert sum(x.size for x in jax.tree.leaves(nnx.state(base, nnx.Param))) == 21718
    for m in METHODS:
        save_field(
            base, cfg, root / f"initials/{m}", provenance="Untouched FW mixed6000; research only"
        )
        restored, cc = load_field(root / f"initials/{m}")
        exact(nnx.state(base, nnx.Param), nnx.state(restored, nnx.Param))
        assert cc == cfg
        save_config(cfg, root / f"configs/{m}.json")
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
        shutil.copytree(GH / folder, root / folder)
    shutil.copytree(GL, root / "procedural-preview", ignore=shutil.ignore_patterns("source.tar.gz"))
    for name in (
        "sampling-plan.json",
        "width-plan.json",
        "frequency-plan.json",
        "width-range-audit.json",
        "probe.json",
    ):
        shutil.copy2(GH / name, root / name)
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(GH / "scene512" / name, root / "scene512" / name)
    (root / "diagnostics").mkdir()
    write(
        root / "train.json",
        dict(
            steps=STEPS,
            parameters=PARAMETERS,
            seed=SEED,
            reference="FW mixed6000 raw",
            optimizer_initialized_fresh=True,
            changed_batches=1500,
            unchanged_batches=4500,
            procedural_examples=6000,
            procedural_examples_each_class=3000,
            adoption=False,
        ),
    )
    files = [p for p in root.rglob("*") if p.is_file()]
    files += [Path(f) for f in read(GH / "plan.json")["bound_files"]]
    for folder in (GJ, GK, GL, LOCALCONTROL):
        files += [folder / "completion.json"]
    files += [
        GJ / "decision.json",
        GK / "decision.json",
        GK / "inputs.npz",
        GL / "recipes.json",
        GL / "observations.json",
        GL / "decision.json",
        DOCUMENT,
    ]
    files += list(source.iterdir())
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initials/control"),
            initial_files={p.name: sha(p) for p in (root / "initials/control").iterdir()},
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
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f


# The bounded training loop and CLI below are derived from the existing audited campaign loop.


def train(root, until=STEPS):
    assert read(GH / "completion.json")["stage"] == "complete_residual_axis"
    assert until in SAVED[3:]
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
        rows = []
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
        prefix = [r["metrics"] for r in rows[: min(start, 24)]]
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
            if not selected(step):
                assert len(set(inputs.values())) == len(set(payloads.values())) == 1
            else:
                assert len(set(inputs.values())) == len(set(payloads.values())) == 2
            for m in METHODS:
                assert stats[m]["merged_fraction"] == sum(flags[m]) / 4
                assert all(np.isfinite(v) for v in stats[m].values())
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
            )
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step < 24:
                prefix.append(stats)
            if step < 24 or (step + 1) % 100 == 0:
                write(out / "progress.json", dict(step=step + 1, metrics=stats))
                print(dict(step=step + 1), flush=True)
            if step == 23:
                replay = arms_from_root(root)
                audits = []
                for k in range(24):
                    actual = {m: update(replay[m], *batch(data, k, m)) for m in METHODS}
                    audits.append({m: replay_state(actual[m], prefix[k][m]) for m in METHODS})
                a, b = (lineart_state.payload(x, 24)["arms"] for x in (replay, arms))
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=36, bindings=bindings))
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
