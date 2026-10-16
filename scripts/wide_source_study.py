"""Matched continuation changing only the upper source-width distribution."""

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
from cue_capacity import make_arm, prepare_inputs
from cue_preserving_data import MODES, strength_map
from d4_geometry_data import transform_prepared
from diagnose_learnability import freeze, verify
from extended_width_data import changed_examples, sources, width_records
from flax import nnx
from fresh_aligned_study import batch as aligned_batch
from fresh_aligned_study import load_data as aligned_load_data
from joint_geometry_study import select as parent_select
from opening_prior_objective import opening_gradient
from physical_field_model import load_field, save_field
from plausible_study import digest, exact
from scale_style_balance_study import update
from support_merge_data import merge_allowed

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = ("standard", "expanded")
REPORT_METHODS = METHODS
PARAMETERS = {m: 16402 for m in METHODS}
CONTROL = Path("../jax-fresh-aligned-20261015/runs/fresh-aligned-20261015")
DIAGNOSTIC = Path("../jax-width-response-20261015/runs/width-response-20261015")
DOCUMENT = Path("docs/wide-source-plan-20261016.md")
SEED = 2026101440
EVAL_SEED = 2026101503
STEPS = 6000
SAVED = (0, 6, *range(1000, STEPS + 1, 1000))
OFFSET = 6000
MODELREF = Path("../jax-opening-prior-20261016/runs/opening-prior-20261016")


def select(step):
    return parent_select(step + OFFSET)


def batch(data, step, method):
    if data.get("cached_step") != step:
        source_step = step + OFFSET
        selection = select(step)
        p, c, d = aligned_batch(data, source_step, "aligned")
        previous = data["aligned_rows"][source_step]
        assert digest((p, c[0], d)) == previous["input_sha256"]["aligned"]
        assert digest(c[:3]) == previous["candidate_payload_sha256"]["aligned"]
        assert np.asarray(c[3]).tolist() == previous["context_keep"]
        records = width_records(selection, SEED, source_step)
        changed = changed_examples(records)
        q = p
        if any(changed):
            masks = np.stack([strength_map(MODES[m]) for m in selection["modes"]])
            key = jax.random.fold_in(jax.random.PRNGKey(SEED), source_step)

            def prepared(expanded):
                src, clean = sources(data, selection, records, expanded=expanded)
                flags = [
                    merge_allowed(s, data["merge_geometry"][i])
                    for s, i in zip(src, selection["indices"], strict=True)
                ]
                assert flags == np.asarray(c[2]).tolist()
                value = prepare_inputs(
                    jnp.asarray(clean),
                    jnp.asarray(src),
                    jnp.asarray(masks),
                    key,
                    selection["depth"],
                )
                return transform_prepared(value, jnp.asarray(selection["transforms"]))

            exact(prepared(False), p)
            q = prepared(True)
            exact(q.noise_levels, p.noise_levels)
            exact(q.progress, p.progress)
            for aa, bb, tt in zip(q.targets, p.targets, p.noise_levels, strict=True):
                active = np.asarray(tt) > 0
                np.testing.assert_array_equal(np.asarray(aa)[active], np.asarray(bb)[active])
            # Changes are confined to the predeclared examples at full native resolution.
            for item, did_change in enumerate(changed):
                equal = np.array_equal(np.asarray(p.guides[0][item]), np.asarray(q.guides[0][item]))
                assert equal != did_change
        data["last_audit"] |= dict(
            aligned_payload_exact=True,
            source_step=source_step,
            source_widths=records,
            changed_examples=changed,
            candidate_teachers_exact=True,
        )
        data["cached_step"] = step
        data["cached_batches"] = dict(standard=(p, c, d), expanded=(q, c, d))
    return data["cached_batches"][method]


def load_data(root):
    d = aligned_load_data(root)
    d["aligned_rows"] = [
        json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()
    ]
    return d


def arms_from_root(root):
    arms = {}
    for m in METHODS:
        model, cfg = load_field(root / f"initials/{m}")
        assert cfg == load_config(root / f"configs/{m}.json")
        arms[m] = make_arm(cfg, model)
        arms[m]["gradient"] = opening_gradient()
    return arms


def prepare(root):
    assert read(CONTROL / "completion.json")["stage"] == "complete_fresh_aligned"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    assert read(MODELREF / "completion.json")["stage"] == "complete_opening_prior"
    source = MODELREF / "snapshots/step-6000/opening/raw"
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
    for name in METHODS:
        save_field(
            model,
            cfg,
            root / f"initials/{name}",
            provenance="FN opening6k; source width range comparison; research only",
        )
        loaded, cc = load_field(root / f"initials/{name}")
        assert cc == cfg
        exact(nnx.state(model, nnx.Param), nnx.state(loaded, nnx.Param))
        assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == PARAMETERS[name]
        save_config(cfg, root / f"configs/{name}.json")
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
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(CONTROL / "scene512" / name, root / "scene512" / name)
    shutil.copytree(CONTROL / "teacher-preview", root / "teacher-preview")
    write(root / "sampling-plan.json", [dict(step=k, **select(k)) for k in range(STEPS)])
    write(
        root / "train.json",
        dict(
            cases=512,
            steps=STEPS,
            seed=SEED,
            parameters=PARAMETERS,
            offset=OFFSET,
            reference="FN opening6000 raw",
            fresh_initialization=False,
            optimizer_initialized_fresh=True,
            same_geometry_and_sampling=True,
            source_policy=(
                "FH steps6000..11999;half of augmented wide components extended32..64px"
            ),
            only_change="extend source width only;opening prior0.1 unchanged in both arms",
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
    write(
        root / "width-plan.json",
        [dict(step=k, records=width_records(select(k), SEED, k + OFFSET)) for k in range(STEPS)],
    )
    audit = []
    for rec in read(CONTROL / "data.json")["evaluation"]:
        if rec["split"] == "new_probe" and rec["index"] >= 58:
            paths = rec["metadata"]["path_geometry"]
            peak = max(max(v["widths"]) for v in paths)
            length = max(
                float(np.linalg.norm(np.diff(np.asarray(v["points"]), axis=0), axis=1).sum())
                for v in paths
            )
            audit.append(
                dict(
                    index=rec["index"],
                    native_width=peak,
                    length=length,
                    widths=[peak * f for f in (0.25, 0.5, 0.75, 1, 1.5, 2)],
                    maximum_width_length_ratio=2 * peak / length,
                )
            )
    write(
        root / "width-range-audit.json",
        dict(old_maximum=32, new_maximum=64, probes=audit, ambiguous_short_marks=[60, 62]),
    )
    bound = {str(p.resolve()): sha(p) for p in root.rglob("*") if p.is_file()}
    for folder in ("data", "stress-data"):
        for p in (CONTROL / folder).rglob("*"):
            if p.is_file():
                bound[str(p.resolve())] = sha(p)
    for f in (
        "plan.json",
        "source.json",
        "completion.json",
        "training/sampling.jsonl",
        "snapshot-audit.json",
    ):
        p = CONTROL / f
        bound[str(p.resolve())] = sha(p)
    bound.update(read(DIAGNOSTIC / "evaluation/metrics.json")["bindings"])
    previous = MODELREF
    for relative in (
        "completion.json",
        "training/sampling.jsonl",
        "snapshots/step-6000/opening/raw/weights.npz",
    ):
        p = (previous / relative).resolve()
        bound[str(p)] = sha(p)
    diagnostic = Path("../jax-field-width-20261016/runs/field-width-20261016")
    for relative in ("completion.json", "diagnostics/decision.json", "width-audit.json"):
        p = (diagnostic / relative).resolve()
        bound[str(p)] = sha(p)
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initials/standard"),
            initial_files={p.name: sha(p) for p in (root / "initials/standard").iterdir()},
            data={},
            bound_files=bound,
            methods=METHODS,
            steps=STEPS,
            saved=SAVED,
            plan_document_sha256=sha(DOCUMENT),
            adoption=False,
            parent=str(CONTROL.resolve()),
            seed=SEED,
            outputs=9516,
            native_pages=34,
            fresh_confirmation=False,
        ),
    )


def check(root):
    verify(root)
    assert sha(root / "source.tar.gz") == read(root / "source.json")["archive_sha256"]
    p = read(root / "plan.json")
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f


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
            for m in METHODS:
                assert stats[m]["merged_fraction"] == sum(flags[m]) / 4
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
                merge_eligible=flags,
                context_keep=context_mask(
                    (step + OFFSET) % 18000, select(step)["depth"], 4, enabled=True
                ).tolist(),
                candidate_payload_sha256=payloads,
                input_sha256=inputs,
                shared_target_sha256=shared,
                seconds=elapsed,
                teacher_audit=data["last_audit"],
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
