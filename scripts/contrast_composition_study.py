"""Mixed-opacity independent stroke composition matched to EN merged geometry."""

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
from continuous_width_data import width_choice, width_source
from contrast_composition_data import overlay_choice, overlay_source
from cue_capacity import make_arm, prepare_inputs
from cue_preserving_data import MODES, strength_map
from d4_geometry_data import transform_directions, transform_numpy_batch, transform_prepared
from diagnose_learnability import freeze, verify
from flax import nnx
from merge_family_data import merge_allowed, merge_geometry
from merge_field_objective import merged_gradient
from plausible_study import digest, exact
from rough_family_data import augmentation
from scale_style_balance_study import update
from subpixel_field_model import infer, load_field, save_field

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = ("mixed",)
REPORT_METHODS = ("control", "mixed")
CONTROL = Path("../jax-merge-geometry-20261015/runs/merge-geometry-20261015")
DIAGNOSTIC = Path("../jax-width-response-20261015/runs/width-response-20261015")
PARAMETERS = {m: 16402 for m in METHODS}
STEPS = 6000
SAVED = (0, 6, 3000, STEPS)
SEED = 2026101502
EVAL_SEED = 2026101503
PARENT = Path("../jax-optimized-broad-20261014/runs/optimized-broad-20261014")
DOCUMENT = Path("docs/contrast-composition-plan-20261015.md")


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
    rng = np.random.default_rng([SEED, step])
    indices = rng.integers(512, size=4)
    modes = rng.integers(len(MODES), size=4)
    transforms = rng.integers(8, size=4)
    companions = rng.integers(432, size=4)
    # Fills, blanks and wide guards keep their original task distribution.
    compose = (rng.random(4) < 0.5) & (indices < 432)
    return dict(
        indices=indices.tolist(),
        modes=modes.tolist(),
        depth=step % 6 + 1,
        augmentation=[augmentation(SEED, step, i) for i in range(4)],
        transforms=transforms.tolist(),
        companions=companions.tolist(),
        compose=compose.tolist(),
        overlays=[overlay_choice(SEED, step, item, int(i)) for item, i in enumerate(indices)],
    )


def components(data, selection, method, step):
    assert method == "mixed"
    sources, clean, fields = [], [], []
    for item, (i, opts, overlay) in enumerate(
        zip(selection["indices"], selection["augmentation"], selection["overlays"], strict=True)
    ):
        source = width_source(data, data["records"][i], i, width_choice(SEED, step, item, i), opts)
        target = data["variants"][i, 13]
        field = np.asarray(data["fields"][i])
        if overlay["enabled"]:
            j = overlay["index"]
            secondary = width_source(
                data, data["records"][j], j, overlay["width"], overlay["augmentation"]
            )
            source = overlay_source(source, secondary)
            target = np.minimum(target, data["variants"][j, 13])
            field = np.minimum(field, data["fields"][j])
        sources.append(source)
        clean.append(target)
        fields.append(field)
    return np.stack(sources), np.stack(clean), np.stack(fields)


def batch(data, step, method):
    s = select(step)
    source, clean, fields = components(data, s, method, step)
    codes = jnp.asarray(s["transforms"])
    candidates = jnp.asarray(transform_numpy_batch(fields[..., None], s["transforms"])[..., 0])
    p = prepare_inputs(
        jnp.asarray(clean),
        jnp.asarray(source),
        jnp.asarray(np.stack([strength_map(MODES[m]) for m in s["modes"]])),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        s["depth"],
    )
    ann = jnp.asarray(data["directions"][s["indices"]])
    eligible = np.array(
        [
            merge_allowed(src, data["merge_geometry"][i])
            for src, i in zip(source, s["indices"], strict=True)
        ]
    )
    merged = np.stack(
        [
            data["merged_fields"][data["merge_index"][i]] if ok else fields[j]
            for j, (i, ok) in enumerate(zip(s["indices"], eligible, strict=True))
        ]
    )
    merged = jnp.asarray(transform_numpy_batch(merged[..., None], s["transforms"])[..., 0])
    return (
        transform_prepared(p, codes),
        (candidates, merged, jnp.asarray(eligible)),
        transform_directions(ann, codes),
    )


def load_data(root):
    with np.load(root / "train.npz") as z:
        data = {k: z[k] for k in z.files}
    data["fields"] = np.load(root / "fields.npy", mmap_mode="r")
    data["directions"] = np.load(root / "directions.npy")
    data["records"] = read(root / "data.json")["records"]
    data["merged_fields"] = np.load(root / "merged-fields.npy", mmap_mode="r")
    data["merge_index"] = {i: j for j, i in enumerate(read(root / "merge-cache.json")["indices"])}
    data["merge_geometry"] = [merge_geometry(r) for r in data["records"]]
    return data


def arms_from_root(root):
    arms = {}
    for method in METHODS:
        model, cfg = load_field(root / f"initials/{method}")
        assert cfg == load_config(root / f"configs/{method}.json")
        arms[method] = make_arm(cfg, model)
        arms[method]["gradient"] = merged_gradient()
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
    assert read(parent / "completion.json")["stage"] == "complete_optimized_broad"
    assert read(CONTROL / "completion.json")["stage"] == "complete_merge_geometry"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copytree(parent / "snapshots/step-18000/group/raw", root / "reference")
    model, cfg = load_field(root / "reference")
    train = cfg.train.model_copy(
        update=dict(
            steps=STEPS,
            learning_rate_decay_steps=STEPS,
            seed=SEED,
            learning_rate=1e-4,
            minimum_learning_rate=2e-5,
            warmup_steps=0,
        )
    )
    cfg = cfg.model_copy(update=dict(train=train))
    assert cfg.train.peak_learning_rate == 2e-4
    assert cfg == load_config(CONTROL / "configs/merge.json")
    for method in METHODS:
        save_field(
            model,
            cfg,
            root / f"initials/{method}",
            provenance="EI group6 raw18000; reset optimizer; mixed-opacity composition",
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
        (root / name).symlink_to((CONTROL / name).resolve())
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
            fresh_initialization=False,
            same_initial_parameters=True,
            same_inputs_except_composition=True,
            objective="merged objective; union of opaque component fields",
            reference="EI group6 raw18000",
            optimizer_reset=True,
        ),
    )
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    (root / "snapshots/step-6000").mkdir(parents=True)
    (root / "snapshots/step-6000/control").symlink_to(
        (CONTROL / "snapshots/step-6000/merge").resolve(), target_is_directory=True
    )
    write(
        root / "width-plan.json",
        [
            dict(
                step=k,
                widths=[
                    width_choice(SEED, k, item, i) for item, i in enumerate(select(k)["indices"])
                ],
            )
            for k in range(STEPS)
        ],
    )
    bound.update({str(p): sha(p) for p in root.rglob("*") if p.is_file()})
    for p in (
        CONTROL / "plan.json",
        CONTROL / "source.json",
        CONTROL / "sampling-plan.json",
        CONTROL / "configs/merge.json",
        DIAGNOSTIC / "completion.json",
        DIAGNOSTIC / "evaluation/metrics.json",
    ):
        bound[str(p.resolve())] = sha(p)
    for p in (parent / "completion.json", parent / "source.json", DOCUMENT):
        bound[str(p.resolve())] = sha(p)
    bound.update(read(DIAGNOSTIC / "evaluation/metrics.json")["bindings"])
    initial = root / "initials/mixed"
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
    assert until in (3000, STEPS)
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    assert read(root / "preflight.json")["passed"]
    assert read(CONTROL / "snapshot-audit.json")["raw_ema_exact"]
    control_rows = [
        json.loads(x) for x in (CONTROL / "training/sampling.jsonl").read_text().splitlines()
    ]
    assert len(control_rows) == STEPS
    data, arms = load_data(root), arms_from_root(root)
    reference, _ = load_field(root / "reference")
    exact(nnx.state(reference, nnx.Param), nnx.state(arms["mixed"]["model"], nnx.Param))
    initial_predictions = {}
    for k in range(6):
        prepared, _, _ = batch(data, k, "mixed")
        a = np.asarray(infer(arms["mixed"]["model"], prepared))
        b = np.asarray(infer(reference, prepared))
        np.testing.assert_array_equal(a, b)
        initial_predictions[str(k + 1)] = digest(a)
    write(
        root / "initial-audit.json",
        dict(six_depth_functions_exact=True, prediction_sha256=initial_predictions),
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
            stats, inputs, elapsed, shared = {}, {}, {}, {}
            for method in METHODS[:: 1 if step % 2 == 0 else -1]:
                p, c, a = batch(data, step, method)
                inputs[method] = digest((p, c[0], a))
                shared[method] = digest((c[0], a, p.noise_levels))
                tick = time.perf_counter()
                stats[method] = update(arms[method], p, c, a)
                elapsed[method] = time.perf_counter() - tick
            widths = [
                width_choice(SEED, step, item, i) for item, i in enumerate(select(step)["indices"])
            ]
            overlays = select(step)["overlays"]
            assert np.asarray(c[2]).tolist() == control_rows[step]["merge_eligible"]
            if not any(x["enabled"] for x in overlays):
                assert inputs["mixed"] == control_rows[step]["input_sha256"]["merge"]
                assert shared["mixed"] == control_rows[step]["shared_target_sha256"]["merge"]
                assert digest(c) == control_rows[step]["candidate_payload_sha256"]
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
                source_widths=widths,
                merge_eligible=np.asarray(c[2]).tolist(),
                candidate_payload_sha256=digest(c),
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=8, bindings=bindings))
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
