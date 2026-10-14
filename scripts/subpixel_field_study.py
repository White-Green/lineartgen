"""Learn subpixel geometry fields with the same recursive backbone and training conventions."""

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
from cue_capacity import make_arm, prepare_inputs
from cue_preserving_data import MODES, strength_map
from d4_geometry_data import transform_directions, transform_numpy_batch, transform_prepared
from diagnose_learnability import freeze, verify
from flax import nnx
from focused_fit_study import replay_state
from opaque_geometry import geometry
from plausible_study import digest, exact
from regional_field_objective import regional_gradient
from rough_family_data import BASE, augmentation, rough_source
from scale_style_balance_study import update
from subpixel_field_data import PAD, geometry_field
from subpixel_field_model import SUBPIXELS, infer, load_field, save_field

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

METHODS = ("control", "d4")
STEPS = 12000
SAVED = (0, 6, 6000, STEPS)
SEED = 2026101429
MODEL_SEED = 2026101408
EVAL_SEED = 2026101430
PARENT = Path("../jax-broad-regional-geometry-20261014/runs/broad-regional-geometry-20261014")
ORIENTATION = Path("../jax-geometry-orientation-20261014/runs/geometry-orientation-20261014")
DOCUMENT = Path("docs/d4-geometry-plan-20261014.md")


def select(step):
    rng = np.random.default_rng([SEED, step])
    return dict(
        indices=rng.integers(512, size=4).tolist(),
        modes=rng.integers(len(MODES), size=4).tolist(),
        depth=step % 6 + 1,
        augmentation=[augmentation(SEED, step, i) for i in range(4)],
        transforms=np.random.default_rng([SEED, step, 0x4434]).integers(8, size=4).tolist(),
    )


def batch(data, step, method):
    assert method in METHODS
    s = select(step)
    ii = np.asarray(s["indices"])
    source = np.stack(
        [
            data["source"][i]
            if opts == BASE
            else rough_source(data["variants"][i, [1, 25]].mean(0), opts)
            for i, opts in zip(ii, s["augmentation"], strict=True)
        ]
    )
    strength = np.stack([strength_map(MODES[m]) for m in s["modes"]])
    p = prepare_inputs(
        jnp.asarray(data["variants"][ii, 13]),
        jnp.asarray(source),
        jnp.asarray(strength),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        s["depth"],
    )
    codes = np.asarray(s["transforms"] if method == "d4" else [0] * 4, np.int32)
    fields = data["fields"][ii]
    directions = transform_directions(data["directions"][ii], codes)
    if method == "d4":
        p = transform_prepared(p, jnp.asarray(codes))
        fields = transform_numpy_batch(fields[..., None], codes)[..., 0]
    return p, jnp.asarray(fields), jnp.asarray(directions)


def canonical_batch(p, fields, directions, step, method):
    codes = np.asarray(select(step)["transforms"] if method == "d4" else [0] * 4, np.int32)
    directions = transform_directions(np.asarray(directions), codes, inverse=True)
    if method == "d4":
        p = transform_prepared(p, jnp.asarray(codes), inverse=True)
        fields = transform_numpy_batch(np.asarray(fields)[..., None], codes, inverse=True)[..., 0]
    return p, fields, directions


def load_data(root):
    with np.load(root / "train.npz") as z:
        data = {k: z[k] for k in z.files}
    data["fields"] = np.load(root / "fields.npy", mmap_mode="r")
    data["directions"] = np.load(root / "directions.npy")
    return data


def arms_from_root(root):
    arms = {}
    for method in METHODS:
        model, cfg = load_field(root / f"initials/{method}")
        assert cfg == load_config(root / f"configs/{method}.json")
        arms[method] = make_arm(cfg, model)
        arms[method]["gradient"] = regional_gradient()
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
                provenance=f"subpixel geometry field {name} step{step} {mode}; research only",
            )


def prepare(root):
    assert read(PARENT / "completion.json")["stage"] == "complete_broad_regional_geometry"
    assert read(ORIENTATION / "completion.json")["stage"] == "complete_geometry_orientation"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    old, cfg = load_field(PARENT / "snapshots/step-18000/regions/raw")
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=STEPS, learning_rate_decay_steps=STEPS, seed=SEED, model_seed=MODEL_SEED
                )
            )
        )
    )
    assert cfg.train.peak_learning_rate == 1e-4
    for method in METHODS:
        model = nnx.clone(old)
        save_field(
            model,
            cfg,
            root / f"initials/{method}",
            provenance=f"{method}: unchanged EA regions raw18000 initialization",
        )
        save_config(cfg, root / f"configs/{method}.json")
    for method in METHODS:
        loaded, _ = load_field(root / f"initials/{method}")
        exact(nnx.state(old, nnx.Param), nnx.state(loaded, nnx.Param))
    shutil.copytree(PARENT / "reference", root / "reference")
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "fields.npy",
        "directions.npy",
        "cache-audit.json",
    ):
        shutil.copy2(PARENT / name, root / name)
    for name in ("data", "stress-data"):
        shutil.copytree(PARENT / name, root / name)
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
            methods=METHODS,
            representation="subpixel_geometry_field",
            gradient_weight=0.1,
            sign_weight={"control": 0.25, "d4": 0.25},
            objective={
                "control": "DW censored + DX RMS + DY near/far sign",
                "d4": "DW censored + DX RMS + DY near/far sign",
            },
            spatial_augmentation={"control": "none", "d4": "uniform exact D4 per item"},
            censored_band=1.0,
            rms_epsilon=0.01,
            region_sign_reduction="sum independent near/far requirements",
            margin_cap=0.25,
            subpixels=4,
            distance_range=8,
        ),
    )
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    bound = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    for p in (
        PARENT / "completion.json",
        ORIENTATION / "completion.json",
        ORIENTATION / "evaluation/visual-review.json",
    ):
        bound[str(p.resolve())] = sha(p)
    initial = root / "initials/control"
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
            cache_complete=True,
        ),
    )


def cache(root):
    assert not (root / "source.json").exists()
    p = read(root / "plan.json")
    assert not p["cache_complete"]
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    records = [r for r in read(root / "data.json")["records"] if r["split"] == "expanded"]
    assert len(records) == 512
    side = (256 + 2 * PAD) * SUBPIXELS
    fields = np.lib.format.open_memmap(
        root / "fields.npy", mode="w+", dtype=np.float16, shape=(512, side, side)
    )
    directions = []
    maximum = 0.0
    for i, r in enumerate(records):
        args = tuple(jnp.asarray(a) for a in geometry(r["metadata"], kind=r["kind"]))
        field = np.asarray(geometry_field(*args, size=256))
        assert np.isfinite(field).all()
        fields[i] = field
        restored = np.asarray(fields[i], np.float32)
        band = abs(field) < 8
        maximum = max(maximum, float(np.max(abs(restored - field)[band], initial=0)))
        directions.append(r["direction"])
        if (i + 1) % 16 == 0:
            print(dict(cached=i + 1), flush=True)
    fields.flush()
    assert maximum <= 0.0021
    np.save(root / "directions.npy", np.asarray(directions, np.float32))
    write(
        root / "cache-audit.json",
        dict(
            cases=512,
            shape=list(fields.shape),
            dtype="float16",
            max_abs_error_within8px=maximum,
            padding=PAD,
        ),
    )
    for name in ("fields.npy", "directions.npy", "cache-audit.json"):
        p["bound_files"][str(root / name)] = sha(root / name)
    p["cache_complete"] = True
    write(root / "plan.json", p)


def check(root):
    verify(root)
    p = read(root / "plan.json")
    assert p["cache_complete"]
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f


def train(root):
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    data, arms = load_data(root), arms_from_root(root)
    initial_predictions = {}
    for k in range(6):
        prepared, _, _ = batch(data, k, "control")
        a = np.asarray(infer(arms["control"]["model"], prepared))
        b = np.asarray(infer(arms["d4"]["model"], prepared))
        np.testing.assert_array_equal(a, b)
        initial_predictions[str(k + 1)] = digest(a)
    write(
        root / "initial-audit.json",
        dict(six_depth_functions_exact=True, prediction_sha256=initial_predictions),
    )
    for arm in arms.values():
        assert sum(x.size for x in jax.tree.leaves(nnx.state(arm["model"], nnx.Param))) == 16064
    write(
        root / "backend.json",
        dict(
            device=jax.devices()[0].device_kind,
            parameters=16064,
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
        for step in range(start, STEPS):
            stats, inputs, elapsed, shared, canonical = {}, {}, {}, {}, {}
            for method in METHODS[:: 1 if step % 2 == 0 else -1]:
                p, c, a = batch(data, step, method)
                inputs[method] = digest((p, c, a))
                cp, cc, ca = canonical_batch(p, c, a, step, method)
                shared[method] = digest((cc, ca, cp.noise_levels))
                canonical[method] = digest((cp, cc, ca))
                tick = time.perf_counter()
                stats[method] = update(arms[method], p, c, a)
                elapsed[method] = time.perf_counter() - tick
            assert len(set(shared.values())) == 1
            assert len(set(canonical.values())) == 1
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
                input_sha256=inputs,
                canonical_input_sha256=canonical,
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=16, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "cache", "freeze", "train"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "freeze":
        freeze(root)
    elif a.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root)
    else:
        globals()[a.action](root)
