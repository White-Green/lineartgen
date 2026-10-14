"""Matched global versus locally coherent clean-profile choice at fixed model capacity."""

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
from coherent_path_data import dense_annotation
from coherent_path_objective import structured_gradient
from cue_capacity import make_arm, prepare_inputs
from cue_preserving_data import MODES, strength_map
from diagnose_learnability import freeze, snapshot, verify
from flax import nnx
from focused_fit_study import replay_state
from plausible_study import digest, exact
from rough_family_data import BASE, augmentation, rough_source
from scale_style_balance_study import update

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state

METHODS = ("global", "path")
STEPS = 6000
SAVED = (0, 6, 2000, STEPS)
SEED = 2026101405
EVAL_SEED = 2026101406
PARENT = Path("../jax-scale-style-balance-20261014/runs/scale-style-balance-20261014")
DOCUMENT = Path("docs/coherent-path-plan-20261014.md")


def select(step):
    rng = np.random.default_rng([SEED, step])
    return dict(
        indices=rng.integers(512, size=4).tolist(),
        modes=rng.integers(len(MODES), size=4).tolist(),
        depth=step % 6 + 1,
        augmentation=[augmentation(SEED, step, i) for i in range(4)],
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
    candidates = jnp.asarray(data["allstroke"][ii, 1::3])
    strength = np.stack([strength_map(MODES[m]) for m in s["modes"]])
    p = prepare_inputs(
        candidates[:, 4],
        jnp.asarray(source),
        jnp.asarray(strength),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        s["depth"],
    )
    ann = {k: jnp.asarray(data[k][ii]) for k in ("dense_probes", "dense_valid", "spacing")}
    return p, candidates, ann


def load_data(root):
    data = {}
    for name in ("train", "allstroke", "dense-annotations"):
        with np.load(root / f"{name}.npz") as z:
            data.update({k: z[k] for k in z.files})
    return data


def arms_from_root(root):
    model, _ = load_inference(root / "initial")
    arms = {}
    for method in METHODS:
        arms[method] = make_arm(load_config(root / f"configs/{method}.json"), model)
        arms[method]["gradient"] = structured_gradient(method)
    return arms


def prepare(root):
    parent = PARENT.resolve()
    assert read(parent / "completion.json")["stage"] == "complete_scale_style_balance"
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copytree(parent / "snapshots/step-18000/both/raw", root / "initial")
    _, cfg = load_inference(root / "initial")
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(steps=STEPS, learning_rate_decay_steps=STEPS, seed=SEED)
            )
        )
    )
    assert cfg.train.peak_learning_rate == 1e-4 and cfg.model.pyramid_conditioning == "fourier"
    for method in METHODS:
        save_config(cfg, root / f"configs/{method}.json")
    for name in ("train.npz", "allstroke.npz", "data.json", "stress-plan.json"):
        shutil.copy2(parent / name, root / name)
    for name in ("data", "stress-data"):
        shutil.copytree(parent / name, root / name)
    rows = [r for r in read(root / "data.json")["records"] if r["split"] == "expanded"]
    assert len(rows) == 512
    annotations = [dense_annotation(r["metadata"]) for r in rows]
    np.savez_compressed(
        root / "dense-annotations.npz",
        **{k: np.stack([a[k] for a in annotations]) for k in annotations[0]},
    )
    scene = root / "scene512"
    scene.mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(parent / "diagnostics/scene512" / name, scene / name)
    write(root / "sampling-plan.json", [dict(step=k, **select(k)) for k in range(STEPS)])
    write(root / "train.json", dict(cases=512, steps=STEPS, seed=SEED, methods=METHODS))
    write(
        root / "probe.json",
        dict(cases=256, seed=EVAL_SEED, modes=MODES, draws=2, depth=3, confirmation=False),
    )
    bound = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    for name in ("completion.json", "source.json", "diagnostics/review-observations.json"):
        bound[str(parent / name)] = sha(parent / name)
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={},
            bound_files=bound,
            methods=METHODS,
            steps=STEPS,
            saved=SAVED,
            plan_document_sha256=sha(DOCUMENT),
            adoption=False,
            parent=str(parent),
            seed=SEED,
        ),
    )


def check(root):
    verify(root)
    p = read(root / "plan.json")
    assert sha(DOCUMENT) == p["plan_document_sha256"]
    for f, h in p["bound_files"].items():
        assert sha(Path(f)) == h, f


def train(root):
    check(root)
    assert os.environ.get("XLA_FLAGS", "") == "" and jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    data, arms = load_data(root), arms_from_root(root)
    exact(
        nnx.state(arms["global"]["model"], nnx.Param),
        nnx.state(arms["path"]["model"], nnx.Param),
    )
    write(
        root / "backend.json",
        dict(
            device=jax.devices()[0].device_kind,
            parameters=15809,
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
            stats, inputs, elapsed, shared = {}, {}, {}, {}
            for method in METHODS[:: 1 if step % 2 == 0 else -1]:
                p, c, a = batch(data, step, method)
                inputs[method] = digest((p, c, a))
                shared[method] = digest((c, a, p.noise_levels))
                tick = time.perf_counter()
                stats[method] = update(arms[method], p, c, a)
                elapsed[method] = time.perf_counter() - tick
            assert len(set(shared.values())) == 1
            row = dict(
                step=step + 1,
                **select(step),
                metrics=stats,
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
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(saved, restored, step) == step
            for method, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                    folder = root / f"snapshots/step-{step}/{method}/{mode}"
                    actual, cfg = load_inference(folder)
                    assert cfg == arm["config"]
                    exact(expected, nnx.state(actual, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=16, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "train"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root)
    elif args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root)
    else:
        prepare(root)
