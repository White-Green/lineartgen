"""Matched continuation with extra coverage of early continuous path times."""

import argparse
import fcntl
import hashlib
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import noisy_gradient
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from progress_modulation_fit import make_arms as parent_arms
from progress_modulation_fit import prepared_batch as original_batch
from velocity_bridge_toy import SEED, batch, velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.diffusion import pyramid
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm

METHODS = ("control", "early")
START = 6000
STEPS = 12000
SAVED = (START, START + 6, 9000, STEPS)
PARAMETERS = 134688


@jax.jit
def focus_time(prepared, tau, coin):
    selected = coin & (tau > 0)
    early_tau = jnp.where(selected, 0.05 * tau**2, tau)
    old = prepared["bridge"]
    direct = prepared["direct"]
    clean = pyramid(direct["target"], len(direct["images"]))
    images = tuple(
        jnp.where(selected, jnp.where(t > 0, (1 - early_tau) * z + early_tau * y, s), x)
        for x, z, y, t, s in zip(
            old["images"], direct["images"], clean, direct["levels"], direct["guides"], strict=True
        )
    )
    progress = tuple(
        jnp.where(selected, jnp.broadcast_to(1 - early_tau, u.shape), u) for u in old["progress"]
    )
    early = dict(old, images=images, progress=progress)
    return dict(direct=direct, bridge=early), early_tau, selected


def input_hash(values):
    h = hashlib.sha256()
    for x in jax.tree.leaves(values):
        h.update(np.asarray(x).tobytes())
    return h.hexdigest()


def prepared_batch(step):
    depth = step % 3 + 1
    original, labels, tau = batch(step, depth)
    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(SEED), step), 0x4541524C)
    focused, early_tau, selected = focus_time(
        original, tau, jax.random.uniform(key, tau.shape) < 0.5
    )
    values = dict(control=velocity_inputs(original), early=velocity_inputs(focused))
    return values, dict(
        step=step + 1,
        depth=depth,
        label_A=int(np.sum(np.asarray(labels) == 0)),
        zero_count=int(np.sum(np.asarray(tau) == 0)),
        focused_count=int(np.sum(np.asarray(selected))),
        control_early_count=int(np.sum((np.asarray(tau) > 0) & (np.asarray(tau) < 0.01))),
        focused_early_count=int(
            np.sum((np.asarray(early_tau) > 0) & (np.asarray(early_tau) < 0.01))
        ),
        control_tau_mean=float(tau.mean()),
        early_tau_mean=float(early_tau.mean()),
        inputs_sha256={m: input_hash(v) for m, v in values.items()},
    )


def split_arms(reference, configs):
    arms = {}
    for name in METHODS:
        arm = create_arm(configs[name], reference["model"])
        nnx.update(arm["model"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["model"])))
        nnx.update(
            arm["optimizer"], jax.tree.map(lambda x: x.copy(), nnx.state(reference["optimizer"]))
        )
        arm["ema"] = jax.tree.map(lambda x: x.copy(), reference["ema"])
        arm["gradient"] = noisy_gradient
        arms[name] = arm
    return arms


def load_parent(root):
    parent = Path(read(root / "plan.json")["parent"])
    arms = parent_arms(parent)
    with checkpoints.manager(root / "parent-checkpoints", keep=1) as saved:
        assert lineart_state.restore(saved, arms, START) == START
    return arms["noisy"]


def make_arms(root):
    reference = load_parent(root)
    arms = split_arms(reference, {m: load_config(root / f"configs/{m}.json") for m in METHODS})
    initial, _ = load_inference(root / "initial")
    for arm in arms.values():
        assert_initial_parameters(initial, arm["model"])
        assert int(arm["optimizer"].step[...]) == START
    return arms


def prepare(root):
    assert not root.exists()
    parent = Path("../jax-progress-modulation-20261011/runs/progress-modulation-20261011").resolve()
    verify(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_progress_modulation_diagnostic"
    model, old = load_inference(parent / f"snapshots/step-{START}/noisy/raw")
    c = old.model_copy(
        update={
            "train": old.train.model_copy(
                update={
                    "steps": STEPS,
                    "learning_rate": 2.5e-6,
                    "minimum_learning_rate": 1e-5,
                    "warmup_steps": 0,
                    "learning_rate_decay_steps": START,
                }
            )
        }
    )
    assert c.train.peak_learning_rate == 1e-5 and c.train.batch_size == 16
    (root / "configs").mkdir(parents=True)
    for name in METHODS:
        save_config(c, root / f"configs/{name}.json")
    save_inference(
        model,
        c,
        root / "initial",
        provenance="BX noisy6000; actual Adam/EMA continuation with early-time sampling comparison",
    )
    assert read(root / "initial/metadata.json")["parameters"] == PARAMETERS
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(parent / name, root / name)
    shutil.copytree(parent / f"training/checkpoints/{START}", root / f"parent-checkpoints/{START}")
    bound = [parent / n for n in ("source.json", "final-audit.json", "evaluation/metrics.json")]
    bound += list((parent / f"snapshots/step-{START}/noisy/raw").iterdir())
    bound += [p for p in (root / f"parent-checkpoints/{START}").rglob("*") if p.is_file()]
    for p in (root / f"parent-checkpoints/{START}").rglob("*"):
        if p.is_file():
            original = (
                parent
                / f"training/checkpoints/{START}"
                / p.relative_to(root / f"parent-checkpoints/{START}")
            )
            assert sha(p) == sha(original)
            bound.append(original)
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            parent=str(parent),
            bound_files={str(p): sha(p) for p in bound},
            start=START,
            steps=STEPS,
            saved=list(SAVED),
            methods=list(METHODS),
            adoption=False,
            parameters=PARAMETERS,
            plan_document_sha256=sha(Path("docs/early-path-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    p = read(root / "plan.json")
    verify(Path(p["parent"]))
    for path, digest in p["bound_files"].items():
        assert sha(Path(path)) == digest, path
    assert sha(Path("docs/early-path-plan-20261011.md")) == p["plan_document_sha256"]
    return p


def preflight(root, arms):
    reference = load_parent(root)
    for arm in arms.values():
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload({"x": reference}, START),
            lineart_state.payload({"x": arm}, START),
        )
        for n in (0, START, STEPS - 1):
            assert abs(float(arm["schedule"](n)) - 1e-5) < 1e-10
    for step in range(START, START + 6):
        values, _ = prepared_batch(step)
        old, _ = original_batch(step)
        jax.tree.map(np.testing.assert_array_equal, values["control"], old)
    write(
        root / "control-preflight.json",
        dict(parent_full_state_exact=True, original_batches_exact=6, constant_lr=1e-5),
    )


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"] and jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    preflight(root, arms)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else START
        journal = out / "sampling.jsonl"
        rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        if any(r["step"] > start for r in rows):
            write(out / f"resume-discarded-{start}.json", [r for r in rows if r["step"] > start])
            rows = [r for r in rows if r["step"] <= start]
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert [r["step"] for r in rows] == list(range(START + 1, start + 1))
        if saved.latest_step() is None:
            jax.tree.map(
                np.testing.assert_array_equal,
                nnx.state(arms["control"]["model"]),
                nnx.state(arms["early"]["model"]),
            )
            for arm in arms.values():
                assert int(arm["optimizer"].step[...]) == START
            write(
                root / "initial-audit.json",
                dict(
                    full_parameters_identical=True,
                    parent_full_state_exact=True,
                    optimizer_step=START,
                    training_parameters=PARAMETERS,
                    inference_parameters=PARAMETERS,
                ),
            )
            lineart_state.save(saved, arms, START)
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, START) == START
            jax.tree.map(
                np.testing.assert_array_equal,
                lineart_state.payload(arms, START),
                lineart_state.payload(restored, START),
            )
            snapshot(root, arms, START)
            write(out / f"restore-audit-{START}.json", dict(all_state_exact=True))
        prefix = []
        for step in range(start, STEPS):
            p, row = prepared_batch(step)
            complete = step + 1
            stats = {m: update(a, p[m]) for m, a in arms.items()}
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if complete <= START + 6:
                prefix.append(stats)
            if complete <= START + 6 or complete % 50 == 0:
                write(out / "progress.json", dict(**row, metrics=stats))
                print(complete, stats, flush=True)
            if complete == START + 6:
                replay = make_arms(root)
                original_control = load_parent(root)
                for n in range(START, START + 6):
                    p, _ = prepared_batch(n)
                    values = {m: update(a, p[m]) for m, a in replay.items()}
                    assert update(original_control, p["control"]) == values["control"]
                    jax.tree.map(
                        np.testing.assert_array_equal,
                        lineart_state.payload({"x": original_control}, n + 1),
                        lineart_state.payload({"x": replay["control"]}, n + 1),
                    )
                    assert values == prefix[n - START]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, START + 6),
                    lineart_state.payload(replay, START + 6),
                )
                write(
                    root / "prefix-audit.json",
                    dict(
                        updates=6,
                        all_state_exact=True,
                        metrics_exact=True,
                        inputs_regenerated=True,
                        original_control_updates_exact=True,
                    ),
                )
            if complete in SAVED:
                lineart_state.save(saved, arms, complete)
                restored = make_arms(root)
                assert lineart_state.restore(saved, restored, complete) == complete
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, complete),
                    lineart_state.payload(restored, complete),
                )
                write(out / f"restore-audit-{complete}.json", dict(all_state_exact=True))
                snapshot(root, arms, complete)
        bindings = {}
        for step in SAVED:
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, step) == step
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.clone(arm["model"])
                    if mode == "ema":
                        nnx.update(expected, arm["ema"])
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    actual, c = load_inference(folder)
                    assert c == arm["config"]
                    jax.tree.map(
                        np.testing.assert_array_equal,
                        nnx.state(expected, nnx.Param),
                        nnx.state(actual, nnx.Param),
                    )
                    bindings.update({str(p): sha(p) for p in folder.iterdir()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=16, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "train"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="continue6000to12000"))
            train(root)
    else:
        (prepare if args.action == "prepare" else freeze)(root)
