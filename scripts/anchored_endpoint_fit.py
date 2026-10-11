"""Matched noisy/source anchors in bounded endpoint flow."""

import argparse
import fcntl
import hashlib
import json
import shutil
from pathlib import Path

import jax
import numpy as np
from anchored_endpoint import endpoints, noisy_gradient, source_gradient
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from learned_bridge_toy import endpoint as original_endpoint
from velocity_bridge_toy import batch, velocity_inputs

from lineartgen.config import load_config, save_config
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm

METHODS = ("noisy", "source")
STEPS = 6000
SAVED = (0, 6, 2000, STEPS)


def prepared_batch(step):
    depth = step % 3 + 1
    p, labels, tau = batch(step, depth)
    values = velocity_inputs(p)
    h = hashlib.sha256()
    for x in jax.tree.leaves(values):
        h.update(np.asarray(x).tobytes())
    return values, dict(
        step=step + 1,
        depth=depth,
        label_A=int(np.sum(np.asarray(labels) == 0)),
        tau_mean=float(tau.mean()),
        inputs_sha256=h.hexdigest(),
    )


def make_arms(root):
    initial, _ = load_inference(root / "initial")
    arms = {}
    for name in METHODS:
        arm = create_arm(load_config(root / f"configs/{name}.json"), initial)
        assert_initial_parameters(initial, arm["model"])
        arm["gradient"] = noisy_gradient if name == "noisy" else source_gradient
        arms[name] = arm
    return arms


def prepare(root):
    assert not root.exists()
    parent = Path("../jax-velocity-bridge-toy-20261010/runs/velocity-bridge-toy-20261010").resolve()
    verify(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_learned_toy_diagnostic"
    model, old = load_inference(parent / "initial")
    c = old
    assert c.train.steps == STEPS and c.train.batch_size == 16
    assert c.train.peak_learning_rate == 2e-4 and c.diffusion.scale_loss_multiplier == 0
    (root / "configs").mkdir(parents=True)
    for name in METHODS:
        save_config(c, root / f"configs/{name}.json")
    save_inference(
        model,
        c,
        root / "initial",
        provenance="AV CNN weights; custom bounded endpoint sampler chooses noisy/source anchor",
    )
    assert read(root / "initial/metadata.json")["parameters"] == 130628
    loaded, _ = load_inference(root / "initial")
    assert loaded.output_mode == "residual"
    jax.tree.map(
        np.testing.assert_array_equal, nnx.state(model, nnx.Param), nnx.state(loaded, nnx.Param)
    )
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(parent / name, root / name)
    shutil.copytree(parent / "snapshots/step-6000/velocity/raw", root / "control")
    bound = [parent / n for n in ("source.json", "final-audit.json", "evaluation/metrics.json")]
    bound += list((parent / "initial").iterdir()) + list((root / "control").iterdir())
    data_parent = Path(
        "../jax-preconditioned-velocity-20261011/runs/preconditioned-velocity-20261011"
    ).resolve()
    verify(data_parent)
    assert (
        read(data_parent / "final-audit.json")["stage"]
        == "complete_preconditioned_velocity_diagnostic"
    )
    bound += [
        data_parent / n for n in ("source.json", "final-audit.json", "training/sampling.jsonl")
    ]
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            parent=str(parent),
            data_parent=str(data_parent),
            bound_files={str(p): sha(p) for p in bound},
            steps=STEPS,
            saved=list(SAVED),
            methods=list(METHODS),
            adoption=False,
            plan_document_sha256=sha(Path("docs/anchored-endpoint-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    p = read(root / "plan.json")
    verify(Path(p["parent"]))
    verify(Path(p["data_parent"]))
    for path, digest in p["bound_files"].items():
        assert sha(Path(path)) == digest, path
    assert sha(Path("docs/anchored-endpoint-plan-20261011.md")) == p["plan_document_sha256"]
    return p


def preflight(root, arms):
    plan = check(root)
    original, _ = load_inference(Path(plan["parent"]) / "initial")
    for arm in arms.values():
        jax.tree.map(
            np.testing.assert_array_equal,
            nnx.state(original, nnx.Param),
            nnx.state(arm["model"], nnx.Param),
        )
    for step in range(3):
        p, _ = prepared_batch(step)
        args = {k: p[k] for k in ("images", "levels", "guides", "progress")}
        a = endpoints(arms["noisy"]["model"], **args, source_anchor=False)[0]
        b = original_endpoint(original, **args)
        jax.tree.map(np.testing.assert_array_equal, a, b)
        assert all(np.isfinite(np.asarray(x)).all() for x in a)
    write(
        root / "control-preflight.json",
        dict(parent_parameters_exact=True, initial_noisy_endpoint_exact=True, depths=3),
    )


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"] and jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    preflight(root, arms)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        if any(r["step"] > start for r in rows):
            write(out / f"resume-discarded-{start}.json", [r for r in rows if r["step"] > start])
            rows = [r for r in rows if r["step"] <= start]
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert [r["step"] for r in rows] == list(range(1, start + 1))
        if saved.latest_step() is None:
            jax.tree.map(
                np.testing.assert_array_equal,
                nnx.state(arms["noisy"]["model"]),
                nnx.state(arms["source"]["model"]),
            )
            for arm in arms.values():
                assert int(arm["optimizer"].step[...]) == 0
                jax.tree.map(
                    np.testing.assert_array_equal, arm["ema"], nnx.state(arm["model"], nnx.Param)
                )
            write(
                root / "initial-audit.json",
                dict(
                    full_parameters_identical=True,
                    optimizer_zero=True,
                    ema_equals_model=True,
                    training_parameters=130628,
                    inference_parameters=130628,
                ),
            )
            lineart_state.save(saved, arms, 0)
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, 0) == 0
            jax.tree.map(
                np.testing.assert_array_equal,
                lineart_state.payload(arms, 0),
                lineart_state.payload(restored, 0),
            )
            snapshot(root, arms, 0)
            write(out / "restore-audit-0.json", dict(all_state_exact=True))
        prefix = []
        for step in range(start, STEPS):
            p, row = prepared_batch(step)
            complete = step + 1
            stats = {m: update(a, p) for m, a in arms.items()}
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if complete <= 6:
                prefix.append(stats)
            if complete <= 6 or complete % 50 == 0:
                write(out / "progress.json", dict(**row, metrics=stats))
                print(complete, stats, flush=True)
            if complete == 6:
                replay = make_arms(root)
                for n in range(6):
                    p, _ = prepared_batch(n)
                    values = {m: update(a, p) for m, a in replay.items()}
                    assert values == prefix[n]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, 6),
                    lineart_state.payload(replay, 6),
                )
                write(
                    root / "prefix-audit.json",
                    dict(
                        updates=6, all_state_exact=True, metrics_exact=True, inputs_regenerated=True
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
            write(root / "workflow-status.json", dict(stage="train6000"))
            train(root)
    else:
        (prepare if args.action == "prepare" else freeze)(root)
