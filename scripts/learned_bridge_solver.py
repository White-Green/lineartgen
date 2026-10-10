"""More Euler steps for a fixed learned two-mode field; zero neural training."""

import argparse
import fcntl
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, verify
from learned_bridge_toy import EVAL_SEED, endpoint, measure, sample, targets
from learned_bridge_toy import check as check_parent
from PIL import Image, ImageDraw

from lineartgen.models.serialization import load_inference

COUNTS = (32, 64, 128, 256)


def prepare(root):
    assert not root.exists()
    parent = Path("../jax-learned-bridge-toy-20261010/runs/learned-bridge-toy-20261010").resolve()
    check_parent(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_learned_toy_diagnostic"
    assert read(parent / "final-audit.json")["final_images_reviewed"] == 3
    assert read(parent / "test-results.json")["passed"]
    inherited = {}
    old_worktree = parent.parents[1]
    for name, digest in read(parent / "source.json")["files"].items():
        relative = Path(name).relative_to(old_worktree)
        actual = Path(relative).resolve()
        assert sha(actual) == digest
        inherited[str(actual)] = digest
    (root / "configs").mkdir(parents=True)
    initial = parent / "snapshots/step-6000/bridge/raw"
    shutil.copyfile(parent / "inputs.npz", root / "inputs.npz")
    write(root / "train.json", dict(neural_updates=0, fixed_checkpoint=str(initial)))
    write(
        root / "probe.json",
        dict(groups=12, samples_per_group=64, seed=EVAL_SEED, steps=list(COUNTS)),
    )
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir()},
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            parent_final_sha256=sha(parent / "final-audit.json"),
            parent_metrics_sha256=sha(parent / "evaluation/metrics.json"),
            inherited_files=inherited,
            adoption=False,
            neural_updates=0,
            plan_document_sha256=sha(Path("docs/learned-bridge-solver-plan-20261010.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    parent = Path(plan["parent"])
    check_parent(parent)
    for key, name in (
        ("parent_source_sha256", "source.json"),
        ("parent_final_sha256", "final-audit.json"),
        ("parent_metrics_sha256", "evaluation/metrics.json"),
    ):
        assert sha(parent / name) == plan[key]
    assert sha(Path("docs/learned-bridge-solver-plan-20261010.md")) == plan["plan_document_sha256"]
    for path, digest in plan["inherited_files"].items():
        assert sha(Path(path)) == digest
    return plan


def run(root):
    plan = check(root)
    parent = Path(plan["parent"])
    assert jax.devices()[0].platform != "cpu"
    model, _ = load_inference(plan["initial"])
    a, b, s = targets()
    with np.load(root / "inputs.npz") as data:
        for key, array in (("a", a), ("b", b), ("source", s)):
            np.testing.assert_array_equal(data[key], array)
    old_rows = {
        (r["depth"], r["partial"], r["strength"]): r
        for r in read(parent / "evaluation/metrics.json")["rows"]
        if r["method"] == "bridge" and r["steps"] == 32
    }
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    bindings = {}
    differences = []
    displays = {}
    for depth in (1, 2, 3):
        for partial in (False, True):
            for strength in (0.1, 0.5):
                source = jnp.broadcast_to(jnp.asarray(s), (64, *s.shape))
                t = jnp.full_like(source, strength)
                if partial:
                    t = t.at[:, :, :32].set(0)
                key = jax.random.fold_in(
                    jax.random.PRNGKey(EVAL_SEED),
                    depth * 4 + int(partial) * 2 + int(strength == 0.5),
                )
                group = f"d{depth}-p{int(partial)}-t{strength}"
                displays[group] = {
                    "A": np.where(np.asarray(t[0]) > 0, a, s),
                    "B": np.where(np.asarray(t[0]) > 0, b, s),
                    "rough": s,
                }
                previous = None
                for steps in COUNTS:
                    begin = time.perf_counter()
                    physical = np.asarray(
                        sample(
                            lambda x, t, s, u: endpoint(model, x, t, s, u),
                            source,
                            t,
                            key,
                            depth=depth,
                            steps=steps,
                        )
                    )
                    elapsed = time.perf_counter() - begin
                    prediction = np.clip(physical, -1, 1)
                    metric = measure(prediction, a, b, np.asarray(source), np.asarray(t))
                    assert metric["finite"] and np.isfinite(physical).all()
                    assert metric["protected_max_error"] == 0
                    row = dict(
                        method="bridge",
                        depth=depth,
                        partial=partial,
                        strength=strength,
                        steps=steps,
                        **metric,
                        unclipped_minimum=float(physical.min()),
                        unclipped_maximum=float(physical.max()),
                    )
                    if steps == 32:
                        assert row == old_rows[depth, partial, strength]
                        with np.load(
                            parent / f"evaluation/predictions/{group}-bridge-32.npz"
                        ) as old:
                            np.testing.assert_array_equal(physical, old["physical"])
                            np.testing.assert_array_equal(prediction, old["prediction"])
                    rows.append(dict(row, seconds=elapsed))
                    path = out / f"predictions/{group}-bridge-{steps}.npz"
                    np.savez_compressed(path, physical=physical, prediction=prediction)
                    bindings[str(path)] = sha(path)
                    if previous is not None:
                        delta = prediction.astype(np.float64) - previous
                        differences.append(
                            dict(
                                depth=depth,
                                partial=partial,
                                strength=strength,
                                steps_from=steps // 2,
                                steps_to=steps,
                                mean_absolute_change=float(np.abs(delta).mean()),
                                maximum_absolute_change=float(np.abs(delta).max()),
                            )
                        )
                    previous = prediction
                    displays[group][str(steps)] = prediction[0]
                    if steps == 256:
                        for i in range(1, 4):
                            displays[group][f"256[{i}]"] = prediction[i]
                    write(out / "progress.json", dict(blocks=len(rows), total=48))
    checks = []
    for r in rows:
        for name, passed in dict(
            near_modes=r["mean_relative_endpoint_rms"] <= 0.15,
            mostly_resolved=r["fraction_within_020"] >= 0.8,
            both_modes=0.1 <= r["nearest_A"] / 64 <= 0.9,
        ).items():
            checks.append(
                dict(
                    depth=r["depth"],
                    partial=r["partial"],
                    strength=r["strength"],
                    steps=r["steps"],
                    guard=name,
                    passed=passed,
                )
            )
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            differences=differences,
            bindings=bindings,
            all32_reference_exact=True,
            finite=True,
            protected_exact=True,
            passed_by_steps={
                str(n): all(c["passed"] for c in checks if c["steps"] == n) for n in COUNTS
            },
        ),
    )
    records = []
    columns = ("A", "B", "rough", "32", "64", "128", "256", "256[1]", "256[2]", "256[3]")
    for depth in (1, 2, 3):
        canvas = Image.new("L", (96 * len(columns), 116 * 4), 255)
        draw = ImageDraw.Draw(canvas)
        for row, (partial, strength) in enumerate(
            (p, t) for p in (False, True) for t in (0.1, 0.5)
        ):
            group = f"d{depth}-p{int(partial)}-t{strength}"
            draw.text((2, row * 116 + 2), group, fill=0)
            for col, label in enumerate(columns):
                pixels = np.rint(
                    np.clip((displays[group][label][..., 0] + 1) * 127.5, 0, 255)
                ).astype(np.uint8)
                draw.text((col * 96 + 2, row * 116 + 18), label, fill=0)
                canvas.paste(Image.fromarray(pixels), (col * 96 + 12, row * 116 + 40))
        path = out / f"depth-{depth}-native.png"
        canvas.save(path)
        records.append(dict(file=path.name, sha256=sha(path)))
    write(out / "render-manifest.json", dict(records=records, first_draws_fixed=True))
    write(
        root / "workflow-status.json",
        dict(stage="numerical_complete_pending_review", neural_updates=0),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "run"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "run":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="solver_evaluation"))
            run(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        prepare(root)
