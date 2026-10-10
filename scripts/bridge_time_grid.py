"""Fixed-weight comparison of uniform and boundary-focused Euler nodes."""

import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, verify
from PIL import Image, ImageDraw
from velocity_bridge_toy import EVAL_SEED, initialize, measure, sample, targets, velocity

from lineartgen.models.serialization import load_inference

METHODS = ("mse_grid", "interval_grid")


def time_nodes(steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    nodes = np.sin(np.linspace(0, np.pi / 2, steps + 1)) ** 2
    nodes[0], nodes[-1] = 0.0, 1.0
    return nodes


def sample_grid(predict, source, strength, key, *, depth, steps):
    images, levels, guides = initialize(source, strength, key, depth)
    nodes = time_nodes(steps)
    for start, end in zip(nodes[:-1], nodes[1:], strict=True):
        progress = tuple(jnp.full_like(x, 1 - start) for x in images)
        estimates = predict(images, levels, guides, progress)
        images = tuple(
            jnp.where(t > 0, x + (end - start) * v, s)
            for x, v, t, s in zip(images, estimates, levels, guides, strict=True)
        )
    return images[0]


def prepare(root):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    parent = Path(
        "../jax-interval-velocity-fit-20261010/runs/interval-velocity-fit-20261010"
    ).resolve()
    verify(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_learned_toy_diagnostic"
    for method in ("mse", "interval"):
        shutil.copytree(parent / f"snapshots/step-36000/{method}/raw", root / "initial" / method)
        shutil.copyfile(parent / f"configs/{method}.json", root / f"configs/{method}.json")
    shutil.copyfile(parent / "inputs.npz", root / "inputs.npz")
    write(root / "train.json", dict(neural_updates=0, fixed_step=36000))
    write(root / "probe.json", dict(seed=EVAL_SEED, samples=64, steps=[1, 8, 32]))
    initial = root / "initial"
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={
                str(f.relative_to(initial)): sha(f) for f in initial.rglob("*") if f.is_file()
            },
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            control_parent=str(parent),
            control_source_sha256=sha(parent / "source.json"),
            control_final_sha256=sha(parent / "final-audit.json"),
            control_metrics_sha256=sha(parent / "evaluation/metrics.json"),
            plan_document_sha256=sha(Path("docs/bridge-time-grid-plan-20261010.md")),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(Path("docs/bridge-time-grid-plan-20261010.md")) == plan["plan_document_sha256"]
    parent = Path(plan["control_parent"])
    verify(parent)
    assert sha(parent / "source.json") == plan["control_source_sha256"]
    assert sha(parent / "final-audit.json") == plan["control_final_sha256"]
    assert sha(parent / "evaluation/metrics.json") == plan["control_metrics_sha256"]
    return plan


def evaluate(root):
    plan = check(root)
    parent = Path(plan["control_parent"])
    assert read(root / "test-results.json")["passed"]
    models = {m: load_inference(root / "initial" / m.removesuffix("_grid"))[0] for m in METHODS}
    replay_rows = []
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    a, b, s = targets()
    old = read(parent / "evaluation/metrics.json")
    rows = list(old["rows"])
    displays = {}
    bindings = dict(old["bindings"])
    for path, digest in bindings.items():
        assert sha(Path(path)) == digest
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
                for label, method, n in (
                    ("mse_uniform", "mse", 32),
                    ("interval_uniform", "interval", 32),
                ):
                    filename = f"{group}-{method}-{n}.npz"
                    paths = [Path(p) for p in old["bindings"] if Path(p).name == filename]
                    assert len(paths) == 1
                    with np.load(paths[0]) as saved:
                        displays[group][label] = saved["prediction"][0]
                for method, model in models.items():
                    for steps in (1, 8, 32):
                        original = method.removesuffix("_grid")
                        uniform = np.asarray(
                            sample(
                                lambda x, t, s, u, m=model: velocity(m, x, t, s, u),
                                source,
                                t,
                                key,
                                depth=depth,
                                steps=steps,
                            )
                        )
                        filename = f"{group}-{original}-{steps}.npz"
                        paths = [Path(p) for p in old["bindings"] if Path(p).name == filename]
                        assert len(paths) == 1
                        with np.load(paths[0]) as cached:
                            np.testing.assert_array_equal(uniform, cached["physical"])
                            np.testing.assert_array_equal(
                                np.clip(uniform, -1, 1), cached["prediction"]
                            )
                        replay_rows.append(
                            dict(method=original, group=group, steps=steps, exact=True)
                        )
                        physical = np.asarray(
                            sample_grid(
                                lambda x, t, s, u, m=model: velocity(m, x, t, s, u),
                                source,
                                t,
                                key,
                                depth=depth,
                                steps=steps,
                            )
                        )
                        prediction = np.clip(physical, -1, 1)
                        metric = measure(prediction, a, b, np.asarray(source), np.asarray(t))
                        assert metric["finite"] and np.isfinite(physical).all()
                        assert metric["protected_max_error"] == 0
                        metric.update(
                            unclipped_minimum=float(physical.min()),
                            unclipped_maximum=float(physical.max()),
                        )
                        rows.append(
                            dict(
                                method=method,
                                depth=depth,
                                partial=partial,
                                strength=strength,
                                steps=steps,
                                **metric,
                            )
                        )
                        path = out / f"predictions/{group}-{method}-{steps}.npz"
                        np.savez_compressed(path, prediction=prediction, physical=physical)
                        bindings[str(path)] = sha(path)
                        if steps == 1:
                            displays[group][method + "1"] = prediction[0]
                        if steps == 32:
                            for i in range(4):
                                displays[group][f"{method}[{i}]"] = prediction[i]
                write(out / "progress.json", dict(groups=len(displays), total=12, rows=len(rows)))
    assert len(rows) == 396
    assert len(replay_rows) == 72
    write(root / "replay-audit.json", dict(all_outputs_exact=True, blocks=72, rows=replay_rows))
    primary = [r for r in rows if r["method"] in METHODS and r["steps"] == 32]
    checks = []
    for r in primary:
        for name, passed in dict(
            near_modes=r["mean_relative_endpoint_rms"] <= 0.15,
            mostly_resolved=r["fraction_within_020"] >= 0.8,
            both_modes=0.1 <= r["nearest_A"] / 64 <= 0.9,
        ).items():
            checks.append(
                dict(
                    method=r["method"],
                    depth=r["depth"],
                    partial=r["partial"],
                    strength=r["strength"],
                    guard=name,
                    passed=passed,
                )
            )
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            all_passed=all(c["passed"] for c in checks),
            finite=True,
            protected_exact=True,
            samples_per_group=64,
            adoption=False,
            bindings=bindings,
        ),
    )
    records = []
    for depth in (1, 2, 3):
        columns = (
            "A",
            "B",
            "rough",
            "mse_uniform",
            "interval_uniform",
            *[f"{method}[{i}]" for method in METHODS for i in range(4)],
        )
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
    write(out / "render-manifest.json", dict(records=records, first_four_fixed=True))
    write(
        root / "workflow-status.json",
        dict(stage="numerical_complete_pending_review", neural_updates=0),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "evaluate"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "freeze":
        freeze(root)
    else:
        globals()[args.action](root)
