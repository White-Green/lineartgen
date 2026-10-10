"""New-noise evaluation of the fixed interval-flow toy candidate."""

import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bridge_time_grid import sample_grid
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, verify
from PIL import Image, ImageDraw
from velocity_bridge_toy import measure, targets, velocity

from lineartgen.models.serialization import load_inference

METHODS = ("mse_grid", "interval_grid")
NEW_SEED = 2026103089
SAMPLES = 1024
CHUNK = 64


def chunk_key(depth, partial, strength, chunk):
    group = depth * 4 + int(partial) * 2 + int(strength == 0.5)
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(NEW_SEED), group), chunk)


def quality_checks(row, samples):
    return dict(
        near_modes=row["mean_relative_endpoint_rms"] <= 0.15,
        mostly_resolved=row["fraction_within_020"] >= 0.8,
        both_modes=0.1 <= row["nearest_A"] / samples <= 0.9,
    )


def prepare(root):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    parent = Path("../jax-bridge-time-grid-20261010/runs/bridge-time-grid-20261010").resolve()
    verify(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_solver_diagnostic"
    for method in ("mse", "interval"):
        shutil.copytree(parent / "initial" / method, root / "initial" / method)
        shutil.copyfile(parent / f"configs/{method}.json", root / f"configs/{method}.json")
    shutil.copyfile(parent / "inputs.npz", root / "inputs.npz")
    write(root / "train.json", dict(neural_updates=0, fixed_step=36000))
    write(root / "probe.json", dict(seed=NEW_SEED, samples=SAMPLES, chunk=CHUNK, steps=[32]))
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
            plan_document_sha256=sha(Path("docs/bridge-new-noise-plan-20261010.md")),
            adoption=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(Path("docs/bridge-new-noise-plan-20261010.md")) == plan["plan_document_sha256"]
    parent = Path(plan["control_parent"])
    verify(parent)
    assert sha(parent / "source.json") == plan["control_source_sha256"]
    assert sha(parent / "final-audit.json") == plan["control_final_sha256"]
    assert sha(parent / "evaluation/metrics.json") == plan["control_metrics_sha256"]
    return plan


def evaluate(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    models = {m: load_inference(root / "initial" / m.removesuffix("_grid"))[0] for m in METHODS}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    a, b, s = targets()
    rows = []
    displays = {}
    bindings = {}
    keys = []
    for depth in (1, 2, 3):
        for partial in (False, True):
            for strength in (0.1, 0.5):
                source = jnp.broadcast_to(jnp.asarray(s), (CHUNK, *s.shape))
                t = jnp.full_like(source, strength)
                if partial:
                    t = t.at[:, :, :32].set(0)
                group = f"d{depth}-p{int(partial)}-t{strength}"
                displays[group] = {
                    "A": np.where(np.asarray(t[0]) > 0, a, s),
                    "B": np.where(np.asarray(t[0]) > 0, b, s),
                    "rough": s,
                }
                for method, model in models.items():
                    for steps in (32,):
                        blocks = []
                        for chunk in range(SAMPLES // CHUNK):
                            key = chunk_key(depth, partial, strength, chunk)
                            if method == METHODS[0]:
                                keys.append(
                                    dict(group=group, chunk=chunk, key=np.asarray(key).tolist())
                                )
                            value = sample_grid(
                                lambda x, t, s, u, m=model: velocity(m, x, t, s, u),
                                source,
                                t,
                                key,
                                depth=depth,
                                steps=steps,
                            )
                            blocks.append(np.asarray(value))
                            write(
                                out / "progress.json",
                                dict(
                                    groups=len(displays) - 1,
                                    total=12,
                                    group=group,
                                    method=method,
                                    chunks_done=chunk + 1,
                                    chunks_total=SAMPLES // CHUNK,
                                ),
                            )
                        physical = np.concatenate(blocks)
                        source_np = np.broadcast_to(np.asarray(source[0]), physical.shape)
                        levels_np = np.broadcast_to(np.asarray(t[0]), physical.shape)
                        prediction = np.clip(physical, -1, 1)
                        metric = measure(prediction, a, b, source_np, levels_np)
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
                        if steps == 32:
                            for i in range(4):
                                displays[group][f"{method}[{i}]"] = prediction[i]
                write(out / "progress.json", dict(groups=len(displays), total=12, rows=len(rows)))
    assert len(rows) == 24 and len(keys) == 192
    assert len({tuple(k["key"]) for k in keys}) == 192
    write(root / "keys.json", dict(seed=NEW_SEED, groups=12, chunks_per_group=16, keys=keys))
    primary = [r for r in rows if r["method"] in METHODS and r["steps"] == 32]
    checks = []
    for r in primary:
        for name, passed in quality_checks(r, SAMPLES).items():
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
            samples_per_group=SAMPLES,
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
