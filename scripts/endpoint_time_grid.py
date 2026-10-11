"""Fixed BV endpoints with explicitly uncapped sampling-time denominators."""

import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import endpoints
from bridge_time_grid import sample_grid
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, verify
from PIL import Image, ImageDraw
from velocity_bridge_toy import EVAL_SEED, measure, sample, targets

from lineartgen.models.serialization import load_inference

ANCHORS = ("noisy", "source")
SCHEMES = ("uniform32", "sine32", "sine128")


def velocity(model, images, levels, guides, progress, source_anchor=False):
    bounded = endpoints(model, images, levels, guides, progress, source_anchor)[1]
    # All prescribed nodes are strictly before1. Epsilon only defines a finite
    # guard outside that domain; BV's training-time1/32 cap is not used here.
    return tuple(
        jnp.where(t > 0, (y - x) / jnp.maximum(u, jnp.finfo(x.dtype).eps), 0)
        for y, x, t, u in zip(bounded, images, levels, progress, strict=True)
    )


def prepare(root):
    assert not root.exists()
    parent = Path("../jax-anchored-endpoint-20261011/runs/anchored-endpoint-20261011").resolve()
    verify(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_anchored_endpoint_diagnostic"
    (root / "configs").mkdir(parents=True)
    for name in ANCHORS:
        shutil.copytree(parent / f"snapshots/step-6000/{name}/raw", root / "models" / name)
        shutil.copyfile(parent / f"configs/{name}.json", root / f"configs/{name}.json")
    for name in ("inputs.npz", "probe.json"):
        shutil.copyfile(parent / name, root / name)
    write(root / "train.json", dict(neural_updates=0, checkpoint=6000))
    bound = [
        parent / n
        for n in (
            "source.json",
            "final-audit.json",
            "evaluation/metrics.json",
            "snapshot-audit.json",
        )
    ]
    bound += [p for name in ANCHORS for p in (parent / f"snapshots/step-6000/{name}/raw").iterdir()]
    write(
        root / "plan.json",
        dict(
            initial=str(root / "models"),
            initial_files={
                str(p.relative_to(root / "models")): sha(p)
                for p in (root / "models").rglob("*")
                if p.is_file()
            },
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            parent=str(parent),
            bound_files={str(p): sha(p) for p in bound},
            anchors=list(ANCHORS),
            schemes=list(SCHEMES),
            adoption=False,
            plan_document_sha256=sha(Path("docs/endpoint-time-grid-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    verify(Path(plan["parent"]))
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest, path
    assert sha(Path("docs/endpoint-time-grid-plan-20261011.md")) == plan["plan_document_sha256"]
    return plan


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu" and read(root / "test-results.json")["passed"]
    parent = Path(plan["parent"])
    previous = read(parent / "evaluation/metrics.json")
    old = {
        (r["method"], r["depth"], r["partial"], r["strength"]): r
        for r in previous["rows"]
        if r["steps"] == 32
    }
    models = {name: load_inference(root / "models" / name)[0] for name in ANCHORS}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    a, b, s = targets()
    with np.load(root / "inputs.npz") as data:
        for k, v in [("a", a), ("b", b), ("source", s)]:
            np.testing.assert_array_equal(data[k], v)
    rows, bindings, displays, replay = [], {}, {}, []
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
                for anchor, model in models.items():
                    for scheme in SCHEMES:
                        steps = 128 if scheme == "sine128" else 32
                        fn = sample if scheme == "uniform32" else sample_grid
                        physical = np.asarray(
                            fn(
                                lambda x, t, s, u, m=model, z=anchor == "source": velocity(
                                    m, x, t, s, u, z
                                ),
                                source,
                                t,
                                key,
                                depth=depth,
                                steps=steps,
                            )
                        )
                        prediction = np.clip(physical, -1, 1)
                        metrics = measure(prediction, a, b, np.asarray(source), np.asarray(t))
                        assert (
                            metrics["finite"]
                            and np.isfinite(physical).all()
                            and metrics["protected_max_error"] == 0
                        )
                        if scheme == "uniform32":
                            path = parent / f"evaluation/predictions/{group}-{anchor}-32.npz"
                            assert sha(path) == previous["bindings"][str(path)]
                            with np.load(path) as data:
                                np.testing.assert_array_equal(physical, data["physical"])
                                np.testing.assert_array_equal(prediction, data["prediction"])
                            r = old[anchor, depth, partial, strength]
                            assert all(r[k] == v for k, v in metrics.items())
                            assert r["unclipped_minimum"] == float(physical.min()) and r[
                                "unclipped_maximum"
                            ] == float(physical.max())
                            replay.append(dict(anchor=anchor, group=group, exact=True))
                        rows.append(
                            dict(
                                method=f"{anchor}_{scheme}",
                                anchor=anchor,
                                scheme=scheme,
                                depth=depth,
                                partial=partial,
                                strength=strength,
                                steps=steps,
                                **metrics,
                                unclipped_minimum=float(physical.min()),
                                unclipped_maximum=float(physical.max()),
                            )
                        )
                        path = out / f"predictions/{group}-{anchor}-{scheme}.npz"
                        np.savez_compressed(path, prediction=prediction, physical=physical)
                        bindings[str(path)] = sha(path)
                        for i in range(4):
                            displays[group][f"{anchor}/{scheme}[{i}]"] = prediction[i]
                write(out / "progress.json", dict(groups=len(displays), total=12, rows=len(rows)))
                print(f"evaluated {len(displays)}/12; rows={len(rows)}", flush=True)
    assert len(rows) == 72 and len(replay) == 24
    checks = []
    for r in rows:
        for name, passed in dict(
            near_modes=r["mean_relative_endpoint_rms"] <= 0.15,
            mostly_resolved=r["fraction_within_020"] >= 0.8,
            both_modes=0.1 <= r["nearest_A"] / 64 <= 0.9,
        ).items():
            checks.append(
                {
                    **{k: r[k] for k in ("method", "depth", "partial", "strength")},
                    "guard": name,
                    "passed": passed,
                }
            )
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            bindings=bindings,
            finite=True,
            protected_exact=True,
            samples_per_group=64,
            adoption=False,
        ),
    )
    write(root / "replay-audit.json", dict(outputs_exact=24, metrics_exact=True, rows=replay))
    records = []
    for anchor in ANCHORS:
        for depth in (1, 2, 3):
            columns = (
                "A",
                "B",
                "rough",
                *[f"{anchor}/{scheme}[{i}]" for scheme in SCHEMES for i in range(4)],
            )
            canvas = Image.new("L", (104 * len(columns), 116 * 4), 255)
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
                    draw.text(
                        (col * 104 + 2, row * 116 + 18), label.removeprefix(anchor + "/"), fill=0
                    )
                    canvas.paste(Image.fromarray(pixels), (col * 104 + 20, row * 116 + 40))
            path = out / f"{anchor}-depth-{depth}-native.png"
            canvas.save(path)
            records.append(dict(file=path.name, sha256=sha(path)))
    write(out / "render-manifest.json", dict(records=records, first_four_fixed=True))
    write(
        root / "workflow-status.json",
        dict(stage="numerical_complete_pending_review", neural_updates=0),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "evaluate"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    {"prepare": prepare, "freeze": freeze, "evaluate": evaluate}[args.action](args.root.resolve())
