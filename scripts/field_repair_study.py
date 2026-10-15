"""Optimize known failing output fields directly, without updating the CNN."""

import argparse
import copy
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from opaque_geometry import geometry, render
from physical_field_model import infer, load_field
from PIL import Image, ImageDraw
from quartic_field_objective import field_parts
from subpixel_field_data import candidate_fields, central_field, geometry_field
from subpixel_field_model import coverage, field_forward

PARENT = Path("../jax-forced-geometry-20261015/runs/forced-geometry-20261015").resolve()
MODEL = PARENT / "snapshots/step-6000/supported/raw"
DOC = Path("docs/field-repair-plan-20261015.md").resolve()
SAVED = (0, 1, 25, 100, 500)
CASES = ("scene7-whole", "scene7-partial", "probe37-whole", "probe37-partial")


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_forced_geometry"
    root.mkdir(parents=True, exist_ok=False)
    records = read(PARENT / "data.json")["evaluation"]
    probe = next(r for r in records if r["split"] == "new_probe" and r["index"] == 37)
    scene = next(r for r in read(PARENT / "scene512/plan.json")["records"] if r["index"] == 7)
    paths = [DOC, PARENT / "completion.json", PARENT / "data.json", PARENT / "scene512/plan.json"]
    paths += list(MODEL.iterdir())
    paths += [PARENT / probe["file"], PARENT / "scene512/input-7.npz"]
    for mode in ("whole", "partial"):
        paths += [PARENT / f"scene512/predictions-7-{mode}-0.npz"]
        p_mode = "whole05" if mode == "whole" else "partial05"
        paths += [PARENT / f"evaluation/predictions/new_probe-037-{p_mode}-0.npz"]
    paths += [
        Path(f).resolve()
        for f in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if f.endswith((".py", ".toml")) or f == "uv.lock"
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            scene=scene,
            probe=probe,
            cases=CASES,
            steps=500,
            learning_rate=0.02,
            saved=SAVED,
            bindings={str(p): sha(p) for p in paths},
            archive_sha256=sha(root / "source.tar.gz"),
            cnn_training=False,
            adoption=False,
            fresh_confirmation=False,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for p, h in plan["bindings"].items():
        assert sha(Path(p)) == h, p
    assert sha(root / "source.tar.gz") == plan["archive_sha256"]
    return plan


def inputs(plan, case):
    name, mode = case.split("-")
    if name == "scene7":
        r, size, depth = plan["scene"], 512, 6
        with np.load(PARENT / "scene512/input-7.npz") as z:
            source, clean = z["quantized"], z["clean"]
        with np.load(PARENT / f"scene512/predictions-7-{mode}-0.npz") as z:
            expected = z["supported_uint8"]
        metadata = r["metadata"]
    else:
        r, size, depth = plan["probe"], 256, 3
        with np.load(PARENT / r["file"]) as z:
            source, clean = z["source"], z["variants"][13]
        p_mode = "whole05" if mode == "whole" else "partial05"
        with np.load(PARENT / f"evaluation/predictions/new_probe-037-{p_mode}-0.npz") as z:
            expected = z["supported"]
        metadata = copy.deepcopy(r["metadata"])
        a, b = metadata["path_geometry"]
        points = (np.asarray(a["points"]) + np.asarray(b["points"])) / 2
        metadata["path_geometry"] = [
            dict(kind="bezier", points=points.tolist(), widths=[2.0] * len(points))
        ]
        metadata["paths"] = 1
    yy, xx = np.indices((size, size))
    strength = (
        np.full_like(source, 0.5)
        if mode == "whole"
        else np.where((xx > size * 0.6 - yy * 0.25)[..., None], 0.5, 0).astype(np.float32)
    )
    args = tuple(jnp.asarray(x) for x in geometry(metadata, size=size))
    base = geometry_field(*args, size=size)[None]
    oracle = coverage(central_field(base)[..., None])[0]
    np.testing.assert_allclose(oracle, render(*args, size=size), atol=1e-6, rtol=1e-6)
    prepared = prepare_inputs(
        clean[None], source[None], strength[None], jax.random.PRNGKey(2026101517), depth=depth
    )
    direction = jnp.asarray(r.get("direction", [1.0, 0.0]), dtype=jnp.float32)[None]
    return (
        prepared,
        candidate_fields(base, direction, depth),
        source,
        strength,
        expected,
        np.asarray(oracle),
    )


def objective(fields, targets, strengths):
    parts = field_parts(fields, targets, strengths, sign_power=4)
    return parts["loss"], parts


OPTIMIZER = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.02))


@jax.jit
def step(fields, state, targets, strengths):
    (_, parts), grads = jax.value_and_grad(objective, has_aux=True)(fields, targets, strengths)
    updates, state = OPTIMIZER.update(grads, state, fields)
    fields = optax.apply_updates(fields, updates)
    return fields, state, parts | dict(gradient_norm=optax.global_norm(grads))


@nnx.jit
def predict_fields(model, prepared):
    return field_forward(model, prepared)


def put(canvas, array, x, y, box=None):
    im = Image.fromarray(
        np.rint(np.clip((array.squeeze() + 1) * 127.5, 0, 255)).astype(np.uint8)
    ).convert("RGB")
    canvas.paste(im.crop(box) if box else im, (x, y))


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    model, _ = load_field(MODEL)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(nnx.state(model, nnx.Param))]
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    all_rows, records, audits = [], [], []
    for case in CASES:
        p, targets, source, strength, expected, oracle = inputs(plan, case)
        initial = predict_fields(model, p)
        np.testing.assert_array_equal(np.asarray(infer(model, p))[0], expected)
        initial_render = np.where(strength > 0, np.asarray(coverage(initial[0]))[0], source)
        # Exposing the field splits the fused inference graph. Use the previously
        # established ROCm replay envelope; retain an exact fused-infer control.
        np.testing.assert_allclose(initial_render, expected, atol=5e-6, rtol=1e-4)
        initial_render_error = float(np.max(abs(initial_render - expected)))
        fields = initial
        state = OPTIMIZER.init(fields)
        values = {"source": source, "oracle": np.where(strength > 0, oracle, source)}
        prefix = None
        for k in range(501):
            if k in SAVED:
                pred = np.where(strength > 0, np.asarray(coverage(fields[0]))[0], source)
                np.testing.assert_array_equal(pred[strength == 0], source[strength == 0])
                assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
                values[str(k)] = pred
                stats = {
                    n: float(v) for n, v in objective(fields, targets, p.noise_levels)[1].items()
                }
                all_rows.append(dict(case=case, step=k, **stats))
            if k == 6:
                prefix = (fields, state)
            if k == 500:
                break
            fields, state, stats = step(fields, state, targets, p.noise_levels)
            assert all(np.isfinite(float(v)) for v in stats.values())
            if (k + 1) % 100 == 0:
                print(dict(case=case, update=k + 1), flush=True)
        replay, rs = initial, OPTIMIZER.init(initial)
        for _ in range(6):
            replay, rs, _ = step(replay, rs, targets, p.noise_levels)
        jax.tree.map(np.testing.assert_array_equal, (replay, rs), prefix)
        leaves, tree = jax.tree.flatten((fields, state))
        state_file = out / f"{case}-state.npz"
        np.savez_compressed(state_file, **{f"leaf{i}": np.asarray(x) for i, x in enumerate(leaves)})
        with np.load(state_file) as z:
            restored = jax.tree.unflatten(tree, [z[f"leaf{i}"] for i in range(len(leaves))])
        jax.tree.map(np.testing.assert_array_equal, (fields, state), restored)
        np.savez_compressed(out / f"{case}.npz", **values)
        size = source.shape[0]
        labels = ("source", "oracle") + tuple(str(k) for k in SAVED)
        canvas = Image.new("RGB", (7 * size, size + 28), "white")
        pen = ImageDraw.Draw(canvas)
        for col, label in enumerate(labels):
            put(canvas, values[label], col * size, 28)
            pen.text((col * size + 4, 6), label, fill="black")
        file = out / f"{case}-native.png"
        canvas.save(file)
        records.append(dict(file=file.name, sha256=sha(file), native=True))
        box = (292, 140, 390, 238) if case.startswith("scene") else (40, 88, 112, 161)
        w, h = box[2] - box[0], box[3] - box[1]
        canvas = Image.new("RGB", (7 * (w + 12), h + 28), "white")
        pen = ImageDraw.Draw(canvas)
        for col, label in enumerate(labels):
            put(canvas, values[label], col * (w + 12), 28, box)
            pen.text((col * (w + 12) + 4, 6), label, fill="black")
        file = out / f"{case}-detail.png"
        canvas.save(file)
        records.append(dict(file=file.name, sha256=sha(file), native=True, box=box))
        audits.append(
            dict(
                case=case,
                initial_prediction_exact=True,
                split_graph_render_max_abs_error=initial_render_error,
                split_graph_render_atol=5e-6,
                split_graph_render_rtol=1e-4,
                oracle_render_close=True,
                prefix6_full_state_exact=True,
                final_full_state_restore_exact=True,
            )
        )
        write(out / "progress.json", dict(cases_complete=len(audits), case=case))
    for a, b in zip(before, jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True):
        np.testing.assert_array_equal(a, b)
    write(
        out / "manifest.json", dict(records=records, thresholding=False, contrast_rescaling=False)
    )
    write(
        out / "metrics.json",
        dict(
            rows=all_rows,
            audits=audits,
            model_unchanged=True,
            protected_exact=True,
            outputs=20,
            bindings={str(p): sha(p) for p in out.glob("*.npz")},
        ),
    )


def finalize(root):
    check(root)
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    assert read(root / "test-results.json")["passed"]
    metrics = read(root / "evaluation/metrics.json")
    assert metrics["model_unchanged"] and metrics["protected_exact"] and len(metrics["audits"]) == 4
    for f, h in metrics["bindings"].items():
        assert sha(Path(f)) == h
    notes = read(root / "evaluation/review.json")
    records = read(root / "evaluation/manifest.json")["records"]
    assert len(records) == len(notes) == 8
    for record in records:
        assert sha(root / "evaluation" / record["file"]) == record["sha256"]
        assert notes[record["file"]]
    files = [
        "plan.json",
        "driver-evaluate.json",
        "test-results.json",
        "evaluation/metrics.json",
        "evaluation/manifest.json",
        "evaluation/review.json",
        "decision.json",
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_field_repair",
            cnn_training=False,
            free_field_updates=500,
            cases=4,
            outputs=20,
            native_pages_reviewed=8,
            adoption=False,
            fresh_confirmation=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.phase](args.root.resolve())
