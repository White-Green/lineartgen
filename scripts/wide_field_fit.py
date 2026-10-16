"""Known-case learnability diagnostic: broad strokes with/without opening prior."""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from consolidated_geometry_study import replay_state, snapshot
from context_dropout_objective import context_mask
from cue_capacity import make_arm, prepare_inputs
from field_repair_study import put
from flax import nnx
from opaque_geometry import geometry, render
from opening_prior_objective import opening_gradient
from physical_field_model import infer, load_field, save_field
from PIL import Image, ImageDraw
from plausible_data import raster
from plausible_study import digest, exact
from rough_family_data import BASE, rough_source
from scale_style_balance_study import update
from subpixel_field_data import central_field, geometry_field
from subpixel_field_model import coverage
from width_response_study import SEED

from lineartgen.training import checkpoints, lineart_state

PARENT = Path("../jax-wide-source-20261016/runs/wide-source-20261016").resolve()
DIAG = Path("../jax-wide-recursion-20261016/runs/wide-recursion-20261016").resolve()
MODEL = PARENT / "snapshots/step-6000/expanded/raw"
DOC = Path("docs/wide-field-fit-plan-20261016.md").resolve()
METHODS = ("opening", "no_opening")
CASES = (59, 61, 63)
SAVED = (0, 18, 400, 800, 1200)


def select(k):
    return dict(case=CASES[(k // 6) % 3], depth=k % 6 + 1)


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_wide_source"
    assert read(DIAG / "completion.json")["stage"] == "complete_wide_recursion"
    root.mkdir(parents=True, exist_ok=False)
    model, cfg = load_field(MODEL)
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=1200,
                    learning_rate_decay_steps=1200,
                    batch_size=1,
                    accumulate_steps=1,
                    learning_rate=2e-4,
                    minimum_learning_rate=2e-5,
                    warmup_steps=0,
                )
            )
        )
    )
    assert cfg.train.peak_learning_rate == 2e-4
    save_field(
        model, cfg, root / "initial", provenance="Known three-wide-shape fit only; never adopt"
    )
    records = [
        r
        for r in read(PARENT / "data.json")["evaluation"]
        if r["split"] == "new_probe" and r["index"] in CASES
    ]
    assert len(records) == 3
    paths = [
        DOC,
        PARENT / "completion.json",
        PARENT / "data.json",
        DIAG / "completion.json",
        DIAG / "decision.json",
    ] + list((root / "initial").iterdir())
    for i in CASES:
        paths += [PARENT / f"width-sweep/wide-{i}-factor-2.0.npz"]
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
            records=records,
            bindings={str(p): sha(p) for p in paths},
            archive_sha256=sha(root / "source.tar.gz"),
            steps=1200,
            saved=SAVED,
            cases=CASES,
            methods=METHODS,
            opening_weights={"opening": 0.1, "no_opening": 0},
            adoption=False,
            fresh_confirmation=False,
        ),
    )


def check(root):
    p = read(root / "plan.json")
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f
    assert sha(root / "source.tar.gz") == p["archive_sha256"]
    return p


def data(root):
    result = {}
    for r in check(root)["records"]:
        i = r["index"]
        metadata = copy.deepcopy(r["metadata"])
        for path in metadata["path_geometry"]:
            path["widths"] = (np.asarray(path["widths"]) * 2).tolist()
        direction = np.asarray(r["direction"], np.float32)
        # Source raster originally used the record's float64 direction.
        mixture = np.stack(
            [raster(metadata, np.asarray(r["direction"]), offset, 1) for offset in (-2.0, 2.0)]
        ).mean(0)
        source = rough_source(mixture, BASE)
        with np.load(PARENT / f"width-sweep/wide-{i}-factor-2.0.npz") as z:
            np.testing.assert_array_equal(source, z["source"])
            expected = z["expanded"]
        args = tuple(jnp.asarray(x) for x in geometry(r["metadata"], kind=r["kind"], size=256))
        assert np.all(np.asarray(args[1]) == 2)
        base = geometry_field(*args, size=256)[None]
        oracle = np.asarray(coverage(central_field(base)[..., None]))[0]
        np.testing.assert_allclose(oracle, render(*args, size=256), atol=1e-6, rtol=1e-6)
        p = prepare_inputs(
            oracle[None], source[None], np.full_like(source[None], 0.5), jax.random.PRNGKey(SEED), 6
        )
        result[i] = dict(
            prepared=p,
            base=base,
            direction=jnp.asarray(direction[None]),
            source=source,
            oracle=oracle,
            expected=expected,
        )
    return result


def arms_from_root(root):
    arms = {}
    for method in METHODS:
        model, cfg = load_field(root / "initial")
        assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 16402
        arms[method] = make_arm(cfg, model)
        arms[method]["gradient"] = opening_gradient(weight=0.1 if method == "opening" else 0)
    return arms


def batch(d, k):
    s = select(k)
    v = d[s["case"]]
    depth = s["depth"]
    p = v["prepared"]
    p = type(p)(*(x[:depth] for x in p))
    keep = jnp.asarray(context_mask(k, depth, 1, enabled=True))
    return p, (v["base"], v["base"], jnp.zeros((1,), dtype=bool), keep), v["direction"]


def audit(root):
    d = data(root)
    arms = arms_from_root(root)
    out = root / "teacher-preview"
    out.mkdir(exist_ok=False)
    canvas = Image.new("RGB", (768, 3 * 284), "white")
    pen = ImageDraw.Draw(canvas)
    bindings = {}
    for row, (i, v) in enumerate(d.items()):
        p = v["prepared"]
        p = type(p)(*(x[:3] for x in p))
        for arm in arms.values():
            np.testing.assert_array_equal(np.asarray(infer(arm["model"], p))[0], v["expected"])
        for col, (label, a) in enumerate(
            [
                ("source", v["source"]),
                ("opaque2px teacher", v["oracle"]),
                ("FO initial", v["expected"]),
            ]
        ):
            put(canvas, a, col * 256, row * 284 + 28)
            pen.text((col * 256 + 4, row * 284 + 5), f"{i} {label}", fill="black")
        f = out / f"{i}.npz"
        np.savez_compressed(
            f,
            source=v["source"],
            oracle=v["oracle"],
            expected=v["expected"],
            base=np.asarray(v["base"]),
        )
        bindings[str(f)] = sha(f)
    f = out / "native.png"
    canvas.save(f)
    write(
        out / "manifest.json", dict(file=f.name, sha256=sha(f), arrays=bindings, native_scale=True)
    )
    write(
        root / "initial-audit.json",
        dict(
            predictions_exact=6,
            source_reconstruction_exact=3,
            geometry_render_parity=True,
            parameters=16402,
        ),
    )


def train(root, until):
    check(root)
    assert jax.devices()[0].platform != "cpu" and until in SAVED[2:]
    review = read(root / "teacher-preview/review.json")
    assert review["reviewed"]
    manifest = read(root / "teacher-preview/manifest.json")
    assert (
        sha(root / "teacher-preview" / manifest["file"]) == manifest["sha256"] == review["sha256"]
    )
    for f, h in manifest["arrays"].items():
        assert sha(Path(f)) == h
    out = root / "training"
    out.mkdir(exist_ok=True)
    d = data(root)
    arms = arms_from_root(root)
    rows = []
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            old = [json.loads(x) for x in journal.read_text().splitlines()]
            if any(x["step"] > start for x in old):
                attempt = out / f"interrupted-{time.time_ns()}.json"
                write(attempt, old)
            rows = [x for x in old if x["step"] <= start]
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert [r["step"] for r in rows] == list(range(1, start + 1))
        if start == 0:
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        assert start < until
        for k in range(start, until):
            b = batch(d, k)
            metrics = {}
            seconds = {}
            for m in METHODS:
                tick = time.monotonic()
                metrics[m] = update(arms[m], *b)
                seconds[m] = time.monotonic() - tick
            row = dict(
                step=k + 1,
                **select(k),
                metrics=metrics,
                seconds=seconds,
                input_target_mask_sha256=digest(b),
                context_keep=np.asarray(b[1][-1]).tolist(),
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k < 18 or (k + 1) % 100 == 0:
                write(out / "progress.json", row)
                print(dict(step=k + 1), flush=True)
            if k + 1 == 18:
                replay = arms_from_root(root)
                audits = []
                for j in range(18):
                    audits.append(
                        {
                            m: replay_state(update(replay[m], *batch(d, j)), rows[j]["metrics"][m])
                            for m in METHODS
                        }
                    )
                state = replay_state(
                    lineart_state.payload(replay, 18), lineart_state.payload(arms, 18)
                )
                write(
                    root / "prefix-audit.json", dict(passed=True, metrics=audits, full_state=state)
                )
            if k + 1 in SAVED:
                lineart_state.save(saved, arms, k + 1)
                snapshot(root, arms, k + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(saved, restored) == k + 1
                exact(lineart_state.payload(arms, k + 1), lineart_state.payload(restored, k + 1))
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
        exports = {}
        for s in [v for v in SAVED if v <= until]:
            restored = arms_from_root(root)
            assert lineart_state.restore(saved, restored, s) == s
            for method in METHODS:
                for mode in ("raw", "ema"):
                    path = root / f"snapshots/step-{s}/{method}/{mode}"
                    m, c = load_field(path)
                    assert c == restored[method]["config"]
                    expected = (
                        nnx.state(restored[method]["model"], nnx.Param)
                        if mode == "raw"
                        else restored[method]["ema"]
                    )
                    exact(nnx.state(m, nnx.Param), expected)
                    exports.update({str(f): sha(f) for f in path.iterdir() if f.is_file()})
        write(
            root / "snapshot-audit.json",
            dict(
                raw_ema_exact=True,
                export_count=len([s for s in SAVED if s <= until]) * 4,
                bindings=exports,
                until=until,
            ),
        )


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["until"] == 1200
    d = data(root)
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    pages = []
    bindings = {}
    models = {
        (m, s): load_field(root / f"snapshots/step-{s}/{m}/raw")[0]
        for m in METHODS
        for s in (0, 400, 800, 1200)
    }
    states = {str(k): digest(nnx.state(m, nnx.Param)) for k, m in models.items()}
    for i, v in d.items():
        for depth in (1, 3, 6):
            p = v["prepared"]
            p = type(p)(*(x[:depth] for x in p))
            arrays = dict(source=v["source"], oracle=v["oracle"])
            for (m, s), model in models.items():
                a = np.asarray(infer(model, p))[0]
                assert np.isfinite(a).all() and a.min() >= -1 and a.max() <= 1
                if s == 0 and depth == 3:
                    np.testing.assert_array_equal(a, v["expected"])
                arrays[f"{m}_{s}"] = a
            f = out / f"{i}-d{depth}.npz"
            np.savez_compressed(f, **arrays)
            bindings[str(f)] = sha(f)
            y, x = np.where(v["source"][..., 0] < 0.98)
            box = (
                max(0, int(x.min()) - 8),
                max(0, int(y.min()) - 8),
                min(256, int(x.max()) + 9),
                min(256, int(y.max()) + 9),
            )
            w, h = box[2] - box[0], box[3] - box[1]
            stride = max(w, 130)
            canvas = Image.new("RGB", (stride * 6, 2 * (h + 30)), "white")
            pen = ImageDraw.Draw(canvas)
            for row, m in enumerate(METHODS):
                values = [("source", v["source"]), ("teacher", v["oracle"])] + [
                    (f"{m} {s}", arrays[f"{m}_{s}"]) for s in (0, 400, 800, 1200)
                ]
                for col, (label, a) in enumerate(values):
                    put(canvas, a, col * stride, row * (h + 30) + 25, box)
                    pen.text((col * stride + 2, row * (h + 30) + 4), label, fill="black")
            page = out / f"native-{i}-d{depth}.png"
            canvas.save(page)
            pages.append(dict(file=page.name, sha256=sha(page), crop=box, native_scale=True))
    assert states == {str(k): digest(nnx.state(m, nnx.Param)) for k, m in models.items()}
    write(
        out / "metrics.json",
        dict(outputs=72, initial_exact=True, states_unchanged=True, bindings=bindings),
    )
    write(
        out / "manifest.json",
        dict(records=pages, native_scale=True, thresholding=False, contrast_rescaling=False),
    )


def finalize(root):
    check(root)
    for phase in ("audit", "train-400", "train-800", "train-1200", "evaluate"):
        assert read(root / f"driver-{phase}.json")["returncode"] == 0
    assert read(root / "prefix-audit.json")["passed"]
    audit = read(root / "snapshot-audit.json")
    assert audit["raw_ema_exact"] and audit["export_count"] == 20
    for f, h in audit["bindings"].items():
        assert sha(Path(f)) == h
    rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert len(rows) == 1200
    for k, r in enumerate(rows):
        assert r["step"] == k + 1 and all(r[n] == v for n, v in select(k).items())
        assert r["context_keep"] == context_mask(k, select(k)["depth"], 1, enabled=True).tolist()
    for s in SAVED[1:]:
        assert read(root / f"training/restore-{s}.json")["full_state_exact"]
    for f, h in read(root / "evaluation/metrics.json")["bindings"].items():
        assert sha(Path(f)) == h
    notes = read(root / "evaluation/review.json")
    records = read(root / "evaluation/manifest.json")["records"]
    assert len(notes) == len(records) == 9
    for r in records:
        assert notes[r["file"]] and sha(root / "evaluation" / r["file"]) == r["sha256"]
    paths = [p for p in root.rglob("*.json") if p.name != "completion.json"] + [
        root / "training/sampling.jsonl"
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_wide_field_fit",
            steps=1200,
            known_training_shapes=3,
            trained_arms=2,
            adoption=False,
            fresh_confirmation=False,
            native_pages_reviewed=9,
            evidence={str(p.relative_to(root)): sha(p) for p in paths},
        ),
    )
    print("Completed known-wide fitting diagnostic", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "audit", "train", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--until", type=int, choices=(400, 800, 1200))
    a = p.parse_args()
    if a.phase == "train":
        train(a.root.resolve(), a.until)
    else:
        globals()[a.phase](a.root.resolve())
