"""Fit the existing small CNN to the two known direct-field diagnostic shapes."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from consolidated_geometry_study import replay_state, snapshot
from context_dropout_objective import context_mask, dropout_forward
from cue_capacity import make_arm
from field_repair_study import CASES, MODEL, inputs, objective, put
from flax import nnx
from physical_field_model import infer, load_field, save_field
from PIL import Image, ImageDraw
from plausible_study import digest, exact
from scale_style_balance_study import update

from lineartgen.training import checkpoints, lineart_state

PARENT = Path("../jax-field-repair-20261015/runs/field-repair-20261015-retry1").resolve()
DOC = Path("docs/field-fit-plan-20261015.md").resolve()
SAVED = (0, 12, 400, 800, 1200)


def select(k):
    return dict(case=("scene7-whole", "probe37-whole")[(k // 6) % 2], depth=k % 6 + 1)


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_field_repair"
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
    save_field(model, cfg, root / "initial", provenance="Known two-shape fitting diagnostic only")
    paths = [
        DOC,
        PARENT / "completion.json",
        PARENT / "plan.json",
        PARENT / "evaluation/metrics.json",
    ]
    paths += list((root / "initial").iterdir())
    paths += [Path(p) for p in read(PARENT / "plan.json")["bindings"]]
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
            bindings={str(p): sha(p) for p in paths},
            archive_sha256=sha(root / "source.tar.gz"),
            steps=1200,
            saved=SAVED,
            cases=CASES,
            training_cases=2,
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


@nnx.jit
def gradient(model, prepared, targets_keep, annotation):
    targets, keep = targets_keep

    def loss(m):
        return objective(dropout_forward(m, prepared, keep), targets, prepared.noise_levels)

    (_, parts), grads = nnx.value_and_grad(loss, has_aux=True)(model)
    return parts, grads


def arms_from_root(root):
    model, cfg = load_field(root / "initial")
    arm = make_arm(cfg, model)
    arm["gradient"] = gradient
    return dict(fit=arm)


def data():
    plan = read(PARENT / "plan.json")
    return {
        case: inputs(plan, case, depth_override=6)[:2] for case in ("scene7-whole", "probe37-whole")
    }


def batch(data, k):
    s = select(k)
    p, t = data[s["case"]]
    d = s["depth"]
    p = type(p)(*(x[:d] for x in p))
    return p, (t[:d], jnp.asarray(context_mask(k, d, 1, enabled=True))), None


def train(root, until):
    check(root)
    assert jax.devices()[0].platform != "cpu" and until in SAVED[2:]
    assert read(root / "test-results.json")["passed"]
    out = root / "training"
    out.mkdir(exist_ok=True)
    d, arms = data(), arms_from_root(root)
    assert sum(x.size for x in jax.tree.leaves(nnx.state(arms["fit"]["model"], nnx.Param))) == 16402
    rows = []
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            rows = [
                json.loads(x)
                for x in journal.read_text().splitlines()
                if json.loads(x)["step"] <= start
            ]
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert [r["step"] for r in rows] == list(range(1, start + 1))
        if start == 0:
            audit = []
            for case in CASES:
                p, _, _, _, expected, _ = inputs(read(PARENT / "plan.json"), case)
                np.testing.assert_array_equal(
                    np.asarray(infer(arms["fit"]["model"], p))[0], expected
                )
                audit.append(case)
            write(root / "initial-audit.json", dict(predictions_exact=audit, parameters=16402))
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
        assert start < until
        for k in range(start, until):
            p, t, ann = batch(d, k)
            tick = time.monotonic()
            metrics = update(arms["fit"], p, t, ann)
            row = dict(
                step=k + 1,
                **select(k),
                metrics=metrics,
                seconds=time.monotonic() - tick,
                input_target_mask_sha256=digest((p, t)),
                context_keep=np.asarray(t[1]).tolist(),
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k < 12 or (k + 1) % 100 == 0:
                write(out / "progress.json", row)
                print(dict(step=k + 1, loss=metrics["loss"]), flush=True)
            if k + 1 == 12:
                replay = arms_from_root(root)
                audits = [
                    replay_state(update(replay["fit"], *batch(d, j)), rows[j]["metrics"])
                    for j in range(12)
                ]
                state_audit = replay_state(
                    lineart_state.payload(replay, 12), lineart_state.payload(arms, 12)
                )
                write(
                    root / "prefix-audit.json",
                    dict(passed=True, metrics=audits, full_state=state_audit),
                )
            if k + 1 in SAVED:
                lineart_state.save(saved, arms, k + 1)
                snapshot(root, arms, k + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(saved, restored) == k + 1
                exact(lineart_state.payload(arms, k + 1), lineart_state.payload(restored, k + 1))
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
        exports = []
        for s in [v for v in SAVED if v <= until]:
            restored = arms_from_root(root)
            assert lineart_state.restore(saved, restored, s) == s
            for mode in ("raw", "ema"):
                path = root / f"snapshots/step-{s}/fit/{mode}"
                m, c = load_field(path)
                assert c == restored["fit"]["config"]
                expected = (
                    nnx.state(restored["fit"]["model"], nnx.Param)
                    if mode == "raw"
                    else restored["fit"]["ema"]
                )
                exact(nnx.state(m, nnx.Param), expected)
                exports.append(str(path))
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=exports, until=until))


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["until"] == 1200
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    models = {s: load_field(root / f"snapshots/step-{s}/fit/raw")[0] for s in (0, 400, 800, 1200)}
    records = []
    bindings = {}
    for case in CASES:
        p, _, source, strength, expected, oracle = inputs(read(PARENT / "plan.json"), case)
        values = dict(source=source, oracle=np.where(strength > 0, oracle, source))
        for s, model in models.items():
            pred = np.asarray(infer(model, p))[0]
            assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
            np.testing.assert_array_equal(pred[strength == 0], source[strength == 0])
            if s == 0:
                np.testing.assert_array_equal(pred, expected)
            values[str(s)] = pred
        file = out / f"{case}.npz"
        np.savez_compressed(file, **values)
        bindings[str(file)] = sha(file)
        box = (292, 140, 390, 238) if case.startswith("scene") else (40, 88, 112, 161)
        for kind, crop in (("native", None), ("detail", box)):
            w, h = (
                (source.shape[1], source.shape[0])
                if crop is None
                else (box[2] - box[0], box[3] - box[1])
            )
            canvas = Image.new("RGB", (6 * (w + 12), h + 28), "white")
            pen = ImageDraw.Draw(canvas)
            for col, (label, v) in enumerate(values.items()):
                put(canvas, v, col * (w + 12), 28, crop)
                pen.text((col * (w + 12) + 4, 6), label, fill="black")
            file = out / f"{case}-{kind}.png"
            canvas.save(file)
            records.append(dict(file=file.name, sha256=sha(file), native=True, crop=crop))
    write(
        out / "manifest.json", dict(records=records, thresholding=False, contrast_rescaling=False)
    )
    write(
        out / "metrics.json",
        dict(outputs=16, protected_exact=True, initial_exact=True, bindings=bindings),
    )


def finalize(root):
    check(root)
    for phase in ("train-400", "train-800", "train-1200", "evaluate"):
        assert read(root / f"driver-{phase}.json")["returncode"] == 0
    assert read(root / "prefix-audit.json")["passed"]
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    assert len(read(root / "snapshot-audit.json")["exports"]) == 10
    rows = [json.loads(x) for x in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert len(rows) == 1200
    for k, row in enumerate(rows):
        assert row["step"] == k + 1 and all(row[n] == v for n, v in select(k).items())
        assert row["context_keep"] == context_mask(k, select(k)["depth"], 1, enabled=True).tolist()
    for f, h in read(root / "evaluation/metrics.json")["bindings"].items():
        assert sha(Path(f)) == h
    notes = read(root / "evaluation/review.json")
    records = read(root / "evaluation/manifest.json")["records"]
    assert len(notes) == len(records) == 8
    for record in records:
        assert (
            notes[record["file"]] and sha(root / "evaluation" / record["file"]) == record["sha256"]
        )
    paths = [p for p in root.rglob("*.json") if p.name != "completion.json"] + [
        root / "training/sampling.jsonl"
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_field_fit",
            steps=1200,
            known_training_shapes=2,
            adoption=False,
            fresh_confirmation=False,
            native_pages_reviewed=8,
            evidence={str(p.relative_to(root)): sha(p) for p in paths},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "train", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--until", type=int, choices=(400, 800, 1200))
    args = p.parse_args()
    if args.phase == "train":
        train(args.root.resolve(), args.until)
    else:
        globals()[args.phase](args.root.resolve())
