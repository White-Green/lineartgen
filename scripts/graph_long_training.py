"""Long continuation of the tested IJ graph, with resumable state and fixed reviews."""

import argparse
import datetime
import fcntl
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import width_exposure_study as previous
from attraction_augmentation import target_action
from attraction_stress import infer
from campaign_protocol import read, sha, write
from cue_capacity import make_arm, prepare_inputs
from fill_negative_objective import hard_negative_gradient
from flax import nnx
from learned_centerline_model import load_weights, save_weights
from learned_centerline_study import PARTIAL, crop, digest, halo, make_model, strength
from PIL import Image, ImageDraw
from plausible_study import exact
from scipy.ndimage import gaussian_filter

from lineartgen.training import checkpoints, lineart_state

PARENT = Path("../jax-width-exposure-20261019/runs/width-exposure-20261019").resolve()
STEPS = 84000
SEED = 2026101910
REVIEWS = (0, 2000, 12000, 36000, 60000, STEPS)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    initial = PARENT / "snapshots/step-6000/control/raw.npz"
    assert (
        sha(initial)
        == read(PARENT / "completion.json")["evidence"][str(initial.relative_to(PARENT))]
    )
    shutil.copy2(initial, root / "initial.npz")
    cfg = read(PARENT / "config.json")
    cfg["train"].update(
        steps=STEPS,
        learning_rate_decay_steps=STEPS,
        minimum_learning_rate=2e-6,
        warmup_steps=600,
        seed=SEED,
    )
    write(root / "config.json", cfg)
    shutil.copy2(PARENT / "training-data.json", root / "training-data.json")
    files = [root / n for n in ("initial.npz", "config.json", "training-data.json")]
    files += [Path("scripts/graph_long_training.py"), Path("docs/graph-krita-plan-20261019.md")]
    files += [Path(p) for p in read(PARENT / "plan.json")["code"]]
    files += [Path(p) for p in read(PARENT / "plan.json")["bindings"] if Path(p).suffix == ".npz"]
    write(
        root / "plan.json",
        dict(
            started_at=now(),
            parent=str(initial),
            parent_steps=16200,
            added_steps=STEPS,
            total_steps=16200 + STEPS,
            reviews=REVIEWS,
            seed=SEED,
            bindings={str(p.resolve()): sha(p) for p in files},
            selection="IJ control: retain 25% extra shapes, not selected by image MSE",
        ),
    )


def check(root):
    for p, expected in read(root / "plan.json")["bindings"].items():
        assert sha(Path(p)) == expected, p


def arms(root):
    model, config = make_model(root)
    arm = make_arm(config, model)
    arm.update(gradient=hard_negative_gradient, target_key="graph")
    return {"graph": arm}


def batch(data, step):
    previous.DATA_SEED = SEED
    audit = previous.sampling(step, len(data["legacy"]), len(data["extra"]))
    items = [
        data["extra"][audit["extra_indices"][k]]
        if audit["replace"]["control"][k]
        else data["legacy"][i]
        for k, i in enumerate(audit["indices"])
    ]
    source = np.stack(
        [
            halo(
                1
                - 2
                * audit["opacity"][k]
                * gaussian_filter(
                    (1 - d["source"]) / 2,
                    (audit["sigma"][k], audit["sigma"][k], 0),
                    mode="constant",
                    cval=0,
                )
            )
            for k, d in enumerate(items)
        ]
    ).astype(np.float32)
    masks = np.stack([halo(strength(256, v), 0 if v else 0.5) for v in audit["partial"]])
    prepared = prepare_inputs(
        jnp.asarray(np.stack([d["graph_oracle"] for d in items])),
        jnp.asarray(source),
        jnp.asarray(masks),
        jax.random.fold_in(jax.random.PRNGKey(SEED), step),
        audit["depth"],
    )
    prepared = previous.augment_prepared(prepared, audit["groups"])
    targets = tuple(
        jnp.asarray(
            np.stack(
                [
                    target_action(d[f"smalltarget{level}"], g)
                    for d, g in zip(items, audit["groups"], strict=True)
                ]
            )
        )
        for level in range(audit["depth"])
    )
    audit.update(input_sha256=digest(prepared), target_sha256=digest(targets))
    return {"graph": prepared}, {"graph": targets}, audit


def preflight(root):
    check(root)
    data = previous.load_training(root)
    a, b = arms(root), arms(root)
    records = []
    for step in range(6):
        p, t, audit = batch(data, step)
        pp, tt, _ = previous.matched_batch(data, step)
        exact(p["graph"], pp["control"])
        exact(t["graph"], tt["control"])
        x = previous.update(a["graph"], p, t)
        y = previous.update(b["graph"], p, t)
        exact(x, y)
        exact(lineart_state.payload(a, step + 1), lineart_state.payload(b, step + 1))
        records.append(dict(step=step, metrics=x, **audit))
    with checkpoints.manager(root / "preflight-checkpoints", keep=1) as manager:
        lineart_state.save(manager, a, 6)
        restored = arms(root)
        lineart_state.restore(manager, restored, 6)
        exact(lineart_state.payload(a, 6), lineart_state.payload(restored, 6))
    write(
        root / "preflight.json",
        dict(
            passed=True,
            batch_equivalence=True,
            state_replay_exact=True,
            checkpoint_roundtrip_exact=True,
            rows=records,
        ),
    )


def snapshot(root, models, step, manager):
    lineart_state.save(manager, models, step)
    for mode in ("raw", "ema"):
        m = nnx.clone(models["graph"]["model"])
        if mode == "ema":
            nnx.update(m, models["graph"]["ema"])
        path = root / "snapshots" / f"step-{step}" / f"{mode}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_weights(m, path)
        restored, _ = make_model(root)
        load_weights(restored, path)
        exact(nnx.state(m, nnx.Param), nnx.state(restored, nnx.Param))


def review(root, step):
    folder = root / "reviews" / f"step-{step}"
    if (folder / "manifest.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    model, _ = make_model(root)
    load_weights(model, root / "snapshots" / f"step-{step}" / "raw.npz")
    records = []
    for r in read(PARENT / "plan.json")["records"]:
        with np.load(PARENT / r["file"]) as z:
            source, teacher = z["source"], crop(z["graph_oracle"])
        panels = [source, teacher]
        labels = ["source", "teacher (one possible shape)"]
        outputs = {}
        for depth, mode in [(1, "whole"), (3, "whole"), (6, "whole")] + (
            [(3, "partial")] if r["name"] in PARTIAL else []
        ):
            output, _, stats = infer(model, source, depth, mode, 0.5, 1.0)
            label = f"d{depth}-{mode}"
            outputs[label] = output
            panels.append(output)
            labels.append(label)
        n = source.shape[0]
        canvas = Image.new("RGB", (n * len(panels), n + 28), "#ddd")
        draw = ImageDraw.Draw(canvas)
        for i, (im, label) in enumerate(zip(panels, labels, strict=True)):
            draw.text((i * n + 3, 3), label, fill="black")
            gray = np.rint(np.clip((im[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
            canvas.paste(Image.fromarray(gray).convert("RGB"), (i * n, 28))
        p = folder / (r["name"] + ".png")
        canvas.save(p)
        np.savez_compressed(folder / (r["name"] + ".npz"), **outputs)
        records.append(dict(file=p.name, sha256=sha(p)))
    write(
        folder / "manifest.json",
        dict(added_steps=step, total_steps=16200 + step, records=records, visually_reviewed=False),
    )


def train(root):
    check(root)
    assert read(root / "preflight.json")["passed"]
    data, models = previous.load_training(root), arms(root)
    logs = root / "training"
    logs.mkdir(exist_ok=True)
    with (root / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with checkpoints.manager(root / "checkpoints", keep=4) as manager:
            start = manager.latest_step()
            if start is None:
                start = 0
                snapshot(root, models, 0, manager)
            else:
                lineart_state.restore(manager, models, start)
            for name in ("metrics", "sampling"):
                p = logs / f"{name}.jsonl"
                if p.exists():
                    lines = p.read_text().splitlines()
                    kept = [line for line in lines if json.loads(line)["step"] <= start]
                    if len(kept) != len(lines):
                        shutil.copy2(p, p.with_suffix(f".before-resume-{time.time_ns()}.jsonl"))
                        p.write_text("\n".join(kept) + ("\n" if kept else ""))
            if start in REVIEWS:
                review(root, start)
            with (
                (logs / "metrics.jsonl").open("a") as log,
                (logs / "sampling.jsonl").open("a") as sample,
            ):
                tick = time.monotonic()
                for step in range(start, STEPS):
                    p, t, audit = batch(data, step)
                    metrics = previous.update(models["graph"], p, t)
                    if step < 6:
                        previous.compare(
                            metrics, read(root / "preflight.json")["rows"][step]["metrics"]
                        )
                    row = dict(
                        step=step + 1,
                        total_steps=16201 + step,
                        depth=audit["depth"],
                        learning_rate=float(models["graph"]["schedule"](step)),
                        **metrics,
                    )
                    log.write(json.dumps(row) + "\n")
                    sample.write(json.dumps(dict(step=step + 1, **audit)) + "\n")
                    if (step + 1) % 100 == 0 or step < 6:
                        log.flush()
                        sample.flush()
                        elapsed = time.monotonic() - tick
                        write(
                            root / "progress.json",
                            dict(
                                **row,
                                updated_at=now(),
                                seconds=elapsed,
                                seconds_per_step=elapsed / (step + 1 - start),
                            ),
                        )
                    if step + 1 == 6 or (step + 1) % 2000 == 0:
                        snapshot(root, models, step + 1, manager)
                    if step + 1 in REVIEWS:
                        review(root, step + 1)
    write(
        root / "training-summary.json",
        dict(
            added_steps=STEPS,
            total_steps=16200 + STEPS,
            finished_at=now(),
            finite=True,
            reviews=REVIEWS,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "train", "review"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if args.phase == "review":
        review(args.root.resolve(), args.step)
    else:
        globals()[args.phase](args.root.resolve())
