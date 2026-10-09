"""Matched single-case fitting with fresh noise; diagnostic only."""

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import arms_from_root, fixed_inputs, snapshot, update, verify
from fit_single_cases import CASES, cached_inputs, evaluate, forward, prefix, render, verify_study
from PIL import Image, ImageDraw
from train_lineart_loss_study import restore_journal

from lineartgen.diffusion import GuidedTrainingInputs
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference
from lineartgen.training import checkpoints, lineart_state

NOISE_SEED = 2026100923
DRAWS = {"old_transfer": 2026100917} | {f"replicate_{i}": 2026100931 + 100 * i for i in range(8)}


def fresh_inputs(prepared, step, pair):
    rng = np.random.default_rng([NOISE_SEED, step, pair])
    images = tuple(
        g * jnp.sqrt(1 - t)
        + jnp.asarray(rng.standard_normal(g.shape, dtype=np.float32)) * jnp.sqrt(t)
        for g, t in zip(prepared.guides, prepared.noise_levels, strict=True)
    )
    return GuidedTrainingInputs(prepared.targets, prepared.noise_levels, images, prepared.guides)


def prepare(root, previous):
    assert all(d.platform != "cpu" for d in jax.devices())
    old = verify_study(previous)
    assert read(previous / "workflow-status.json")["stage"] == "complete_diagnostic_insufficient"
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    for n in ("train.json", "probe.json", "cached-inputs.npz"):
        shutil.copyfile(previous / n, root / n)
    for name in CASES:
        shutil.copyfile(previous / f"configs/{name}.json", root / f"configs/{name}.json")
    plan = dict(old)
    old_transfer = previous.parent / "single-case-transfer-20261009"
    assert read(old_transfer / "workflow-status.json")["stage"] == "complete_reviewed"
    plan.update(
        previous=str(previous),
        previous_transfer=str(old_transfer),
        previous_transfer_files={
            n: sha(old_transfer / n)
            for n in ("plan.json", "metrics.json", "predictions.npz", "review.json")
        },
        previous_files={
            n: sha(previous / n)
            for n in (
                "source.json",
                "plan.json",
                "cached-inputs.npz",
                "preflight/restore-audit.json",
                "evaluation/step-0/predictions.npz",
                "evaluation/step-800/predictions.npz",
                "evaluation/step-800/metrics.json",
                "visual-review/review.json",
            )
        },
        previous_checkpoint_files={
            str(p.relative_to(previous)): sha(p)
            for p in (previous / "preflight/checkpoints/6").rglob("*")
            if p.is_file()
        },
        previous_model_files={
            str(p.relative_to(previous)): sha(p)
            for name in CASES
            for p in (previous / f"snapshots/step-800/{name}/raw").iterdir()
            if p.is_file()
        },
        plan_document_sha256=sha(Path("docs/fresh-noise-fitting-plan-20261009.md")),
        noise_training="fresh_per_update",
        noise_seed=NOISE_SEED,
        evaluation_draws=DRAWS,
        primary="raw depth3 own-case on all eight independent diagnostic noise draws",
    )
    write(root / "plan.json", plan)
    inputs = cached_inputs(root)
    audit = []
    for name, pair in CASES.items():
        original = inputs[name]
        a, b, c = (fresh_inputs(original, step, pair) for step in (0, 0, 1))
        jax.tree.map(np.testing.assert_array_equal, a, b)
        rng = np.random.default_rng([NOISE_SEED, 0, pair])
        for d in range(6):
            g, t = original.guides[d], original.noise_levels[d]
            e = rng.standard_normal(g.shape, dtype=np.float32)
            expected = g * jnp.sqrt(1 - t) + jnp.asarray(e) * jnp.sqrt(t)
            np.testing.assert_array_equal(a.noisy[d], expected)
            assert not np.array_equal(a.noisy[d], c.noisy[d])
            assert not np.array_equal(a.noisy[d], original.noisy[d])
            for field in ("targets", "noise_levels", "guides"):
                np.testing.assert_array_equal(getattr(a, field)[d], getattr(original, field)[d])
            if d == 0:
                assert abs(float(e.mean())) < 0.01 and abs(float(e.std()) - 1) < 0.01
            audit.append(
                dict(
                    case=name,
                    depth=d + 1,
                    shape=list(e.shape),
                    mean=float(e.mean()),
                    std=float(e.std()),
                )
            )
    write(
        root / "input-audit.json",
        dict(
            reproducible=True,
            all_levels_resampled=True,
            conditioning_unchanged=True,
            gaussian_construction_exact=True,
            rows=audit,
        ),
    )
    print(dict(prepared=str(root), noise_seed=NOISE_SEED), flush=True)


def verify_fresh(root):
    plan = verify_study(root)
    previous = Path(plan["previous"])
    verify(previous)
    for field in ("previous_files", "previous_checkpoint_files", "previous_model_files"):
        assert all(sha(previous / n) == digest for n, digest in plan[field].items())
    assert all(
        sha(Path(plan["previous_transfer"]) / n) == digest
        for n, digest in plan["previous_transfer_files"].items()
    )
    if "resume_from" in plan:
        resume = Path(plan["resume_from"])
        verify(resume)
        assert all(sha(resume / n) == digest for n, digest in plan["resume_files"].items())
    assert plan["noise_seed"] == NOISE_SEED and plan["evaluation_draws"] == DRAWS
    return plan


def replay(root):
    plan = verify_fresh(root)
    arms, inputs = arms_from_root(root), cached_inputs(root)
    for step in range(6):
        for name in arms:
            update(arms[name], prefix(inputs[name], step % 6 + 1))
    expected = arms_from_root(root)
    with checkpoints.manager(Path(plan["previous"]) / "preflight/checkpoints", keep=3) as saved:
        last = lineart_state.restore(saved, expected, 6)
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(arms, 6),
        lineart_state.payload(expected, last),
    )
    write(root / "cached-replay-audit.json", dict(steps=6, all_state_exact=True, cases=CASES))
    print("cached six-update replay: all state exact", flush=True)


def train(root, *, preflight=False, end=800):
    plan = verify_fresh(root)
    assert all(d.platform != "cpu" for d in jax.devices())
    assert read(root / "cached-replay-audit.json")["all_state_exact"]
    if not preflight:
        assert read(root / "preflight/restore-audit.json")["all_state_exact"]
        assert read(root / "initial-forward-audit.json")["all_exact"]
    assert end in (1, 200, 800, 801, 806, 1600, 4000)
    preflight_end = plan.get("resume_step", 0) + 6
    arms, inputs = arms_from_root(root), cached_inputs(root)
    destination = root / ("preflight" if preflight else "training")
    destination.mkdir(exist_ok=True)
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(destination, start)
        if preflight and start == 0:
            evaluate(root, arms, inputs, 0)
        if not preflight and start == 0:
            snapshot(root, arms, 0)
        names = tuple(arms)
        for step in range(start, preflight_end if preflight else end):
            begin = time.perf_counter()
            depth = step % 6 + 1
            stats = {
                name: update(
                    arms[name], prefix(fresh_inputs(inputs[name], step, CASES[name]), depth)
                )
                for name in names[step % 4 :] + names[: step % 4]
            }
            complete = step + 1
            row = dict(step=complete, depth=depth, arms=stats, seconds=time.perf_counter() - begin)
            if preflight or complete % 25 == 0 or complete == end:
                with (destination / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                write(destination / "progress.json", row)
                print(json.dumps(row), flush=True)
            if complete % 100 == 0 or preflight and complete == preflight_end or complete == end:
                lineart_state.save(saved, arms, complete)
            if not preflight and complete in (200, 800, 1600, 4000):
                snapshot(root, arms, complete)
                evaluate(root, arms, inputs, complete)
        restored = arms_from_root(root)
        last = lineart_state.restore(saved, restored)
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload(arms, last),
            lineart_state.payload(restored, last),
        )
        audit = dict(step=last, all_state_exact=True)
        write(destination / "restore-audit.json", audit)
        if not preflight:
            write(destination / f"restore-audit-step-{last}.json", audit)


def transfer(root, step):
    plan = verify_fresh(root)
    assert step in (0, 200, 800, 1600, 4000)
    out = root / f"transfer/step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "metrics.json").exists()
    cached = cached_inputs(root)
    all_inputs = {}
    for name, pair in CASES.items():
        p = cached[name]
        for draw, seed in DRAWS.items():
            x, t, g, target, _ = fixed_inputs(
                np.asarray(p.targets[0]), np.asarray(p.guides[0]), 0.5, seed + pair
            )
            np.testing.assert_array_equal(target, p.targets[0])
            all_inputs[name, draw] = GuidedTrainingInputs(p.targets[:3], t, x, g)
    arrays, rows, bindings, checks = {}, [], {}, []
    old_transfer = np.load(Path(plan["previous_transfer"]) / "predictions.npz")
    for mode in ("initial", "fixed", "fresh"):
        for name, pair in CASES.items():
            folder = (
                Path(plan["initial"])
                if mode == "initial" or mode == "fresh" and step == 0
                else Path(plan["previous"]) / f"snapshots/step-800/{name}/raw"
                if mode == "fixed"
                else root / f"snapshots/step-{step}/{name}/raw"
            )
            bindings.update(
                {
                    str(folder / f): sha(folder / f)
                    for f in ("weights.npz", "config.json", "metadata.json")
                }
            )
            model, _ = load_inference(folder)
            for draw in DRAWS:
                p = all_inputs[name, draw]
                image = np.asarray(forward(model, p))[0]
                target, level = np.asarray(p.targets[0])[0], np.asarray(p.noise_levels[0])[0]
                assert image.shape == target.shape == (512, 512, 1) and np.isfinite(image).all()
                if draw == "old_transfer" and mode in ("initial", "fixed"):
                    old_name = "initial" if mode == "initial" else name
                    np.testing.assert_array_equal(
                        image, old_transfer[f"{old_name}-new_noise-{name}"]
                    )
                    checks.append(dict(model=mode, case=name, exact=True))
                if step == 0 and mode == "fresh":
                    np.testing.assert_array_equal(image, arrays[f"initial-{name}-{draw}"])
                m = halo_metrics(image, target, level)
                passed = (
                    m["ink_f1_alpha50"] >= 0.95
                    and m["ink_deficit_fraction"] <= 0.05
                    and m["halo_opacity_excess"] <= 0.02
                )
                rows.append(
                    dict(model=mode, case=name, pair=pair, draw=draw, fit_screen_pass=passed, **m)
                )
                arrays[f"{mode}-{name}-{draw}"] = image
    np.savez_compressed(out / "predictions.npz", **arrays)
    write(
        out / "metrics.json",
        dict(
            step=step,
            rows=rows,
            previous_transfer_exact=checks,
            bindings=bindings,
            predictions_sha256=sha(out / "predictions.npz"),
            plan_sha256=sha(root / "plan.json"),
            source_sha256=sha(root / "source.json"),
            adoption=False,
        ),
    )
    print(dict(transfer_step=step, conditions=len(rows)), flush=True)


def render_transfer(root, step=800):
    verify_fresh(root)
    out = root / "transfer/visual-review"
    out.mkdir(exist_ok=True)
    scores = read(root / f"transfer/step-{step}/metrics.json")
    predictions = np.load(root / f"transfer/step-{step}/predictions.npz")
    cached = np.load(root / "cached-inputs.npz")
    parent_predictions = None
    plan = read(root / "plan.json")
    if "resume_from" in plan:
        parent_predictions = np.load(
            Path(plan["resume_from"]) / "transfer/step-800/predictions.npz"
        )
    files = {}
    for selector in ("old_transfer", "replicate_0", "worst_f1"):
        canvas = Image.new("RGB", (5 * 160, 4 * 180), "white")
        draw = ImageDraw.Draw(canvas)
        crops = []
        for row, name in enumerate(CASES):
            noise = (
                selector
                if selector != "worst_f1"
                else min(
                    (
                        r
                        for r in scores["rows"]
                        if r["model"] == "fresh"
                        and r["case"] == name
                        and r["draw"].startswith("replicate")
                    ),
                    key=lambda r: r["ink_f1_alpha50"],
                )["draw"]
            )
            target, source = (cached[f"{name}-{field}-0"][0] for field in ("targets", "guides"))
            ys, xs = np.where(target[..., 0] < 0.96)
            y, x = (int(np.clip(np.mean(a) - 80, 0, 352)) for a in (ys, xs))
            imgs = [target, source] + [
                predictions[f"{mode}-{name}-{noise}"] for mode in ("initial", "fixed", "fresh")
            ]
            if parent_predictions is not None:
                imgs[3] = parent_predictions[f"fresh-{name}-{noise}"]
            for col, (label, value) in enumerate(
                zip(
                    (
                        "target",
                        "source",
                        "initial",
                        "fresh800" if parent_predictions is not None else "fixed800",
                        f"fresh{step}",
                    ),
                    imgs,
                    strict=True,
                )
            ):
                gray = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(
                    Image.fromarray(gray[y : y + 160, x : x + 160]).convert("RGB"),
                    (col * 160, row * 180 + 20),
                )
                draw.text((col * 160 + 2, row * 180 + 2), f"{name[4:]} {label}", fill="black")
            crops.append(dict(case=name, noise=noise, crop=[x, y, x + 160, y + 160]))
        filename = f"{selector}.png"
        canvas.save(out / filename)
        files[filename] = dict(sha256=sha(out / filename), crops=crops, inspected=False)
    write(out / "manifest.json", dict(files=files, inspection_complete=False, adoption=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("prepare", "replay", "preflight", "train", "transfer", "render")
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--end", type=int, default=800)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "prepare":
        prepare(root, a.previous.resolve())
    elif a.action == "transfer":
        transfer(root, a.end)
    elif a.action == "render":
        render(root)
        render_transfer(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.action == "replay":
                replay(root)
            else:
                train(root, preflight=a.action == "preflight", end=a.end)
