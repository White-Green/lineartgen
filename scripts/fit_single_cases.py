"""Fit four independent cached training cases; diagnostic only, never adoption."""

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
from diagnose_learnability import (
    SEED,
    arms_from_root,
    fixed_inputs,
    snapshot,
    update,
    verify,
)
from flax import nnx
from PIL import Image, ImageDraw
from train_lineart_loss_study import restore_journal

from lineartgen.config import ExperimentConfig, load_config, save_config
from lineartgen.diffusion import GuidedTrainingInputs, pyramid, sample_prepared
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.training import checkpoints, lineart_state

CASES = {"fit_straight": 38, "fit_thin": 39, "fit_corner": 41, "fit_gap": 45}
FIELDS = ("targets", "noise_levels", "noisy", "guides")


def prepare(root, parent):
    assert all(d.platform != "cpu" for d in jax.devices()), (
        "Cache the pyramid on the historical evaluation GPU backend for exact comparisons"
    )
    verify(parent)
    if root.exists():
        raise FileExistsError(root)
    (root / "configs").mkdir(parents=True)
    initial = parent / "snapshots/step-1200/warm_preclip_030/raw"
    old = load_config(initial / "config.json")
    assert old.loss.opacity_deficit_preclip and old.loss.opacity_deficit_weight == 0.3
    assert old.diffusion.scale_loss_multiplier == 0
    assert old.model.detail_channels == 16 and not old.model.progress_conditioning
    values = old.model_dump()
    values["train"].update(
        batch_size=1,
        accumulate_steps=1,
        steps=800,
        warmup_steps=50,
        learning_rate=1e-4,
        minimum_learning_rate=1e-5,
    )
    config = ExperimentConfig.model_validate(values)
    assert config.train.peak_learning_rate == 1e-4
    for name in CASES:
        save_config(config, root / f"configs/{name}.json")
    data = np.load(parent / "train.npz")
    arrays = {}
    for name, pair in CASES.items():
        clean, source = (data[k][pair : pair + 1] for k in ("clean", "source"))
        guides = pyramid(jnp.asarray(source), 6)
        levels = pyramid(jnp.full_like(jnp.asarray(source), 0.5), 6)
        rng = np.random.default_rng(SEED + 10000 + pair)
        images = tuple(
            g * jnp.sqrt(1 - t)
            + jnp.asarray(rng.standard_normal(g.shape, dtype=np.float32)) * jnp.sqrt(t)
            for g, t in zip(guides, levels, strict=True)
        )
        targets = pyramid(jnp.asarray(clean), 6)
        reference = fixed_inputs(clean, source, 0.5, SEED + 10000 + pair)
        for actual, expected in zip(
            (images[:3], levels[:3], guides[:3]), reference[:3], strict=True
        ):
            jax.tree.map(np.testing.assert_array_equal, actual, expected)
        np.testing.assert_array_equal(targets[0], reference[3])
        for field, values in zip(FIELDS, (targets, levels, images, guides), strict=True):
            for depth, value in enumerate(values):
                arrays[f"{name}-{field}-{depth}"] = np.asarray(value)
    cache = root / "cached-inputs.npz"
    np.savez_compressed(cache, **arrays)
    validation = parent.parent / "preclip-opacity-validation-20261009"
    tested = read(validation / "tested-files.json")
    assert read(validation / "result.json")["full_suite_passed"]
    assert all(sha(Path(n)) == digest for n, digest in tested.items())
    files = [parent / n for n in ("source.json", "plan.json", "train.npz", "train.json")]
    files += [parent / "evaluation/step-1200/warm_preclip_030-raw-train-n1-whole.npz"]
    for split in ("train", "probe"):
        shutil.copyfile(parent / f"{split}.json", root / f"{split}.json")
    plan = dict(
        parent=str(parent),
        parent_files={str(p.relative_to(parent)): sha(p) for p in files},
        initial=str(initial),
        initial_files={
            n: sha(initial / n) for n in ("weights.npz", "config.json", "metadata.json")
        },
        data={
            s: dict(path=str(parent / f"{s}.npz"), sha256=sha(parent / f"{s}.npz"))
            for s in ("train", "probe")
        },
        arms=list(CASES),
        warm_start_methods=list(CASES),
        cases=CASES,
        cached_inputs_sha256=sha(cache),
        preparation_backend=jax.devices()[0].platform,
        plan_document_sha256=sha(Path("docs/single-case-fitting-plan-20261009.md")),
        core_validation=dict(
            tests=298,
            files=len(tested),
            hashes=tested,
            result_sha256=sha(validation / "result.json"),
        ),
        steps=800,
        primary_depth=3,
        primary="raw one-step on each fitted case",
        screen=dict(
            ink_f1_alpha50_min=0.95, ink_deficit_fraction_max=0.05, halo_opacity_excess_max=0.02
        ),
        diagnostic_only=True,
        adoption=False,
    )
    write(root / "plan.json", plan)
    write(
        root / "input-audit.json",
        dict(
            first_three_levels_exact=True,
            arrays=len(arrays),
            cases=CASES,
            tested_core_files_unchanged=len(tested),
            cache_sha256=sha(cache),
        ),
    )
    print(dict(prepared=str(root), cases=CASES, cached_prefixes_exact=True), flush=True)


def verify_study(root):
    verify(root)
    plan = read(root / "plan.json")
    parent = Path(plan["parent"])
    verify(parent)
    assert sha(root / "cached-inputs.npz") == plan["cached_inputs_sha256"]
    assert all(sha(parent / n) == digest for n, digest in plan["parent_files"].items())
    assert all(sha(Path(n)) == digest for n, digest in plan["core_validation"]["hashes"].items())
    return plan


def cached_inputs(root):
    data = np.load(root / "cached-inputs.npz")
    return {
        name: GuidedTrainingInputs(
            *(tuple(jnp.asarray(data[f"{name}-{field}-{d}"]) for d in range(6)) for field in FIELDS)
        )
        for name in CASES
    }


def prefix(prepared, depth):
    return GuidedTrainingInputs(*(values[:depth] for values in prepared))


forward = nnx.jit(
    lambda model, prepared: sample_prepared(
        model, prepared.noisy, prepared.noise_levels, steps=1, guides=prepared.guides
    )
)


def evaluate(root, arms, inputs, step):
    folder = root / f"evaluation/step-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    for name, arm in arms.items():
        for depth in range(1, 7):
            prepared = prefix(inputs[name], depth)
            result = np.asarray(forward(arm["model"], prepared))[0]
            target, level = (
                np.asarray(prepared.targets[0])[0],
                np.asarray(prepared.noise_levels[0])[0],
            )
            assert result.shape == target.shape == level.shape == (512, 512, 1)
            metrics = halo_metrics(result, target, level)
            metrics["active_mse"] = float(np.mean(np.square(result - target)))
            passed = (
                metrics["ink_f1_alpha50"] >= 0.95
                and metrics["ink_deficit_fraction"] <= 0.05
                and metrics["halo_opacity_excess"] <= 0.02
            )
            rows.append(
                dict(method=name, pair=CASES[name], depth=depth, fit_screen_pass=passed, **metrics)
            )
            arrays[f"{name}-{depth}"] = result
    np.savez_compressed(folder / "predictions.npz", **arrays)
    write(
        folder / "metrics.json",
        dict(
            step=step,
            rows=rows,
            predictions_sha256=sha(folder / "predictions.npz"),
            primary_passes=sum(x["fit_screen_pass"] for x in rows if x["depth"] == 3),
            all_depth_passes=sum(x["fit_screen_pass"] for x in rows),
            plan_sha256=sha(root / "plan.json"),
            source_sha256=sha(root / "source.json"),
            adoption=False,
        ),
    )
    if step == 0:
        plan = read(root / "plan.json")
        old = np.load(
            Path(plan["parent"]) / "evaluation/step-1200/warm_preclip_030-raw-train-n1-whole.npz"
        )["images"]
        for name, pair in CASES.items():
            np.testing.assert_array_equal(arrays[f"{name}-3"], old[57 + pair])
        write(root / "initial-forward-audit.json", dict(all_exact=True, cases=CASES))
    print(dict(evaluated_step=step, conditions=len(rows)), flush=True)


def train(root, *, preflight=False, end=800):
    plan = verify_study(root)
    assert read(Path(plan["parent"]) / "workflow-status.json")["stage"] == "complete_rejected"
    assert all(d.platform != "cpu" for d in jax.devices())
    if not preflight:
        assert read(root / "preflight/restore-audit.json")["all_state_exact"]
        assert read(root / "initial-forward-audit.json")["all_exact"]
    assert end in (1, 200, 800)
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
        for step in range(start, 6 if preflight else end):
            begin = time.perf_counter()
            depth = step % 6 + 1
            stats = {
                name: update(arms[name], prefix(inputs[name], depth))
                for name in names[step % 4 :] + names[: step % 4]
            }
            complete = step + 1
            row = dict(step=complete, depth=depth, arms=stats, seconds=time.perf_counter() - begin)
            if preflight or complete % 25 == 0 or complete == end:
                with (destination / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                write(destination / "progress.json", row)
                print(json.dumps(row), flush=True)
            if complete % 100 == 0 or preflight and complete == 6 or complete == end:
                lineart_state.save(saved, arms, complete)
            if not preflight and complete in (200, 800):
                snapshot(root, arms, complete)
                evaluate(root, arms, inputs, complete)
        restored = arms_from_root(root)
        last = lineart_state.restore(saved, restored)
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload(arms, last),
            lineart_state.payload(restored, last),
        )
        write(
            destination / "restore-audit.json",
            dict(
                step=last,
                all_state_exact=True,
                parameter_counts={
                    n: sum(x.size for x in jax.tree.leaves(nnx.state(a["model"], nnx.Param)))
                    for n, a in arms.items()
                },
            ),
        )
        if not preflight:
            shutil.copyfile(
                destination / "restore-audit.json", destination / f"restore-audit-step-{last}.json"
            )


def render(root):
    verify_study(root)
    out = root / "visual-review"
    out.mkdir(exist_ok=True)
    data = np.load(root / "cached-inputs.npz")
    predictions = {
        step: np.load(root / f"evaluation/step-{step}/predictions.npz") for step in (0, 200, 800)
    }
    files = {}
    for filename, columns in (
        ("progress-depth3.png", [(s, 3) for s in (0, 200, 800)]),
        ("final-all-depths.png", [(800, d) for d in range(1, 7)]),
    ):
        canvas = Image.new("RGB", ((len(columns) + 2) * 160, len(CASES) * 180), "white")
        draw = ImageDraw.Draw(canvas)
        crops = []
        for row, name in enumerate(CASES):
            target = data[f"{name}-targets-0"][0]
            ys, xs = np.where(target[..., 0] < 0.96)
            y, x = (int(np.clip(np.mean(a) - 80, 0, 352)) for a in (ys, xs))
            imgs = [target, data[f"{name}-guides-0"][0]] + [
                predictions[s][f"{name}-{d}"] for s, d in columns
            ]
            labels = ["target", "source"] + [f"step{s} depth{d}" for s, d in columns]
            for col, (label, value) in enumerate(zip(labels, imgs, strict=True)):
                gray = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(
                    Image.fromarray(gray[y : y + 160, x : x + 160]).convert("RGB"),
                    (col * 160, row * 180 + 20),
                )
                draw.text((col * 160 + 2, row * 180 + 2), f"{name[4:]} {label}", fill="black")
            crops.append(dict(method=name, pair=CASES[name], crop=[x, y, x + 160, y + 160]))
        canvas.save(out / filename)
        files[filename] = dict(
            sha256=sha(out / filename), columns=columns, crops=crops, inspected=False
        )
    write(
        out / "manifest.json",
        dict(
            files=files,
            inspection_complete=False,
            adoption=False,
            script_sha256=sha(Path(__file__)),
            plan_sha256=sha(root / "plan.json"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "preflight", "train", "render"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--end", type=int, default=800)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "prepare":
        prepare(root, a.parent.resolve())
    elif a.action == "render":
        render(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, preflight=a.action == "preflight", end=a.end)
