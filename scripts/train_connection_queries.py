"""Paired shared-model fitting of four synthetic cases with local graph queries."""

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
from diagnose_learnability import arms_from_root, fixed_inputs, freeze, snapshot, update, verify
from fit_fresh_noise import DRAWS, fresh_inputs
from fit_single_cases import CASES, forward, prefix
from flax import nnx
from PIL import Image, ImageDraw
from scipy import ndimage
from train_lineart_loss_study import restore_journal

from lineartgen.config import GeometryDataConfig, load_config, save_config
from lineartgen.data.geometry import draw_geometry, normals
from lineartgen.diffusion import GuidedTrainingInputs, pyramid, recursive_forward_with_preclip
from lineartgen.evaluation import probe_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.connectivity import connection_capacity, connectivity_loss
from lineartgen.training.losses import image_loss

METHODS = {"pixel": 0.0, "connection": 0.1}
PATCH = 64
STEPS = 2000


def point_at(points, distance):
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.array([np.interp(distance, arc, points[:, axis]) for axis in (0, 1)])


def query_spec(case, a, b, connected):
    origin = np.clip(np.rint((a + b) / 2).astype(int) - PATCH // 2, 0, 512 - PATCH)
    masks = []
    for point in (a, b):
        x, y = np.rint(point - origin).astype(int)
        assert 1 <= x < PATCH - 1 and 1 <= y < PATCH - 1
        mask = np.zeros((PATCH, PATCH), bool)
        mask[y - 1 : y + 2, x - 1 : x + 2] = True
        masks.append(mask)
    return dict(
        case=case, origin=origin.tolist(), a=a.tolist(), b=b.tolist(), connected=connected
    ), masks


def prepare(root, parent):
    assert not root.exists() and jax.devices()[0].platform != "cpu"
    (root / "configs").mkdir(parents=True)
    old = read(parent / "plan.json")
    data = {split: np.load(old["data"][split]["path"]) for split in ("train", "probe")}
    base = load_config(Path(old["initial"]) / "config.json")
    config = base.model_copy(
        update={
            "train": base.train.model_copy(
                update={
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "steps": STEPS,
                    "learning_rate": 1e-4,
                    "minimum_learning_rate": 1e-5,
                    "warmup_steps": 50,
                }
            )
        }
    )
    assert config.train.peak_learning_rate == 2e-4
    assert config.diffusion.scale_loss_multiplier == 0
    for name in METHODS:
        save_config(config, root / f"configs/{name}.json")
    arrays, query_meta = {}, {}
    for split in ("train", "probe"):
        shutil.copyfile(parent / f"{split}.json", root / f"{split}.json")
        records = read(root / f"{split}.json")
        specs, aa, bb = [], [], []
        for i, (name, pair) in enumerate(CASES.items()):
            clean, source = (data[split][key][pair : pair + 1] for key in ("clean", "source"))
            for field, values in (
                ("targets", pyramid(jnp.asarray(clean), 6)),
                ("guides", pyramid(jnp.asarray(source), 6)),
                ("noise_levels", pyramid(jnp.full_like(jnp.asarray(clean), 0.5), 6)),
            ):
                for depth, value in enumerate(values):
                    arrays[f"{split}-{name}-{field}-{depth}"] = np.asarray(value)
            record = records[pair]
            gray, metadata = draw_geometry(
                np.random.default_rng(record["generation_key"]),
                512,
                GeometryDataConfig(minimum_width=0.6, maximum_width=6, supersampling=4),
                kind=record["kind"],
                include_paths=True,
            )
            np.testing.assert_array_equal(np.clip((gray * 2 - 1) * 1.1, -1, 1), clean[0])
            paths = [np.asarray(p["points"]) for p in metadata["path_geometry"]]
            probes, connections = [], []
            for path_id, points in enumerate(paths):
                ns = normals(points)
                for j in np.unique(
                    np.linspace(len(points) * 0.15, len(points) * 0.85, 48).astype(int)
                ):
                    if name == "fit_corner" and abs(j - len(points) // 2) < 12:
                        continue
                    if np.any(points[j] < 13) or np.any(points[j] > 498):
                        continue
                    probes.append([*points[j], *ns[j], 12, path_id])
                connections.append([*points[len(points) // 5], *points[4 * len(points) // 5], 1])
                length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                for fraction in (0.5,) if len(paths) > 1 else (0.3, 0.7):
                    a, b = (point_at(points, length * fraction + d) for d in (-10, 10))
                    spec, masks = query_spec(i, a, b, True)
                    specs.append(spec)
                    aa.append(masks[0])
                    bb.append(masks[1])
            if name == "fit_gap":
                connections.append(
                    [*paths[0][len(paths[0]) // 2], *paths[1][len(paths[1]) // 2], 0]
                )
                closest = min(
                    ((a, b) for a in paths[0][[0, -1]] for b in paths[1][[0, -1]]),
                    key=lambda ab: np.linalg.norm(ab[0] - ab[1]),
                )
                anchors = []
                for points, end in zip(paths, closest, strict=True):
                    ordered = points if np.all(end == points[0]) else points[::-1]
                    anchors.append(point_at(ordered, 8))
                spec, masks = query_spec(i, *anchors, False)
                specs.append(spec)
                aa.append(masks[0])
                bb.append(masks[1])
                arrays[f"{split}-gap-endpoints"] = np.stack(closest)
            arrays[f"{split}-{name}-probes"] = np.asarray(probes, np.float32)
            arrays[f"{split}-{name}-connections"] = np.asarray(connections, np.float32)
        query_meta[split] = specs
        arrays[f"{split}-anchor-a"] = np.stack(aa)
        arrays[f"{split}-anchor-b"] = np.stack(bb)
    np.savez_compressed(root / "inputs.npz", **arrays)
    write(root / "queries.json", query_meta)
    write(
        root / "plan.json",
        dict(
            arms=list(METHODS),
            warm_start_methods=list(METHODS),
            coefficients=METHODS,
            initial=old["initial"],
            initial_files=old["initial_files"],
            data=old["data"],
            parent=str(parent),
            cases=CASES,
            steps=STEPS,
            milestones=[0, 600, STEPS],
            draws=DRAWS,
            inputs_sha256=sha(root / "inputs.npz"),
            queries_sha256=sha(root / "queries.json"),
            plan_document_sha256=sha(Path("docs/connection-query-plan-20261009.md")),
            diagnostic_only=True,
            adoption=False,
        ),
    )
    print("prepared shared four-case paired study", flush=True)


def inputs(root, split):
    cache = np.load(root / "inputs.npz")
    return {
        name: GuidedTrainingInputs(
            *(
                tuple(jnp.asarray(cache[f"{split}-{name}-{field}-{d}"]) for d in range(6))
                for field in ("targets", "noise_levels", "guides", "guides")
            )
        )
        for name in CASES
    }


def query_arrays(root, split):
    specs = read(root / "queries.json")[split]
    cache = np.load(root / "inputs.npz")
    yy, xx = np.indices((PATCH, PATCH))
    indices = tuple(
        jnp.asarray(x)
        for x in (
            np.array([q["case"] for q in specs])[:, None, None],
            np.array([q["origin"][1] for q in specs])[:, None, None] + yy,
            np.array([q["origin"][0] for q in specs])[:, None, None] + xx,
        )
    )
    return indices, jnp.asarray(cache[f"{split}-anchor-a"]), jnp.asarray(cache[f"{split}-anchor-b"])


def make_arms(root):
    arms = arms_from_root(root)
    indices, a, b = query_arrays(root, "train")
    truth = jnp.concatenate([p.targets[0] for p in inputs(root, "train").values()])
    truth_alpha = jnp.clip((1 - truth[..., 0]) / 2, 0, 1)[indices]
    expected = jax.jit(jax.vmap(connection_capacity))(truth_alpha, a, b)
    assert np.all(np.asarray(expected[:-1]) > 0) and expected[-1] == 0
    for name, arm in arms.items():
        if METHODS[name] == 0:
            continue
        config = arm["config"]

        def objective(model, prepared, config=config):
            predictions, physical = recursive_forward_with_preclip(
                model, prepared.noisy, prepared.noise_levels, guides=prepared.guides
            )
            base, metrics = image_loss(
                predictions[0],
                prepared.targets[0],
                prepared.noise_levels[0],
                config.loss,
                config.diffusion.min_noise,
                preclip=physical[0],
            )
            alpha = jnp.clip((1 - predictions[0][..., 0]) / 2, 0, 1)[indices]
            positive, _ = connectivity_loss(alpha[:-1], truth_alpha[:-1], a[:-1], b[:-1])
            negative, _ = connectivity_loss(alpha[-1:], truth_alpha[-1:], a[-1:], b[-1:])
            extra = (positive + negative) / 2
            total = base + METHODS["connection"] * extra
            metrics = dict(
                metrics,
                loss=total,
                base_loss=base,
                connection_loss=extra,
                connected_deficit=positive,
                false_bridge=negative,
            )
            return total, metrics

        @nnx.jit
        def gradients(model, prepared):
            (_, metrics), gradient = nnx.value_and_grad(objective, has_aux=True)(model, prepared)
            return metrics, gradient

        arm["gradient"] = gradients
    return arms


def batch(prepared, step):
    return jax.tree.map(
        lambda *x: jnp.concatenate(x, axis=0),
        *(
            prefix(fresh_inputs(prepared[name], step, pair), step % 6 + 1)
            for name, pair in CASES.items()
        ),
    )


def evaluate(root, arms, step):
    out = root / f"evaluation/step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    cache = np.load(root / "inputs.npz")
    for split in ("train", "probe"):
        prepared = inputs(root, split)
        for draw, seed in DRAWS.items():
            if draw == "old_transfer":
                continue
            xs = []
            for name, pair in CASES.items():
                p = prepared[name]
                x, t, g, _, _ = fixed_inputs(
                    np.asarray(p.targets[0]),
                    np.asarray(p.guides[0]),
                    0.5,
                    seed + pair + (1000000 if split == "probe" else 0),
                )
                xs.append(GuidedTrainingInputs(p.targets[:3], t, x, g))
            whole = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *xs)
            for method, arm in arms.items():
                results = np.asarray(forward(arm["model"], whole))
                for i, name in enumerate(CASES):
                    image, target = results[i], np.asarray(prepared[name].targets[0])[0]
                    assert image.shape == (512, 512, 1) and np.isfinite(image).all()
                    metrics = halo_metrics(image, target, np.full_like(target, 0.5))
                    metrics.update(
                        probe_metrics(
                            image,
                            target,
                            cache[f"{split}-{name}-probes"],
                            cache[f"{split}-{name}-connections"],
                        )
                    )
                    key = f"{method}-{split}-{name}-{draw}"
                    arrays[key] = image
                    gap_mean = None
                    if name == "fit_gap":
                        a, b = cache[f"{split}-gap-endpoints"]
                        points = np.linspace(a, b, 61)[15:46]
                        gap_mean = float(
                            ndimage.map_coordinates(
                                (1 - image[..., 0]) / 2, [points[:, 1], points[:, 0]], order=1
                            ).mean()
                        )
                    passed = (
                        metrics["ink_f1_alpha50"] >= 0.95
                        and metrics["ink_deficit_fraction"] <= 0.05
                        and metrics["halo_opacity_excess"] <= 0.02
                    )
                    rows.append(
                        dict(
                            method=method,
                            split=split,
                            case=name,
                            draw=draw,
                            fit_screen_pass=passed,
                            gap_middle_alpha=gap_mean,
                            **metrics,
                        )
                    )
    np.savez_compressed(out / "predictions.npz", **arrays)
    if step == 0:
        for split in ("train", "probe"):
            for case in CASES:
                for draw in DRAWS:
                    if draw != "old_transfer":
                        np.testing.assert_array_equal(
                            arrays[f"pixel-{split}-{case}-{draw}"],
                            arrays[f"connection-{split}-{case}-{draw}"],
                        )
        write(root / "initial-forward-audit.json", dict(all_exact=True, conditions=64))
    write(
        out / "metrics.json",
        dict(step=step, rows=rows, predictions_sha256=sha(out / "predictions.npz"), adoption=False),
    )
    print(dict(evaluated_step=step, conditions=len(rows)), flush=True)


def train(root, preflight):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(root / "inputs.npz") == plan["inputs_sha256"]
    assert sha(root / "queries.json") == plan["queries_sha256"]
    assert jax.devices()[0].platform != "cpu"
    destination = root / ("preflight" if preflight else "training")
    destination.mkdir(exist_ok=True)
    if not preflight:
        assert read(root / "preflight/restore-audit.json")["all_state_exact"]
    arms, prepared = make_arms(root), inputs(root, "train")
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(destination, start)
        if start == 0 and not preflight:
            snapshot(root, arms, 0)
            evaluate(root, arms, 0)
        for step in range(start, 6 if preflight else STEPS):
            begin = time.perf_counter()
            p = batch(prepared, step)
            stats = {
                name: update(arms[name], p)
                for name in (tuple(arms) if step % 2 else tuple(arms)[::-1])
            }
            complete = step + 1
            row = dict(
                step=complete, depth=step % 6 + 1, arms=stats, seconds=time.perf_counter() - begin
            )
            if preflight or complete % 25 == 0:
                with (destination / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                write(destination / "progress.json", row)
                print(json.dumps(row), flush=True)
            if complete % 100 == 0 or complete == 6:
                lineart_state.save(saved, arms, complete)
            if not preflight and complete == 6:
                check = make_arms(root)
                with checkpoints.manager(root / "preflight/checkpoints", keep=3) as prior:
                    lineart_state.restore(prior, check, 6)
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(check, 6),
                    lineart_state.payload(arms, 6),
                )
                write(root / "replay-audit.json", dict(all_state_exact=True, step=6))
            if not preflight and complete in (600, STEPS):
                snapshot(root, arms, complete)
                evaluate(root, arms, complete)
        restored = make_arms(root)
        last = lineart_state.restore(saved, restored)
        jax.tree.map(
            np.testing.assert_array_equal,
            lineart_state.payload(arms, last),
            lineart_state.payload(restored, last),
        )
        write(destination / "restore-audit.json", dict(all_state_exact=True, step=last))
    if not preflight:
        render(root)
        write(
            root / "workflow-status.json", dict(stage="trained_pending_visual_review", step=STEPS)
        )


def render(root):
    out = root / "visual-review"
    out.mkdir(exist_ok=True)
    cache = np.load(root / "inputs.npz")
    images = np.load(root / f"evaluation/step-{STEPS}/predictions.npz")
    rows = read(root / f"evaluation/step-{STEPS}/metrics.json")["rows"]
    files = {}
    for split in ("train", "probe"):
        for selector in ("replicate_0", "worst_connection_f1"):
            canvas = Image.new("RGB", (4 * 160, 4 * 180), "white")
            painter = ImageDraw.Draw(canvas)
            crops = []
            for row, name in enumerate(CASES):
                draw = (
                    selector
                    if selector == "replicate_0"
                    else min(
                        (
                            r
                            for r in rows
                            if r["method"] == "connection"
                            and r["split"] == split
                            and r["case"] == name
                        ),
                        key=lambda r: r["ink_f1_alpha50"],
                    )["draw"]
                )
                target, source = (
                    cache[f"{split}-{name}-{field}-0"][0] for field in ("targets", "guides")
                )
                yy, xx = np.where(target[..., 0] < 0.96)
                y, x = (int(np.clip(np.mean(a) - 80, 0, 352)) for a in (yy, xx))
                values = [target, source] + [images[f"{m}-{split}-{name}-{draw}"] for m in METHODS]
                for col, (label, value) in enumerate(
                    zip(("target", "source", *METHODS), values, strict=True)
                ):
                    gray = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                    canvas.paste(
                        Image.fromarray(gray[y : y + 160, x : x + 160]).convert("RGB"),
                        (160 * col, row * 180 + 20),
                    )
                    painter.text(
                        (160 * col + 2, row * 180 + 2), f"{name[4:]} {label}", fill="black"
                    )
                crops.append(dict(case=name, draw=draw, crop=[x, y, x + 160, y + 160]))
            filename = f"{split}-{selector}.png"
            canvas.save(out / filename)
            files[filename] = dict(sha256=sha(out / filename), crops=crops, inspected=False)
    write(out / "manifest.json", dict(files=files, inspection_complete=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "preflight", "train", "render"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root, args.parent.resolve())
    elif args.action == "freeze":
        freeze(root)
    elif args.action == "render":
        render(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, args.action == "preflight")
