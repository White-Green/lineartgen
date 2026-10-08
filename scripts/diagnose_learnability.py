"""Small fixed training-only fitting probe; never a campaign adoption decision."""

import argparse
import fcntl
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx
from PIL import Image, ImageDraw
from train_lineart_loss_study import restore_journal

from lineartgen.config import GeometryDataConfig, load_config, save_config
from lineartgen.data.geometry import KINDS, draw_geometry
from lineartgen.diffusion import (
    pyramid,
    recursive_forward_with_preclip,
    restoration_target,
    training_inputs,
)
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.sketch import gaussian_blur
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm
from lineartgen.training.steps import accumulate_gradients, make_steps

SEED = 2026100801
NAMES = ("warm_hard", "warm_interval", "random_small", "random_wide")


def generate_pairs(seed, *, size=512):
    config = GeometryDataConfig(minimum_width=0.6, maximum_width=6, supersampling=4)
    clean, rows = [], []
    for variant in range(2):
        for index, kind in enumerate(KINDS):
            key = [seed, variant, index, 0x464954]
            gray, metadata = draw_geometry(np.random.default_rng(key), size, config, kind=kind)
            clean.append(np.clip((gray * 2 - 1) * 1.1, -1, 1))
            rows.append(dict(kind=kind, variant=variant, generation_key=key, metadata=metadata))
    clean.append(np.ones((size, size, 1), np.float32))
    rows.append(dict(kind="blank", variant=0, generation_key=[seed, 0x424C414E]))
    clean = np.stack(clean)
    ink = (1 - jnp.asarray(clean)) * 0.5
    blurred = np.asarray(gaussian_blur(ink, jnp.full((len(clean),), 2.0), 2.0))
    sources = [clean, 1 - (1 - clean) * 0.3, 1 - 2 * blurred * 0.3]
    records = [
        dict(row, family=family, pair=len(rows) * f + i)
        for f, family in enumerate(("clean", "tone", "blur"))
        for i, row in enumerate(rows)
    ]
    return np.concatenate([clean] * 3), np.concatenate(sources), records


def prepare_inputs(clean, source, key, config, levels):
    if config.model.progress_conditioning:
        return training_inputs(
            clean,
            key,
            config.diffusion,
            config.matching,
            levels=levels,
            with_source=True,
            with_progress=True,
            paired_source=source,
        )
    prepared = training_inputs(
        source, key, config.diffusion, config.matching, levels=levels, with_source=True
    )
    return prepared._replace(
        targets=pyramid(restoration_target(clean, source, prepared.noise_levels[0]), levels)
    )


def prepare(root, base_path, initial_path):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    base = load_config(base_path)
    base = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={"source_conditioning": True, "guide_encoding": "raw"}
            ),
            "sketch": base.sketch.model_copy(update={"mode": "none"}),
            "geometry": base.geometry.model_copy(update={"probability": 0}),
            "train": base.train.model_copy(
                update={
                    "steps": 1200,
                    "batch_size": 4,
                    "accumulate_steps": 1,
                    "learning_rate": 1e-4,
                    "minimum_learning_rate": 1e-5,
                    "warmup_steps": 50,
                    "seed": SEED,
                    "model_seed": 42,
                    "checkpoint_every": 100,
                    "log_every": 25,
                }
            ),
        }
    )
    assert base.loss.contrast_weight == 0.01 and base.loss.line_region_weight == 0
    (root / "configs").mkdir()
    for name in NAMES:
        config = base
        if name == "warm_interval":
            config = config.model_copy(
                update={"loss": config.loss.model_copy(update={"pixel_loss": "interval_mse"})}
            )
        elif name == "random_wide":
            config = config.model_copy(
                update={"model": config.model.model_copy(update={"channels": (8, 20, 40, 80)})}
            )
        save_config(config, root / f"configs/{name}.json")
    data = {}
    for split, seed in (("train", SEED), ("probe", SEED + 1)):
        clean, source, rows = generate_pairs(seed)
        destination = root / f"{split}.npz"
        np.savez_compressed(destination, clean=clean, source=source)
        write(root / f"{split}.json", rows)
        data[split] = dict(path=str(destination), sha256=sha(destination), pairs=len(rows))
    write(
        root / "plan.json",
        dict(
            seed=SEED,
            arms=list(NAMES),
            updates=1200,
            milestones=[0, 400, 1200],
            primary="raw",
            initial=str(initial_path),
            initial_files={p.name: sha(p) for p in initial_path.iterdir() if p.is_file()},
            data=data,
            adoption=False,
            campaign_confirmation_generated=False,
            plan_document_sha256=sha(Path("docs/learnability-plan-20261008.md")),
        ),
    )
    print(dict(prepared=str(root), pairs_per_split=57), flush=True)


def arms_from_root(root):
    plan = read(root / "plan.json")
    initial, _ = load_inference(plan["initial"])
    arms = {}
    warm_methods = plan.get("warm_start_methods", [n for n in plan["arms"] if n.startswith("warm")])
    for name in plan["arms"]:
        config = load_config(root / f"configs/{name}.json")
        start = (
            initial
            if name in warm_methods
            else create_model(config.model, rngs=nnx.Rngs(config.train.initialization_seed))
        )
        arm = create_arm(config, start)
        assert_initial_parameters(start, arm["model"])
        arm["gradient"], arm["update"], _ = make_steps(config)
        arms[name] = arm
    return arms


def freeze(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    paths = [Path(p).resolve() for p in tracked if p.endswith((".py", ".toml")) or p == "uv.lock"]
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    settings = [root / n for n in ("plan.json", "train.json", "probe.json")]
    settings += sorted((root / "configs").glob("*.json"))
    write(
        root / "source.json",
        dict(
            commit=commit,
            files={str(p): sha(p) for p in paths},
            settings={str(p.relative_to(root)): sha(p) for p in settings},
            archive_sha256=sha(root / "source.tar.gz"),
            versions={
                n: importlib.metadata.version(n)
                for n in ("jax", "jaxlib", "flax", "optax", "numpy")
            },
            purpose="Controlled fitting diagnostic; not campaign selection",
        ),
    )


def verify(root):
    plan = read(root / "plan.json")
    for name, digest in plan["initial_files"].items():
        assert sha(Path(plan["initial"]) / name) == digest
    for entry in plan["data"].values():
        assert sha(Path(entry["path"])) == entry["sha256"]
    manifest = read(root / "source.json")
    for path, digest in manifest["files"].items():
        assert sha(Path(path)) == digest, path
    for path, digest in manifest["settings"].items():
        assert sha(root / path) == digest, path


def build_batch(step, clean, source):
    indices = np.random.default_rng([SEED, step, 0x424154]).integers(len(clean), size=4)
    return jax.device_put(clean[indices]), jax.device_put(source[indices]), indices


def update(arm, prepared):
    metrics, gradient, norm = accumulate_gradients(
        arm["gradient"], arm["model"], prepared, arm["config"].train
    )
    assert np.isfinite(float(norm)) and np.isfinite(float(metrics["loss"]))
    _, arm["ema"] = arm["update"](arm["model"], arm["optimizer"], gradient, arm["ema"])
    jax.block_until_ready(arm["ema"])
    return {k: float(v) for k, v in metrics.items()} | {"gradient_norm": float(norm)}


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            p = root / f"snapshots/step-{step}/{name}/{mode}"
            model = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(model, arm["ema"])
            if not p.exists():
                save_inference(
                    model, arm["config"], p, provenance=f"Fitting probe {name} {step} {mode}"
                )
            restored, config = load_inference(p)
            assert config == arm["config"]
            jax.tree.map(
                np.testing.assert_array_equal,
                nnx.state(model, nnx.Param),
                nnx.state(restored, nnx.Param),
            )


def train(root, *, preflight=False, end=1200):
    verify(root)
    if not 1 <= end <= 1200:
        raise ValueError("Invalid fixed diagnostic end step")
    if not preflight:
        audit = read(root / "preflight/restore-audit.json")
        assert audit["step"] == 6 and audit["all_state_exact"]
    arms = arms_from_root(root)
    names = tuple(arms)
    data = np.load(root / "train.npz")
    clean, source = data["clean"], data["source"]
    preparers = {
        name: jax.jit(
            lambda a, b, k, levels, c=arm["config"]: prepare_inputs(a, b, k, c, levels),
            static_argnums=3,
        )
        for name, arm in arms.items()
    }
    destination = root / ("preflight" if preflight else "training")
    destination.mkdir(exist_ok=True)
    with checkpoints.manager(destination / "checkpoints", keep=3) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(destination, start)
        if not preflight and start in (0, 400, 1200):
            snapshot(root, arms, start)
        for step in range(start, 6 if preflight else end):
            begin = time.perf_counter()
            a, b, indices = build_batch(step, clean, source)
            stats = {}
            for name in names[step % len(names) :] + names[: step % len(names)]:
                prepared = preparers[name](
                    a, b, jax.random.fold_in(jax.random.PRNGKey(SEED), step), step % 6 + 1
                )
                stats[name] = update(arms[name], prepared)
            complete = step + 1
            row = dict(
                step=complete,
                depth=step % 6 + 1,
                indices=indices.tolist(),
                arms=stats,
                seconds=time.perf_counter() - begin,
            )
            if preflight or complete % 25 == 0 or complete == end:
                with (destination / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                write(destination / "progress.json", row)
                print(json.dumps(row), flush=True)
            if complete % 100 == 0 or preflight and complete == 6 or complete == end:
                lineart_state.save(saved, arms, complete)
            if not preflight and complete in (400, 1200):
                snapshot(root, arms, complete)
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


def fixed_inputs(clean, source, strength, seed, *, partial=False):
    level = np.full_like(source, strength)
    if partial:
        level[:, :128] = 0
        level[:, :, :170] = 0
    masks, guides = pyramid(jnp.asarray(level), 3), pyramid(jnp.asarray(source), 3)
    rng = np.random.default_rng(seed)
    images = tuple(
        g * jnp.sqrt(1 - t)
        + jnp.asarray(rng.standard_normal(g.shape, dtype=np.float32)) * jnp.sqrt(t)
        for g, t in zip(guides, masks, strict=True)
    )
    target = np.where(level > 0, clean, source)
    return images, masks, guides, target, level


def evaluate(root, step):
    verify(root)
    names = tuple(read(root / "plan.json")["arms"])
    out = root / f"evaluation/step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        for mode in ("raw", "ema"):
            model, _ = load_inference(root / f"snapshots/step-{step}/{name}/{mode}")
            forward = nnx.jit(
                lambda m, x, t, g: recursive_forward_with_preclip(
                    m,
                    x,
                    t,
                    guides=g,
                    progress=tuple(jnp.ones_like(a) for a in x)
                    if m.progress_conditioning
                    else None,
                )
            )
            for split in ("train", "probe"):
                data, records = np.load(root / f"{split}.npz"), read(root / f"{split}.json")
                predictions = []
                for partial in (False, True) if step == 1200 else (False,):
                    for strength in (0.1, 0.5):
                        for i, record in enumerate(records):
                            clean, source = data["clean"][i : i + 1], data["source"][i : i + 1]
                            x, t, g, target, level = fixed_inputs(
                                clean,
                                source,
                                strength,
                                SEED + 10000 + i + (1000 if split == "probe" else 0),
                                partial=partial,
                            )
                            pred, preclip = forward(model, x, t, g)
                            image, raw = np.asarray(pred[0])[0], np.asarray(preclip[0])[0]
                            target, level = target[0], level[0]
                            metrics = halo_metrics(image, target, level)
                            assert metrics["protected_max_error"] in (None, 0.0)
                            core = (target < 0.96) & (level > 0)
                            metrics["wrong_white_saturation"] = (
                                float((raw[core] >= 1).mean()) if core.any() else None
                            )
                            metrics["active_mse"] = float(((image - target) ** 2)[level > 0].mean())
                            row = dict(
                                method=name,
                                mode=mode,
                                split=split,
                                partial=partial,
                                strength=strength,
                                **record,
                                **metrics,
                            )
                            rows.append(row)
                            if not partial:
                                predictions.append(image)
                np.savez_compressed(
                    out / f"{name}-{mode}-{split}.npz", images=np.stack(predictions)
                )
                print(dict(evaluated=name, mode=mode, split=split, step=step), flush=True)
    write(
        out / "metrics.json",
        dict(
            step=step, results=rows, predictions=len(rows), source_sha256=sha(root / "source.json")
        ),
    )
    summaries = []
    for name in names:
        for mode in ("raw", "ema"):
            for split in ("train", "probe"):
                for family in ("clean", "tone", "blur"):
                    selected = [
                        r
                        for r in rows
                        if (r["method"], r["mode"], r["split"], r["family"])
                        == (name, mode, split, family)
                        and not r["partial"]
                        and r["kind"] != "blank"
                    ]
                    keys = (
                        "ink_f1_alpha50",
                        "ink_deficit_fraction",
                        "halo_opacity_excess",
                        "wrong_white_saturation",
                        "active_mse",
                    )
                    means = {
                        k: float(np.mean([r[k] for r in selected if r[k] is not None]))
                        for k in keys
                    }
                    failed = [
                        dict(pair=r["pair"], strength=r["strength"])
                        for r in selected
                        if r["ink_f1_alpha50"] < 0.95
                        or r["ink_deficit_fraction"] > 0.05
                        or (r["halo_opacity_excess"] or 0) > 0.02
                    ]
                    summaries.append(
                        dict(
                            method=name,
                            mode=mode,
                            split=split,
                            family=family,
                            means=means,
                            conditions=len(selected),
                            failed_fit=failed,
                        )
                    )
    write(out / "summary.json", summaries)
    render(root, step)


def render(root, step):
    out = root / f"evaluation/step-{step}"
    names = tuple(read(root / "plan.json")["arms"])
    for split in ("train", "probe"):
        data = np.load(root / f"{split}.npz")
        records = read(root / f"{split}.json")
        # Fixed first variant of nine kinds, blur/.5: native 160px core crops.
        indices = list(range(38, 47))
        predictions = {n: np.load(out / f"{n}-raw-{split}.npz")["images"] for n in names}
        canvas = Image.new("RGB", ((len(names) + 2) * 160, len(indices) * 180), "white")
        draw = ImageDraw.Draw(canvas)
        for row, i in enumerate(indices):
            clean = data["clean"][i]
            ys, xs = np.where(clean[..., 0] < 0.96)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            y, x = np.clip(cy - 80, 0, 352), np.clip(cx - 80, 0, 352)
            images = [clean, data["source"][i]] + [predictions[n][len(records) + i] for n in names]
            for col, (label, image) in enumerate(
                zip(("target", "source", *names), images, strict=True)
            ):
                gray = np.uint8(np.rint(np.clip((image[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(
                    Image.fromarray(gray[y : y + 160, x : x + 160]).convert("RGB"),
                    (col * 160, row * 180 + 20),
                )
                draw.text(
                    (col * 160 + 2, row * 180 + 2), f"{records[i]['kind']} {label}", fill="black"
                )
        canvas.save(out / f"{split}-blur05-native.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "freeze", "preflight", "train", "evaluate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base", type=Path)
    p.add_argument("--initial", type=Path)
    p.add_argument("--end", type=int, default=1200)
    p.add_argument("--step", type=int, default=1200)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root.resolve(), a.base.resolve(), a.initial.resolve())
    elif a.action == "freeze":
        freeze(a.root.resolve())
    else:
        with (a.root / "process.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.action == "evaluate":
                evaluate(a.root.resolve(), a.step)
            else:
                train(a.root.resolve(), preflight=a.action == "preflight", end=a.end)
