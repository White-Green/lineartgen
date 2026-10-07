"""Paired resumable fine-tuning of the five frozen lineart objectives."""

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import training_inputs
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart import make_profiles
from lineartgen.training.lineart_adversarial import patch_coordinates
from lineartgen.training.lineart_study import create_arm, study_methods, update_arm


def verify(root):
    protocol = read(root / "protocol.json")
    for name, digest in protocol["initial_files"].items():
        assert sha(Path(protocol["initial"]) / name) == digest, name
    for path, digest in read(root / "source.json")["files"].items():
        assert sha(Path(path)) == digest, f"Study source changed: {path}"
    for path, digest in read(root / "source.json")["settings"].items():
        assert sha(root / path) == digest, f"Study setting changed: {path}"
    return protocol


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            destination = root / "snapshots" / f"step-{step}" / name / mode
            if destination.exists():
                load_inference(destination)  # Refuse broken snapshots after an interrupted save.
                continue
            temporary = destination.with_name(mode + ".partial")
            if temporary.exists():
                import shutil

                shutil.rmtree(temporary)
            model = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(model, arm["ema"])
            save_inference(
                model,
                arm["config"],
                temporary,
                provenance=f"Lineart loss study {name}, additional step {step}, {mode}",
            )
            temporary.rename(destination)


def restore_journal(root, completed):
    """Archive uncommitted log records so a resumed run cannot count them twice."""
    path = root / "training.jsonl"
    kept, removed = [], []
    lines = path.read_text().splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            removed.append(line)  # The process may have stopped during the final write.
            continue
        (kept if record["step"] <= completed else removed).append(line)
    progress_path = root / "progress.json"
    progress = read(progress_path) if progress_path.exists() else None
    if removed or (progress is not None and progress["step"] > completed):
        write(
            root / "resume-history" / f"{time.time_ns()}.json",
            dict(
                restored_step=completed, rolled_back_log_lines=removed, previous_progress=progress
            ),
        )
        temporary = path.with_suffix(".jsonl.partial")
        temporary.write_text("".join(line + "\n" for line in kept))
        temporary.replace(path)
        if kept:
            write(progress_path, json.loads(kept[-1]))
        else:
            progress_path.unlink(missing_ok=True)


def train(root, end):
    protocol = verify(root)
    methods = study_methods(protocol)
    config = load_config(root / "config.json")
    if not 0 <= end <= config.train.steps:
        raise ValueError("Invalid end step")
    initial, _ = load_inference(protocol["initial"])
    sources = discover(config.data)
    training, dev, final = split_experiment(sources, config.data, config.train.seed)
    assert dataset_description(sources, training, dev, final) == read(root / "dataset.json")
    configs = {name: load_config(root / "configs" / f"{name}.json") for name in methods}
    arms = {
        name: create_arm(
            settings, initial, adversarial=name == "patch_prior", patch_size=protocol["patch_size"]
        )
        for name, settings in configs.items()
    }
    prepare = jax.jit(
        lambda x, k, levels: training_inputs(
            x, k, config.diffusion, config.matching, levels=levels, sketch=config.sketch
        ),
        static_argnums=2,
    )
    coefficients = read(root / "calibration.json")["coefficients"]
    with checkpoints.manager(root / "checkpoints", keep=3) as saved:
        completed = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        restore_journal(root, completed)
        if saved.latest_step() is None:
            snapshot(root, arms, 0)
        if completed >= end:
            snapshot(root, arms, completed)
            return completed
        stream = batches(
            training,
            config.data,
            batch_size=config.train.effective_batch_size,
            count=end - completed,
            seed=config.train.seed,
            start=completed,
        )
        key = jax.random.PRNGKey(config.train.seed)
        print(f"Training {completed} -> {end}; devices {jax.devices()}", flush=True)
        try:
            for step, batch in enumerate(stream, start=completed):
                started = time.perf_counter()
                clean, records = mix_geometry(
                    batch.images,
                    config.geometry,
                    seed=config.train.seed,
                    step=step,
                    image_scale=config.data.image_scale,
                    include_paths=True,
                )
                profiles = make_profiles(
                    records, len(clean), len(clean[0]), seed=config.train.seed, step=step
                )
                coordinates = patch_coordinates(
                    clean, seed=config.train.seed, step=step, size=protocol["patch_size"]
                )
                levels = config.diffusion.levels_for_scale(
                    config.data.crop_size, config.model.minimum_size, batch.scale_level
                )
                prepared = prepare(jax.device_put(clean), jax.random.fold_in(key, step), levels)
                profiles, coordinates = jax.device_put((profiles, coordinates))
                jax.block_until_ready(prepared)
                preparation_seconds = time.perf_counter() - started
                order = list(methods[step % len(methods) :] + methods[: step % len(methods)])
                metrics = {}
                for name in order:
                    begin = time.perf_counter()
                    weight = coefficients.get("patch_prior", 0.0) * min(
                        step / protocol["prior_ramp_steps"], 1
                    )
                    row = update_arm(
                        arms[name], prepared, profiles, coordinates, prior_weight=weight
                    )
                    metrics[name] = row | dict(
                        update_seconds=time.perf_counter() - begin,
                        learning_rate=float(arms[name]["schedule"](step)),
                    )
                completed = step + 1
                record = dict(
                    step=completed,
                    recursion_levels=levels,
                    scale_level=batch.scale_level,
                    generated=len(records),
                    profiles=float(np.asarray(profiles.valid).sum()),
                    preparation_seconds=preparation_seconds,
                    arms=metrics,
                    total_seconds=time.perf_counter() - started,
                )
                if completed % config.train.log_every == 0 or completed == end:
                    with (root / "training.jsonl").open("a") as output:
                        output.write(json.dumps(record, allow_nan=False) + "\n")
                    write(root / "progress.json", record)
                    print(json.dumps(record), flush=True)
                stopping = (root / "stop").exists()
                if completed % config.train.checkpoint_every == 0 or completed == end or stopping:
                    lineart_state.save(saved, arms, completed)
                    write(
                        root / "checkpoint.json",
                        dict(
                            step=completed, clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest()
                        ),
                    )
                if completed == end or stopping:
                    snapshot(root, arms, completed)
                if stopping:
                    break
        finally:
            stream.close()
        return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Completed additional step {train(root, args.end)}", flush=True)


if __name__ == "__main__":
    main()
