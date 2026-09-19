"""Copy a stopped campaign, changing only the lossless image reader location."""

import argparse
import datetime
import fcntl
import hashlib
import json
import shutil
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import ExperimentConfig, load_config, save_config
from lineartgen.data.images import cache_matches
from lineartgen.models import create_model
from lineartgen.training import checkpoints
from lineartgen.training.steps import create_optimizer


def with_cache(config, cache):
    values = config.model_dump(mode="json")
    values["data"]["prepared_root"] = str(cache.resolve())
    result = ExperimentConfig.model_validate(values)
    original, changed = config.model_dump(mode="json"), result.model_dump(mode="json")
    original["data"].pop("prepared_root")
    changed["data"].pop("prepared_root")
    assert original == changed
    return result


def migrate_configs(directory, cache):
    changes = {}
    for path in sorted(directory.rglob("config.json")):
        before = sha(path)
        save_config(with_cache(load_config(path), cache), path)
        changes[str(path.relative_to(directory))] = dict(before=before, after=sha(path))
    for path in sorted(directory.rglob("variants.json")):
        before = sha(path)
        write(
            path,
            {
                name: with_cache(ExperimentConfig.model_validate(value), cache).model_dump(
                    mode="json"
                )
                for name, value in read(path).items()
            },
        )
        changes[str(path.relative_to(directory))] = dict(before=before, after=sha(path))
    return changes


def restored_payload(run, step):
    config = load_config(run / "config.json")
    model = create_model(config.model, rngs=nnx.Rngs(config.train.initialization_seed))
    optimizer, _ = create_optimizer(model, config)
    ema = nnx.state(model, nnx.Param)
    with checkpoints.manager(run / "checkpoints", config.train.keep_checkpoints) as saved:
        restored, ema = checkpoints.restore(saved, model, optimizer, ema, step)
    assert restored == step and int(optimizer.step[...]) == step
    return jax.device_get(checkpoints.payload(model, optimizer, ema, step))


def compare_states(original, copied, step):
    left, right = restored_payload(original, step), restored_payload(copied, step)
    jax.tree.map(np.testing.assert_array_equal, left, right)
    digest = hashlib.sha256()
    leaves, structure = jax.tree.flatten(left)
    digest.update(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        assert np.isfinite(array).all()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return dict(step=step, arrays=len(leaves), exact=True, payload_sha256=digest.hexdigest())


def validate_cache(cache, original):
    preparation = read(cache / "preparation.json")
    source = read(cache / "preparation-source.json")
    assert preparation["stage"] == "complete" and preparation["final_pixels_untouched"]
    assert source["config_sha256"] == sha(original / "config.json")
    config = load_config(original / "config.json")
    allowed, excluded = set(source["selected_sources"]), set(source["excluded_final_sources"])
    assert len(allowed) == 955 and len(excluded) == 64 and not allowed & excluded
    expected = {(name, i) for name in allowed for i in range(len(config.data.scale_dirs))}
    actual = {(r["source"], r["scale"]) for r in preparation["records"]}
    assert expected == actual and len(preparation["records"]) == len(expected) == 5730
    for row in preparation["records"]:
        source_path, destination = Path(row["source_path"]), Path(row["destination"])
        assert destination.is_relative_to(cache) and destination.is_file()
        assert cache_matches(source_path, destination)
        assert destination.stat().st_size == row["tiff_bytes"]
        assert source_path.stat().st_size == row["png_bytes"]
        assert row["pixels_exact"] and len(row["crop_checks"]) == 3
    assert not any(
        (cache / scale / name).with_suffix(".png.tif").exists()
        for scale in config.data.scale_dirs
        for name in excluded
    )
    return preparation


def migrate(original, destination, cache):
    original, destination, cache = original.resolve(), destination.resolve(), cache.resolve()
    assert original != destination and not destination.exists()
    assert (original / "stop").exists()
    assert read(original / "status.json")["stage"] == "stop_requested"
    assert not (original / "selection-seal.json").exists()
    preparation = validate_cache(cache, original)
    frozen = read(original / "training-source.json")
    for path, digest in frozen["file_sha256"].items():
        assert sha(path) == digest, path
    # Full copies: future checkpoint retention must never modify original payloads.
    shutil.copytree(
        original,
        destination,
        ignore=shutil.ignore_patterns("stop", "controller.lock", "io-comparison"),
    )
    before = {
        str(p.relative_to(destination)): sha(p) for p in destination.rglob("*") if p.is_file()
    }
    changes = migrate_configs(destination, cache)
    base = destination / "base.toml"
    text = base.read_text()
    assert "prepared_root" not in text and text.count("[data]") == 1
    base.write_text(text.replace("[data]", f"[data]\nprepared_root = {json.dumps(str(cache))}"))
    assert load_config(base) == load_config(destination / "config.json")
    changes["base.toml"] = dict(before=before["base.toml"], after=sha(base))
    preserved = {}
    for name, digest in before.items():
        if name not in changes:
            assert sha(destination / name) == digest
            preserved[name] = digest
        assert sha(original / name) == digest, f"Original modified during copy: {name}"
    states = {}
    for arm in sorted((destination / "arms").iterdir()):
        step = read(arm / "training/progress.json")["step"]
        states[arm.name] = compare_states(
            original / "arms" / arm.name / "training/model", arm / "training/model", step
        )
    # Retain every original guard and add the derived config and reader provenance.
    for path in (
        original / "training-source.json",
        original / "cache-migration-source.json",
        original / "cache-migration-source.tar.gz",
        destination / "base.toml",
        destination / "config.json",
        destination / "protocol.json",
        cache / "preparation.json",
        cache / "preparation-source.json",
        Path(__file__).resolve(),
        Path(__file__).with_name("run_cached_campaign.py").resolve(),
        Path(__file__).with_name("run_completion_campaign.py").resolve(),
        Path(__file__).with_name("prepare_campaign_cache.py").resolve(),
        Path(__file__).with_name("benchmark_io_cache.py").resolve(),
        Path(__file__).with_name("audit_campaign_cache.py").resolve(),
    ):
        frozen["file_sha256"][str(path)] = sha(path)
    write(destination / "training-source.json", frozen)
    record = dict(
        created_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        original=str(original),
        destination=str(destination),
        cache=str(cache),
        source_sha256=sha(__file__),
        only_setting_changed="data.prepared_root",
        configurations=changes,
        preserved_files=preserved,
        checkpoints=states,
        cache_manifest_sha256=sha(cache / "preparation.json"),
        cache_files=preparation["files"],
        verified_crops=preparation["verified_crops"],
        timing_segments={
            name: dict(png_through_step=row["step"], tiff_from_step=row["step"] + 1)
            for name, row in states.items()
        },
        note="Original directory retained. Historical timings and reports retain original "
        "paths and PNG measurements. Frozen training-source extends original guards with "
        "derived configs and migration provenance; numerical code and decision rules unchanged.",
    )
    write(destination / "migration.json", record)
    frozen["file_sha256"][str(destination / "migration.json")] = sha(destination / "migration.json")
    write(destination / "training-source.json", frozen)
    write(destination / "status.json", dict(stage="cache_migration_verified", checkpoints=states))
    return destination / "migration.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    with (args.original / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(migrate(args.original, args.root, args.cache))


if __name__ == "__main__":
    main()
