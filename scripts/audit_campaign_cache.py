"""Verify immutable PNG history, reader-only migration and lossless cache provenance."""

import argparse
from pathlib import Path

from campaign_protocol import read, sha, write
from migrate_campaign_cache import validate_cache, with_cache

from lineartgen.config import ExperimentConfig, load_config


def audit(root, *, payload_hashes=False):
    root = root.resolve()
    record = read(root / "migration.json")
    original, cache = Path(record["original"]), Path(record["cache"])
    assert record["destination"] == str(root)
    assert record["only_setting_changed"] == "data.prepared_root"
    assert all(r["exact"] for r in record["checkpoints"].values())
    frozen = read(root / "training-source.json")
    original_frozen = read(original / "training-source.json")
    for path, digest in original_frozen["file_sha256"].items():
        assert frozen["file_sha256"][path] == digest
    for path, digest in frozen["file_sha256"].items():
        assert sha(path) == digest, path
    for name, change in record["configurations"].items():
        assert sha(original / name) == change["before"]
        assert sha(root / name) == change["after"]
        if name.endswith("variants.json"):
            expected = {
                k: with_cache(ExperimentConfig.model_validate(v), cache).model_dump(mode="json")
                for k, v in read(original / name).items()
            }
            assert read(root / name) == expected
        else:
            assert load_config(root / name) == with_cache(load_config(original / name), cache)
    for name, digest in record["preserved_files"].items():
        # The continuation note may record the redirect; numerical history stays fixed.
        if name != "continuation.json":
            assert sha(original / name) == digest, name
    for arm, state in record["checkpoints"].items():
        path = Path("arms") / arm / "training/timings.jsonl"
        prefix = (original / path).read_bytes()
        assert (root / path).read_bytes().startswith(prefix)
        for name in ("metrics.jsonl", "dataset.json"):
            path = Path("arms") / arm / "training/model" / name
            assert (root / path).read_bytes().startswith((original / path).read_bytes())
        assert record["timing_segments"][arm] == dict(
            png_through_step=state["step"], tiff_from_step=state["step"] + 1
        )
    preparation = validate_cache(cache, original)
    assert sha(cache / "preparation.json") == record["cache_manifest_sha256"]
    if payload_hashes:
        for row in preparation["records"]:
            assert sha(row["source_path"]) == row["png_sha256"]
            assert sha(row["destination"]) == row["tiff_sha256"]
    result = dict(
        original=str(original),
        cache=str(cache),
        only_setting_changed="data.prepared_root",
        checkpoints=record["checkpoints"],
        cache_files=preparation["files"],
        verified_crops=preparation["verified_crops"],
        timing_segments=record["timing_segments"],
        historical_payload_hashes_verified=True,
        cache_payload_hashes_verified=payload_hashes,
        final_pixels_excluded=True,
        source_sha256=sha(__file__),
    )
    write(root / "cache-integrity.json", result)
    return root / "cache-integrity.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--payload-hashes", action="store_true")
    args = parser.parse_args()
    print(audit(args.root, payload_hashes=args.payload_hashes))


if __name__ == "__main__":
    main()
