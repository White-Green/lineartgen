"""Resume the unchanged campaign with a verified, lossless tiled-image reader."""

import argparse
import fcntl
from pathlib import Path

from analyze_completion_campaign import original_dependencies
from campaign_protocol import read, sha
from run_completion_campaign import Campaign

from lineartgen.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    migration = read(root / "migration.json")
    assert migration["destination"] == str(root)
    assert migration["only_setting_changed"] == "data.prepared_root"
    assert all(r["exact"] for r in migration["checkpoints"].values())
    assert sha(Path(migration["cache"]) / "preparation.json") == migration["cache_manifest_sha256"]
    assert load_config(root / "config.json") == load_config(root / "base.toml")
    assert load_config(root / "config.json").data.prepared_root.resolve() == Path(
        migration["cache"]
    )
    original_dependencies(root)
    with (root / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(root, lock)
        try:
            campaign.execute()
        except InterruptedError:
            return
        except Exception as exc:
            campaign.status("failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
