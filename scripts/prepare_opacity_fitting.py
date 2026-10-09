"""Prepare the fixed tempered-loss opacity-retention comparison."""

import argparse
from pathlib import Path

from campaign_protocol import read, sha, write
from diagnose_learnability import verify
from prepare_tone_fitting import prepare

from lineartgen.config import ExperimentConfig, load_config, save_config


def prepare_opacity(root, parent, tempered):
    verify(tempered)
    old_plan = read(tempered / "plan.json")
    old_config = load_config(tempered / "configs/warm_tone_sqrt.json")
    assert old_config.loss.tone_balance_power == 0.5
    assert old_config.loss.opacity_deficit_weight == 0
    assert read(tempered / "workflow-status.json")["stage"] == "complete_rejected"
    prepare(root, parent)
    files = [
        tempered / name
        for name in (
            "source.json",
            "plan.json",
            "configs/warm_tone_sqrt.json",
            "evaluation/step-1200/metrics.json",
            "evaluation/step-1200/summary.json",
            "evaluation/step-1200/aggregation-audit.json",
        )
    ]
    files += sorted((tempered / "evaluation/step-1200").glob("warm_tone_sqrt-*"))
    for mode in ("raw", "ema"):
        files += [
            tempered / f"snapshots/step-1200/warm_tone_sqrt/{mode}/{name}"
            for name in ("weights.npz", "config.json", "metadata.json")
        ]
    for destination in (root, root / "baseline-replay"):
        plan = read(destination / "plan.json")
        assert all(
            plan["data"][s]["sha256"] == old_plan["data"][s]["sha256"] for s in ("train", "probe")
        )
        plan.update(
            experiment="tempered tone weights with linear opacity retention",
            plan_document_sha256=sha(Path("docs/opacity-retention-fitting-plan-20261009.md")),
            tempered_reference=str(tempered),
            tempered_reference_files={str(p.relative_to(tempered)): sha(p) for p in files},
            training_condition=(
                "Wait for unit-tone path numerical completion; no shared GPU training"
            ),
        )
        if destination == root:
            weights = {"warm_opacity_010": 0.1, "warm_opacity_030": 0.3}
            for name in plan["arms"]:
                (destination / f"configs/{name}.json").unlink()
            for name, weight in weights.items():
                values = old_config.model_dump()
                values["loss"]["opacity_deficit_weight"] = weight
                config = ExperimentConfig.model_validate(values)
                back = config.model_dump()
                back["loss"]["opacity_deficit_weight"] = 0
                assert back == old_config.model_dump()
                save_config(config, destination / f"configs/{name}.json")
            plan.update(
                arms=list(weights),
                warm_start_methods=list(weights),
                counterparts={n: "warm_detail16" for n in weights},
                detail_channels={n: 16 for n in weights},
                tone_balance_powers={n: 0.5 for n in weights},
                opacity_deficit_weights=weights,
            )
        write(destination / "plan.json", plan)
    print(
        dict(
            prepared=str(root),
            data_exact=True,
            only_change_from_tempered="loss.opacity_deficit_weight",
        ),
        flush=True,
    )


def verify_tempered(root):
    plan = read(root / "plan.json")
    folder = Path(plan["tempered_reference"])
    verify(folder)
    for name, digest in plan["tempered_reference_files"].items():
        assert sha(folder / name) == digest, name
    return folder


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--tempered", type=Path, required=True)
    args = p.parse_args()
    prepare_opacity(args.root.resolve(), args.parent.resolve(), args.tempered.resolve())
