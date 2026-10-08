"""Prepare a loss-only comparison on the unchanged local fitting pairs."""

import argparse
import shutil
from pathlib import Path

from campaign_protocol import read, sha, write
from diagnose_learnability import verify

from lineartgen.config import ExperimentConfig, load_config, save_config


def prepare(root, parent):
    verify(parent)
    if root.exists():
        raise FileExistsError(root)
    old = read(parent / "plan.json")
    assert old["updates"] == 1200 and old["seed"] == 2026100801
    base = load_config(parent / "configs/warm_detail16.json")
    assert base.model.detail_channels == 16 and base.model.source_conditioning
    assert not base.model.progress_conditioning and base.diffusion.training_path == "rough"
    assert base.loss.balance_tones and base.loss.tone_balance_power == 1
    assert base.loss.pixel_loss == "clipped_mse" and base.train.steps == 1200
    for destination, powers in (
        (root, {"warm_tone_sqrt": 0.5, "warm_tone_plain": 0}),
        (root / "baseline-replay", {"warm_detail16": 1}),
    ):
        (destination / "configs").mkdir(parents=True)
        plan = dict(old)
        for split in ("train", "probe"):
            for suffix in ("json", "npz"):
                shutil.copyfile(parent / f"{split}.{suffix}", destination / f"{split}.{suffix}")
            assert sha(destination / f"{split}.npz") == old["data"][split]["sha256"]
        plan.update(
            arms=list(powers),
            warm_start_methods=list(powers),
            counterparts={n: "warm_detail16" for n in powers},
            detail_channels={n: 16 for n in powers},
            tone_balance_powers=powers,
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            plan_document_sha256=sha(Path("docs/tone-balance-fitting-plan-20261008.md")),
            data={
                s: dict(old["data"][s], path=str(destination / f"{s}.npz"))
                for s in ("train", "probe")
            },
            primary="raw",
            primary_steps=1,
            evaluation_steps=[1, 2, 5],
            advance_rule="Blur/.1 halo reduction >=10% in both splits; family/split/strength/"
            "mask F1, ink deficit and halo safeguards; native support. No adoption.",
            campaign_confirmation_generated=False,
            reference=dict(
                method="warm_detail16",
                step=1200,
                checkpoints={
                    mode: dict(
                        path=str(parent / f"snapshots/step-1200/warm_detail16/{mode}"),
                        files={
                            n: sha(parent / f"snapshots/step-1200/warm_detail16/{mode}/{n}")
                            for n in ("weights.npz", "config.json", "metadata.json")
                        },
                    )
                    for mode in ("raw", "ema")
                },
            ),
        )
        write(destination / "plan.json", plan)
        for name, power in powers.items():
            values = base.model_dump()
            values["loss"]["tone_balance_power"] = power
            config = ExperimentConfig.model_validate(values)
            check = config.model_dump()
            check["loss"]["tone_balance_power"] = 1
            assert check == base.model_dump()
            save_config(config, destination / f"configs/{name}.json")
    print(dict(root=str(root), data_exact=True, only_difference="tone_balance_power"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    args = p.parse_args()
    prepare(args.root.resolve(), args.parent.resolve())
