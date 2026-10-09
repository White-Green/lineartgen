"""Prepare/replay the fixed preclip extension against its clipped counterparts."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write
from diagnose_learnability import arms_from_root, verify
from prepare_opacity_fitting import prepare_opacity

from lineartgen.config import ExperimentConfig, load_config, save_config
from lineartgen.training import lineart_state


def prepare(root, parent, tempered, clipped):
    verify(clipped)
    assert read(clipped / "workflow-status.json")["stage"] == "complete_rejected"
    prepare_opacity(root, parent, tempered)
    old = read(clipped / "plan.json")
    aliases = {"warm_preclip_010": "warm_opacity_010", "warm_preclip_030": "warm_opacity_030"}
    files = [clipped / n for n in ("source.json", "plan.json")]
    files += [clipped / f"evaluation/step-1200/{n}.json" for n in ("metrics", "summary")]
    for name in aliases.values():
        files.append(clipped / f"configs/{name}.json")
        for mode in ("raw", "ema"):
            files += [
                clipped / f"snapshots/step-1200/{name}/{mode}/{n}"
                for n in ("weights.npz", "config.json", "metadata.json")
            ]
    binding = {str(p.relative_to(clipped)): sha(p) for p in files}
    for destination in (root, root / "baseline-replay"):
        plan = read(destination / "plan.json")
        plan.update(
            experiment="physical preclip opacity deficit",
            plan_document_sha256=sha(Path("docs/preclip-opacity-fitting-plan-20261009.md")),
            clipped_reference=str(clipped),
            clipped_reference_files=binding,
            clipped_counterparts=aliases,
            training_condition="Only after clipped-opacity rejection and completed native review",
        )
        if destination == root:
            for name in plan["arms"]:
                (destination / f"configs/{name}.json").unlink()
            for name, counterpart in aliases.items():
                before = load_config(clipped / f"configs/{counterpart}.json")
                values = before.model_dump()
                values["loss"]["opacity_deficit_preclip"] = True
                after = ExperimentConfig.model_validate(values)
                back = after.model_dump()
                back["loss"]["opacity_deficit_preclip"] = False
                assert back == before.model_dump()
                save_config(after, destination / f"configs/{name}.json")
            plan.update(
                arms=list(aliases),
                warm_start_methods=list(aliases),
                counterparts={n: "warm_detail16" for n in aliases},
                detail_channels={n: 16 for n in aliases},
                tone_balance_powers={n: 0.5 for n in aliases},
                opacity_deficit_weights={
                    n: load_config(destination / f"configs/{n}.json").loss.opacity_deficit_weight
                    for n in aliases
                },
                opacity_deficit_preclip=True,
            )
        write(destination / "plan.json", plan)
    replay = root / "clipped-replay"
    (replay / "configs").mkdir(parents=True)
    plan = dict(old)
    for split in ("train", "probe"):
        for suffix in ("json", "npz"):
            shutil.copyfile(clipped / f"{split}.{suffix}", replay / f"{split}.{suffix}")
        assert sha(replay / f"{split}.npz") == old["data"][split]["sha256"]
    plan.update(
        data={s: dict(old["data"][s], path=str(replay / f"{s}.npz")) for s in ("train", "probe")},
        replay_of=str(clipped),
    )
    write(replay / "plan.json", plan)
    for name in old["arms"]:
        save_config(load_config(clipped / f"configs/{name}.json"), replay / f"configs/{name}.json")
    print(
        dict(prepared=str(root), only_paired_change="opacity_deficit_preclip", data_exact=True),
        flush=True,
    )


def verify_clipped(root):
    plan = read(root / "plan.json")
    folder = Path(plan["clipped_reference"])
    verify(folder)
    for name, digest in plan["clipped_reference_files"].items():
        assert sha(folder / name) == digest, name
    return folder


def audit_clipped(root):
    parent = verify_clipped(root)

    def restore(folder):
        arms = arms_from_root(folder)
        with ocp.CheckpointManager(
            (folder / "preflight/checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        return lineart_state.payload(arms, 6)

    jax.tree.map(np.testing.assert_array_equal, restore(parent), restore(root / "clipped-replay"))
    write(
        root / "clipped-replay-audit.json",
        dict(
            step=6,
            full_state_exact=True,
            methods=read(parent / "plan.json")["arms"],
            parent_source_sha256=sha(parent / "source.json"),
            replay_source_sha256=sha(root / "clipped-replay/source.json"),
        ),
    )
    print(dict(clipped_opacity_replay_exact=True), flush=True)


def compare_clipped(root):
    reference = verify_clipped(root)
    plan = read(root / "plan.json")
    fields = ("mode", "split", "denoise_steps", "family", "partial", "strength")
    old = {
        (x["method"], *(x[k] for k in fields)): x
        for x in read(reference / "evaluation/step-1200/summary.json")
    }
    rows = []
    for x in read(root / "evaluation/step-1200/summary.json"):
        before = old[(plan["clipped_counterparts"][x["method"]], *(x[k] for k in fields))]
        assert x["conditions"] == before["conditions"]
        rows.append(
            dict(
                method=x["method"],
                **{k: x[k] for k in fields},
                before=before["means"],
                after=x["means"],
                delta={k: x["means"][k] - before["means"][k] for k in before["means"]},
            )
        )
    assert len(rows) == 432
    write(
        root / "clipped-opacity-deltas.json",
        dict(
            comparisons=rows,
            descriptive_only=True,
            adoption=False,
            reference_summary_sha256=sha(reference / "evaluation/step-1200/summary.json"),
            plan_sha256=sha(root / "plan.json"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "audit-clipped", "compare"))
    p.add_argument("--root", type=Path, required=True)
    for name in ("parent", "tempered", "clipped"):
        p.add_argument(f"--{name}", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root.resolve(), a.parent.resolve(), a.tempered.resolve(), a.clipped.resolve())
    elif a.action == "audit-clipped":
        audit_clipped(a.root.resolve())
    else:
        compare_clipped(a.root.resolve())
