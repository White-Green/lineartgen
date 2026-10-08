"""Declare four paired guide-encoding trajectories before training or inference."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write
from parameter_blend_study import original_source
from prepare_source_guide_study import relocate

from lineartgen.config import load_config, save_config
from lineartgen.data.pipeline import dataset_description, discover, split_experiment
from lineartgen.models.serialization import load_inference
from lineartgen.training import lineart_state
from lineartgen.training.lineart_study import create_arm, study_methods


def prepare(coordinator, strokes, mixed, control):
    if coordinator.exists():
        raise FileExistsError(coordinator)
    for root in (strokes, mixed, control):
        original_source(root)
    parents = {"strokes": (strokes, "source_ink_guard"), "mixed": (mixed, "mixed_source_ink_guard")}
    roots = {family: coordinator.parent / f"ink-guide-{family}-20261008" for family in parents}
    assert all(not p.exists() for p in roots.values())
    plan = dict(
        seed=20261007,
        independent_confirmation_seed=20261005,
        encodings=["ink", "sqrt_ink"],
        corruption_families=["strokes", "mixed"],
        trained_endpoints={
            family: dict(
                root=str(roots[family]),
                methods={encoding: f"{encoding}_{family}" for encoding in ("ink", "sqrt_ink")},
            )
            for family in parents
        },
        alpha_grid=[0.0, 1 / 16, 1 / 8, 1 / 4, 1.0],
        alpha_meaning="(1-alpha)*strokes + alpha*mixed within the SAME guide encoding",
        step=2000,
        total_schedule_steps=10000,
        primary_weights="ema",
        parameters=126828,
        selection=(
            "Apply unchanged STRICT FINAL gates to all ten fixed 2k EMA models; "
            "native visual review; select at most one by summed primary halo gains. "
            "No permissive 2k screen and no automatic 10k continuation."
        ),
        confirmation=(
            "Only a passing selected model advances to new fine-tuning seed20261005 "
            "with a newly trained matched control, fixed encoding/alpha, then sealed fresh draws. "
            "Seed20261006 informed this hypothesis and will not count as independent confirmation."
        ),
        intervention=(
            "Guide is the original augmented rough source, downsampled before per-scale encoding. "
            "ink=(1-guide)/2; sqrt_ink=sqrt(clip((1-guide)/2,0,1)). Both are zero on white. "
            "Keep the same two bias-free 3x3 zero-initialized projections."
        ),
        unchanged=(
            "Common 100k EMA, parameter count, losses/coefficients, clean examples, augmentation "
            "family settings, independent Gaussian noise, masks, depths1-6, stage weights, "
            "sqrt(batch) LR, optimizer/EMA reset and original 10k clock"
        ),
        prior_rejection=(
            "Raw-guide alpha1/16 EMA passed primary development but failed independent "
            "seed20261006 on cuts. That study remains rejected; this is a new experiment."
        ),
        control=dict(
            root=str(control), method="control", source_sha256=sha(control / "source.json")
        ),
        parents={
            family: dict(root=str(p), method=n, source_sha256=sha(p / "source.json"))
            for family, (p, n) in parents.items()
        },
        acceptance_sha256=sha(control / "acceptance.json"),
        confirmation_plan_sha256=sha(control / "confirmation-plan.json"),
    )
    write(coordinator / "plan.json", plan)
    for family, (parent, oldname) in parents.items():
        root = roots[family]
        (root / "configs").mkdir(parents=True)
        for name in (
            "dataset.json",
            "evaluation.json",
            "acceptance.json",
            "supplemental-panels.json",
            "confirmation-plan.json",
        ):
            assert read(parent / name) == read(strokes / name) == read(control / name)
            shutil.copyfile(parent / name, root / name)
        write(root / "plan.json", plan)
        c = relocate(parent / f"configs/{oldname}.json", parent.parent.parent)
        assert c.train.seed == 20261007 and c.train.steps == 10000
        assert c.model.source_conditioning and c.model.guide_encoding == "raw"
        assert c.sketch.mode == family
        methods = plan["trained_endpoints"][family]["methods"]
        for encoding, name in methods.items():
            updated = c.model_copy(
                update={"model": c.model.model_copy(update={"guide_encoding": encoding})}
            )
            save_config(updated, root / f"configs/{name}.json")
            check = root / "initial-checks" / encoding
            check.mkdir(parents=True)
            save_config(updated, check / "config.json")
            shutil.copyfile(parent / "protocol.json", check / "protocol.json")
            shutil.copyfile(parent / "evaluation.json", check / "evaluation.json")
        shutil.copyfile(root / f"configs/{methods['ink']}.json", root / "config.json")
        original = read(parent / "protocol.json")
        protocol = {
            k: original[k]
            for k in (
                "initial",
                "initial_files",
                "total_schedule_steps",
                "patch_size",
                "prior_ramp_steps",
                "selected_panels",
            )
        }
        protocol.update(
            methods=list(methods.values()),
            milestones=[2000],
            primary_weights="ema",
            step=2000,
            plan_sha256=sha(root / "plan.json"),
            counterpart=dict(root=str(parent), method=oldname),
            next_phase="Evaluate both endpoints and fixed within-encoding blends at 2k",
        )
        write(root / "protocol.json", protocol)
        write(root / "calibration.json", dict(coefficients={}, no_new_coefficient_fitting=True))
        replay = root / "baseline-replay"
        (replay / "configs").mkdir(parents=True)
        save_config(c, replay / f"configs/{oldname}.json")
        save_config(c, replay / "config.json")
        write(replay / "protocol.json", original | dict(methods=[oldname]))
        write(replay / "calibration.json", dict(coefficients={}))
        sources = discover(c.data)
        train, dev, final = split_experiment(sources, c.data, c.train.seed)
        assert dataset_description(sources, train, dev, final) == read(root / "dataset.json")
        # Existing initial outputs are identical at the common 100k parent.
        shutil.copytree(parent / "evaluation/step-0-ema", root / "evaluation/step-0-ema")
    print(
        dict(coordinator=str(coordinator), roots={k: str(v) for k, v in roots.items()}), flush=True
    )


def audit_replay(root):
    protocol = read(root / "protocol.json")
    parent, method = Path(protocol["counterpart"]["root"]), protocol["counterpart"]["method"]
    initial, _ = load_inference(protocol["initial"])

    def restore(directory):
        names = study_methods(read(directory / "protocol.json"))
        arms = {n: create_arm(load_config(directory / f"configs/{n}.json"), initial) for n in names}
        with ocp.CheckpointManager(
            (directory / "preflight/checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        return {method: arms[method]}

    before, after = restore(parent), restore(root / "baseline-replay")
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload(before, 6),
        lineart_state.payload(after, 6),
    )
    write(
        root / "baseline-replay-audit.json",
        dict(
            full_model_optimizer_ema_exact=True,
            step=6,
            method=method,
            parent=str(parent),
            old_preflight_sha256=sha(parent / "preflight/results.json"),
            new_preflight_sha256=sha(root / "baseline-replay/preflight/results.json"),
            script_sha256=sha(Path(__file__)),
        ),
    )
    reports = {}
    for encoding in ("ink", "sqrt_ink"):
        path = root / "initial-checks" / encoding / "initial-guide-audit.json"
        x = read(path)
        assert x["all_exact"] and len(x["rows"]) == 32
        reports[encoding] = dict(path=str(path), sha256=sha(path))
    write(root / "initial-guide-audit.json", dict(reports=reports, all_exact=True, comparisons=64))
    print(dict(root=str(root), replay_exact=True, zero_guide_forward_comparisons=64), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "audit-replay"])
    parser.add_argument("--root", type=Path, required=True)
    for name in ("strokes", "mixed", "control"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            args.root.resolve(),
            args.strokes.resolve(),
            args.mixed.resolve(),
            args.control.resolve(),
        )
    else:
        audit_replay(args.root.resolve())
