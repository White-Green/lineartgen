"""Freeze the selected corruption blend recipe under a second fine-tuning seed."""

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path

from campaign_protocol import read, sha, write
from compare_corruption_mixture_study import paired_config, settings
from prepare_source_guide_study import relocate

from lineartgen.config import save_config
from lineartgen.data.pipeline import dataset_description, discover, split_experiment


def historical_source(root):
    """The historical checkout moved on; validate the sealed archive, not current files."""
    source = read(root / "source.json")
    assert sha(root / "source.tar.gz") == source["archive_sha256"]
    with tarfile.open(root / "source.tar.gz") as archive:
        for path, digest in source["files"].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == digest, path
    for path, digest in source["settings"].items():
        assert sha(root / path) == digest, path
    return source


def prepare(root, selected, baseline):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    review = read(selected / "review.json")
    assert review["approved_methods"] == ["corruption_blend0625"]
    assert review["selected_alpha"] == 1 / 16 and review["weights"] == "ema"
    assert review["quality_gate_sha256"] == sha(selected / "quality-gate-with-baselines-2000.json")
    gate = read(selected / "quality-gate-with-baselines-2000.json")
    chosen = next(x for x in gate["results"] if x["method"] == "corruption_blend0625")
    assert not chosen["screening"] and chosen["numerical_gate_passed"]
    for path, digest in review["inspected_panels"].items():
        assert sha(path) == digest
    root.mkdir(parents=True)
    shutil.copyfile(selected / "review.json", root / "selection-review.json")
    baseline_source = historical_source(baseline)
    original = read(selected / "protocol.json")
    old_protocol = read(baseline / "protocol.json")
    assert original["initial_files"] == old_protocol["initial_files"]
    assert original["initial"] == old_protocol["initial"]
    assert read(selected / "dataset.json") == read(baseline / "dataset.json")
    new_evaluation, old_evaluation = (
        read(selected / "evaluation.json"),
        read(baseline / "evaluation.json"),
    )
    # Only the explanatory role label was clarified in the later study.
    roles = [x.pop("role") for x in (new_evaluation, old_evaluation)]
    assert new_evaluation == old_evaluation
    endpoints = read(selected / "endpoint-seal.json")["endpoints"]
    before_control = Path(endpoints["strokes"]["root"]) / "configs/source_control.json"
    paired_config(
        settings(before_control, before_control.parents[3]),
        settings(baseline / "configs/control.json", baseline.parent.parent),
        {"model": {"source_conditioning": True}, "train": {"seed": 20261007}},
    )
    common = [
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
    ]
    for name in common:
        shutil.copyfile(selected / name, root / name)
    paths = {}
    for tag, declaration in endpoints.items():
        parent, method = Path(declaration["root"]), declaration["method"]
        destination = root.parent / f"corruption-seed20261006-{tag}"
        if destination.exists():
            raise FileExistsError(destination)
        (destination / "configs").mkdir(parents=True)
        config = relocate(parent / f"configs/{method}.json", parent.parent.parent)
        assert config.train.seed == 20261007 and config.train.steps == 10000
        config = config.model_copy(
            update={"train": config.train.model_copy(update={"seed": 20261006})}
        )
        save_config(config, destination / "config.json")
        save_config(config, destination / f"configs/{method}.json")
        for name in common:
            shutil.copyfile(selected / name, destination / name)
        write(destination / "calibration.json", dict(coefficients={}, no_new_fitting=True))
        write(
            destination / "protocol.json",
            {
                k: original[k]
                for k in (
                    "initial",
                    "initial_files",
                    "selected_panels",
                    "total_schedule_steps",
                    "patch_size",
                )
            }
            | dict(
                methods=[method],
                milestones=[2000],
                independent_fine_tuning_seed=20261006,
                initialization_seed=42,
                prior_ramp_steps=read(parent / "protocol.json")["prior_ramp_steps"],
                selected_alpha=1 / 16,
                stop_step=2000,
                schedule="Unchanged original 10k cosine clock; exactly 2k endpoint updates",
                recipe_parent=str(parent),
                selection_review_sha256=sha(root / "selection-review.json"),
            ),
        )
        sources = discover(config.data)
        train, dev, final = split_experiment(sources, config.data, config.train.seed)
        assert dataset_description(sources, train, dev, final) == read(root / "dataset.json")
        paths[tag] = dict(root=str(destination), method=method, recipe_parent=str(parent))
    replay = root / "baseline-replay"
    (replay / "configs").mkdir(parents=True)
    control = relocate(baseline / "configs/control.json", baseline.parent.parent)
    assert control.train.seed == 20261006 and not control.model.source_conditioning
    save_config(control, replay / "config.json")
    save_config(control, replay / "configs/control.json")
    write(replay / "protocol.json", old_protocol | dict(methods=["control"]))
    write(replay / "calibration.json", dict(coefficients={}))
    baseline_files = {}
    for mode in ("ema", "raw"):
        folder = baseline / f"snapshots/step-2000/control/{mode}"
        baseline_files[mode] = dict(
            path=str(folder),
            files={n: sha(folder / n) for n in ("weights.npz", "config.json", "metadata.json")},
        )
    write(
        root / "plan.json",
        dict(
            selected_root=str(selected),
            selected_review_sha256=sha(selected / "review.json"),
            alpha=1 / 16,
            seed=20261006,
            endpoint_steps=2000,
            schedule_steps=10000,
            endpoint_roots=paths,
            baseline_root=str(baseline),
            baseline_modes=baseline_files,
            baseline_source_sha256=sha(baseline / "source.json"),
            baseline_archive_sha256=baseline_source["archive_sha256"],
            baseline_preflight_sha256=sha(baseline / "preflight/results.json"),
            evaluation_role_labels=roles,
            independent="Fine-tuning seed only; the common 100k initialization is unchanged",
            required=(
                "Fixed alpha, EMA primary, all unchanged strict final criteria, "
                "actual visual/raw/repeat/blank review"
            ),
            fresh="Generate no fresh synthetic conditions until this seed comparison passes",
        ),
    )
    print(dict(root=str(root), endpoints=paths), flush=True)


def audit(root):
    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    from lineartgen.config import load_config
    from lineartgen.models.serialization import load_inference
    from lineartgen.training import lineart_state
    from lineartgen.training.lineart_study import create_arm, study_methods

    plan = read(root / "plan.json")
    baseline = Path(plan["baseline_root"])
    historical_source(baseline)
    protocol = read(baseline / "protocol.json")
    initial, _ = load_inference(protocol["initial"])

    def restore(directory):
        p = read(directory / "protocol.json")
        arms = {
            n: create_arm(
                load_config(directory / f"configs/{n}.json"),
                initial,
                adversarial=n == "patch_prior",
                patch_size=p["patch_size"],
            )
            for n in study_methods(p)
        }
        with ocp.CheckpointManager(
            (directory / "preflight/checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as saved:
            assert lineart_state.restore(saved, arms, 6) == 6
        return arms

    old, new = restore(baseline), restore(root / "baseline-replay")
    jax.tree.map(
        np.testing.assert_array_equal,
        lineart_state.payload({"control": old["control"]}, 6),
        lineart_state.payload(new, 6),
    )
    write(
        root / "baseline-replay-audit.json",
        dict(
            full_model_optimizer_ema_exact=True,
            method="control",
            steps=6,
            baseline_root=str(baseline),
            historical_archive_valid=True,
            baseline_source_sha256=sha(baseline / "source.json"),
            replay_preflight_sha256=sha(root / "baseline-replay/preflight/results.json"),
            baseline_preflight_sha256=sha(baseline / "preflight/results.json"),
            script_sha256=sha(Path(__file__)),
        ),
    )
    print("Historical seed20261006 control model/optimizer/EMA replay exact", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "audit"])
    for n in ["root", "selected", "baseline"]:
        p.add_argument("--" + n, type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root.resolve(), a.selected.resolve(), a.baseline.resolve())
    else:
        audit(a.root.resolve())
