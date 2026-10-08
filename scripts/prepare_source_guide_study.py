"""Freeze source-conditioning comparisons without changing their paired losses or data."""

import argparse
import shutil
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import load_config, save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import training_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart import make_profiles
from lineartgen.training.lineart_adversarial import patch_coordinates
from lineartgen.training.lineart_steps import accumulate_generator, make_lineart_steps
from lineartgen.training.lineart_study import assert_initial_parameters


def relocate(path, worktree):
    c = load_config(path)
    paths = {
        k: (worktree / getattr(c.data, k)).resolve()
        for k in ["root", "prepared_root"]
        if getattr(c.data, k) is not None
    }
    return c.model_copy(update={"data": c.data.model_copy(update=paths)})


def prepare(root, hard, continuation, interval, hard_replay):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for n in [
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
    ]:
        shutil.copyfile(hard / n, root / n)
    counterparts = {}
    configs = {}
    for parent, n in [(hard, "control"), (hard, "ink_guard"), (interval, "interval_ink_guard")]:
        c = relocate(parent / f"configs/{n}.json", parent.parent.parent)
        assert not c.model.source_conditioning and c.model.output_clip_gradient == "hard"
        name = f"source_{n}"
        configs[name] = c.model_copy(
            update={"model": c.model.model_copy(update={"source_conditioning": True})}
        )
        counterparts[name] = dict(
            root=str(parent), method=n, changes={"model": {"source_conditioning": True}}
        )
    plan = dict(
        methods=list(configs),
        guide="Original rough source after sketch augmentation, before noise; never clean target",
        injection="Zero 3x3 projections before the first encoder/head ReLU; 63 added parameters",
        parameter_count=126828,
        clipped_recursive_feedback=True,
        diagnostic=dict(seed=20262006, batches=12, depths=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
        selection=(
            "Inherited EMA 2k screen plus native visual review; "
            "up to two passing arms ranked by summed halo gains"
        ),
        unchanged=(
            "Data, Gaussian draws, masking, old weights, losses, recursion, "
            "stage weights, optimizer, sqrt(batch) LR and quality gates"
        ),
        confirmation=(
            "Final-gate passing candidates receive second-seed and sealed fresh evaluation"
        ),
        changed_training_files=[
            "src/lineartgen/diffusion.py",
            "src/lineartgen/training/steps.py",
            "src/lineartgen/training/lineart_steps.py",
            "src/lineartgen/training/lineart_study.py",
            "scripts/evaluate_lineart_loss_study.py",
        ],
    )
    write(root / "plan.json", plan)

    def declaration(parent, names):
        return dict(root=str(parent), methods=names, source_sha256=sha(parent / "source.json"))

    external = {
        "2000": [
            declaration(hard, ["control", "ink_guard"]),
            declaration(interval, ["interval_ink_guard"]),
        ],
        "10000": [declaration(continuation, ["control", "ink_guard"])],
    }
    replays = [
        dict(root=str(hard_replay), methods=["control", "ink_guard"]),
        dict(root=str(interval), methods=["interval_control", "interval_ink_guard"]),
    ]
    original = read(hard / "protocol.json")
    write(
        root / "protocol.json",
        original
        | dict(
            methods=list(configs),
            milestones=[2000],
            external_baselines=external,
            config_counterparts=counterparts,
            replay_parents=replays,
            plan_sha256=sha(root / "plan.json"),
            calibration="No coefficient fitting; training-only parameter-gradient diagnostic",
            optimizer="Reset optimizer/EMA from common 100k EMA; unchanged 10k cosine clock",
            next_phase="Unchanged 2k screen then eligible full-state 10k continuation",
        ),
    )
    (root / "configs").mkdir()
    for name, c in configs.items():
        save_config(c, root / f"configs/{name}.json")
    base = configs["source_control"]
    save_config(base, root / "config.json")
    train, dev, final = split_experiment(discover(base.data), base.data, base.train.seed)
    assert dataset_description(discover(base.data), train, dev, final) == read(
        root / "dataset.json"
    )
    replay = root / "baseline-replay"
    (replay / "configs").mkdir(parents=True)
    for parent in replays:
        directory = Path(parent["root"])
        worktree = directory.parents[2] if directory == hard_replay else directory.parent.parent
        for name in parent["methods"]:
            save_config(
                relocate(directory / f"configs/{name}.json", worktree),
                replay / f"configs/{name}.json",
            )
    shutil.copyfile(replay / "configs/control.json", replay / "config.json")
    write(
        replay / "protocol.json",
        original | dict(methods=[n for d in replays for n in d["methods"]]),
    )
    write(replay / "calibration.json", dict(coefficients={}))
    print(dict(root=str(root), methods=list(configs)), flush=True)


def diagnose(root):
    if (root / "calibration.json").exists():
        raise FileExistsError(root / "calibration.json")
    protocol = read(root / "protocol.json")
    plan = read(root / "plan.json")
    base = load_config(root / "config.json")
    assert sha(root / "plan.json") == protocol["plan_sha256"]
    initial, _ = load_inference(protocol["initial"])
    guided = create_model(base.model, rngs=nnx.Rngs(base.train.initialization_seed))
    nnx.update(guided, nnx.state(initial, nnx.Param))
    assert_initial_parameters(initial, guided)
    before = [jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(m)) for m in [initial, guided]]
    configs = {}
    gradients = {}
    models = {}
    for name in protocol["methods"]:
        c = load_config(root / f"configs/{name}.json")
        counterpart = name.removeprefix("source_")
        configs[name] = c
        models[name] = guided
        configs[counterpart] = c.model_copy(
            update={"model": c.model.model_copy(update={"source_conditioning": False})}
        )
        models[counterpart] = initial
    for name, c in configs.items():
        gradients[name] = make_lineart_steps(c)[0]
    training, _, _ = split_experiment(discover(base.data), base.data, base.train.seed)
    seed = plan["diagnostic"]["seed"]
    stream = batches(
        training, base.data, batch_size=base.train.effective_batch_size, count=12, seed=seed
    )
    prepare_inputs = jax.jit(
        lambda x, k, n: training_inputs(
            x, k, base.diffusion, base.matching, levels=n, sketch=base.sketch, with_source=True
        ),
        static_argnums=2,
    )
    rows = []
    try:
        for step, batch in enumerate(stream):
            clean, geometry = mix_geometry(
                batch.images,
                base.geometry,
                seed=seed,
                step=step,
                image_scale=base.data.image_scale,
                include_paths=True,
            )
            profiles, coordinates = jax.device_put(
                (
                    make_profiles(geometry, len(clean), len(clean[0]), seed=seed, step=step),
                    patch_coordinates(clean, seed=seed, step=step, size=protocol["patch_size"]),
                )
            )
            levels = plan["diagnostic"]["depths"][step]
            prepared = prepare_inputs(
                jax.device_put(clean), jax.random.fold_in(jax.random.PRNGKey(seed), step), levels
            )
            measures = {}
            for name in configs:
                metrics, _, norm = accumulate_generator(
                    gradients[name],
                    models[name],
                    prepared,
                    profiles,
                    coordinates,
                    None,
                    0,
                    base.train,
                )
                measures[name] = dict(loss=float(metrics["loss"]), norm=float(norm))
                assert all(np.isfinite(v) for v in measures[name].values())
            row = dict(
                batch=step,
                levels=levels,
                measures=measures,
                norm_ratios={
                    n: measures[n]["norm"] / max(measures[n.removeprefix("source_")]["norm"], 1e-12)
                    for n in protocol["methods"]
                },
            )
            rows.append(row)
            print(row, flush=True)
    finally:
        stream.close()
    for original, model in zip(before, [initial, guided], strict=True):
        jax.tree.map(np.testing.assert_array_equal, original, nnx.state(model))
    write(
        root / "calibration.json",
        dict(
            coefficients={},
            diagnostic_only=True,
            no_model_updates=True,
            rows=rows,
            plan_sha256=sha(root / "plan.json"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "diagnose"])
    p.add_argument("--root", type=Path, required=True)
    for n in ["hard", "continuation", "interval", "hard-replay"]:
        p.add_argument("--" + n, type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        if any(getattr(a, n) is None for n in ["hard", "continuation", "interval", "hard_replay"]):
            p.error("prepare requires all baseline paths")
        prepare(
            a.root.resolve(),
            a.hard.resolve(),
            a.continuation.resolve(),
            a.interval.resolve(),
            a.hard_replay.resolve(),
        )
    else:
        diagnose(a.root.resolve())
