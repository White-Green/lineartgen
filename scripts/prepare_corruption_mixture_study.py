"""Pair an equal corruption-family mixture with completed original-source guide runs."""

import argparse
import shutil
from functools import partial
from pathlib import Path

from campaign_protocol import read, sha, write
from prepare_source_guide_study import relocate

from lineartgen.config import save_config
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment


def prepare(root, guide, guide_continuation, hard, hard_continuation):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in (
        "dataset.json",
        "evaluation.json",
        "acceptance.json",
        "supplemental-panels.json",
        "confirmation-plan.json",
    ):
        shutil.copyfile(guide / name, root / name)
    names = ["source_control", "source_ink_guard"]
    configs, counterparts = {}, {}
    for name in names:
        c = relocate(guide / f"configs/{name}.json", guide.parent.parent)
        assert c.model.source_conditioning and c.sketch.mode == "strokes"
        mixed = "mixed_" + name
        configs[mixed] = c.model_copy(
            update={"sketch": c.sketch.model_copy(update={"mode": "mixed"})}
        )
        counterparts[mixed] = dict(
            root=str(guide), method=name, changes={"sketch": {"mode": "mixed"}}
        )
    plan = dict(
        methods=list(configs),
        intervention=(
            "Per-example equal mixture of tone-only, blur and displaced strokes; "
            "only sketch.mode changes"
        ),
        total_augmentation_probability=0.5,
        expected_input_fractions=dict(clean=0.5, tone=1 / 6, blur=1 / 6, strokes=1 / 6),
        selection=(
            "Inherited EMA 2k screen and native visual review; at most two passing arms, "
            "ranked by summed halo gains"
        ),
        unchanged=(
            "Initial 100k EMA, clean samples, noise keys/masks, corruption parameter ranges, "
            "source guide, losses, recursion, stage weights, sqrt(batch) LR, optimizer, "
            "10k clock and gates"
        ),
        confirmation=(
            "Only final-gate passing candidates proceed to second seed and sealed fresh conditions"
        ),
        changed_training_files=[
            "src/lineartgen/diffusion.py",
            "src/lineartgen/training/steps.py",
            "src/lineartgen/training/lineart_steps.py",
            "src/lineartgen/training/lineart_study.py",
            "scripts/evaluate_lineart_loss_study.py",
            "src/lineartgen/sketch.py",
        ],
        counterpart_changed_source_files=["src/lineartgen/config.py", "src/lineartgen/sketch.py"],
        diagnostic=dict(seed=20262006, batches=12, training_only=True, parameter_fitting=False),
    )
    write(root / "plan.json", plan)

    def declaration(parent, methods):
        return dict(root=str(parent), methods=methods, source_sha256=sha(parent / "source.json"))

    original = read(guide / "protocol.json")
    protocol = original | dict(
        methods=list(configs),
        milestones=[2000],
        config_counterparts=counterparts,
        external_baselines={
            "2000": [declaration(hard, ["control", "ink_guard"]), declaration(guide, names)],
            "10000": [
                declaration(hard_continuation, ["control", "ink_guard"]),
                declaration(guide_continuation, names),
            ],
        },
        replay_parents=[dict(root=str(guide), methods=names)],
        plan_sha256=sha(root / "plan.json"),
        calibration="No new coefficients; fixed per-example corruption family mixture",
        optimizer="Reset optimizer/EMA from common 100k EMA; unchanged 10k cosine clock",
        next_phase="Unchanged 2k screen then eligible full-state 10k continuation",
    )
    write(root / "protocol.json", protocol)
    (root / "configs").mkdir()
    for name, c in configs.items():
        save_config(c, root / f"configs/{name}.json")
    base = configs["mixed_source_control"]
    save_config(base, root / "config.json")
    sources = discover(base.data)
    train, dev, final = split_experiment(sources, base.data, base.train.seed)
    assert dataset_description(sources, train, dev, final) == read(root / "dataset.json")
    write(root / "calibration.json", dict(coefficients={}, no_parameter_fitting=True))
    replay = root / "baseline-replay"
    (replay / "configs").mkdir(parents=True)
    for name in names:
        save_config(
            relocate(guide / f"configs/{name}.json", guide.parent.parent),
            replay / f"configs/{name}.json",
        )
    shutil.copyfile(replay / "configs/source_control.json", replay / "config.json")
    write(replay / "protocol.json", original | dict(methods=names))
    write(replay / "calibration.json", dict(coefficients={}))
    print(dict(root=str(root), methods=list(configs)))


def diagnose(root):
    """Audit family/enablement draws on training batches without model updates."""
    import jax
    import numpy as np

    from lineartgen.config import load_config
    from lineartgen.sketch import augment_sketch

    c = load_config(root / "config.json")
    plan = read(root / "plan.json")
    seed = plan["diagnostic"]["seed"]
    training, _, _ = split_experiment(discover(c.data), c.data, c.train.seed)
    stream = batches(training, c.data, batch_size=c.train.effective_batch_size, count=12, seed=seed)
    functions = {
        mode: jax.jit(partial(augment_sketch, config=c.sketch.model_copy(update={"mode": mode})))
        for mode in ("mixed", "tone", "blur", "strokes")
    }
    rows, counts = [], dict(clean=0, tone=0, blur=0, strokes=0)
    try:
        for step, batch in enumerate(stream):
            clean = jax.device_put(batch.images)
            key = jax.random.fold_in(jax.random.PRNGKey(seed), step)
            enabled = np.asarray(
                jax.random.bernoulli(
                    jax.random.fold_in(key, 0), c.sketch.probability, (len(clean),)
                )
            )
            families = np.asarray(
                jax.random.randint(jax.random.fold_in(key, 4), (len(clean),), 0, 3)
            )
            outputs = {name: np.asarray(fn(clean, key)) for name, fn in functions.items()}
            assert all(np.isfinite(x).all() for x in outputs.values())
            for i, family in enumerate(families):
                mode = ("tone", "blur", "strokes")[family]
                np.testing.assert_allclose(outputs["mixed"][i], outputs[mode][i], rtol=0, atol=3e-7)
                if not enabled[i]:
                    np.testing.assert_array_equal(outputs["mixed"][i], np.asarray(clean)[i])
                counts[mode if enabled[i] else "clean"] += 1
            rows.append(dict(batch=step, family=families.tolist(), enabled=enabled.tolist()))
    finally:
        stream.close()
    write(
        root / "mixture-diagnostic.json",
        dict(
            counts=counts,
            rows=rows,
            seed=seed,
            no_model_updates=True,
            legacy_family_equivalence_atol=3e-7,
            script_sha256=sha(Path(__file__)),
        ),
    )
    print(counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "diagnose"])
    p.add_argument("--root", type=Path, required=True)
    for n in ["guide", "guide-continuation", "hard", "hard-continuation"]:
        p.add_argument("--" + n, type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        paths = [a.guide, a.guide_continuation, a.hard, a.hard_continuation]
        if any(x is None for x in paths):
            p.error("prepare requires all four baseline paths")
        prepare(a.root.resolve(), *(x.resolve() for x in paths))
    else:
        diagnose(a.root.resolve())
