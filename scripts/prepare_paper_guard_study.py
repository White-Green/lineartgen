"""Freeze no-contrast surrogate-gradient ablations and a training-calibrated paper term."""

import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from flax import nnx

from lineartgen.config import load_config, save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import recursive_forward, training_inputs
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference
from lineartgen.training.lineart import make_profiles
from lineartgen.training.lineart_adversarial import patch_coordinates
from lineartgen.training.lineart_steps import accumulate_generator, make_lineart_steps
from lineartgen.training.paper import paper_peak_loss


@nnx.jit
def paper_gradient(model, prepared, profiles, coordinates, unused_critic, unused_weight):
    def objective(generator):
        prediction = recursive_forward(generator, prepared.noisy, prepared.noise_levels)[0]
        value = paper_peak_loss(
            prediction, prepared.targets[0], prepared.noise_levels[0] > 0.001, tile=8
        )
        return value, dict(paper_peak=value)

    (_, metrics), gradient = nnx.value_and_grad(objective, has_aux=True)(model)
    return metrics, gradient


@jax.jit
def tree_dot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


def relocate(path, worktree):
    config = load_config(path)
    changes = {
        name: (worktree / getattr(config.data, name)).resolve()
        for name in ("root", "prepared_root")
        if getattr(config.data, name) is not None
    }
    return config.model_copy(update={"data": config.data.model_copy(update=changes)})


def prepare(root, hard, ste, continuation):
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
        shutil.copyfile(hard / name, root / name)
    configs = {}
    for name, counterpart in [
        ("ste_no_contrast", "no_contrast"),
        ("ste_neighbor_guard", "neighbor_guard"),
    ]:
        config = relocate(hard / f"configs/{counterpart}.json", hard.parent.parent)
        configs[name] = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"output_clip_gradient": "straight_through"}
                )
            }
        )
    base = relocate(ste / "configs/ste_ink_guard.json", ste.parent.parent)
    assert base.diffusion.min_noise == 0.001 and base.diffusion.scale_loss_multiplier == 0
    assert base.model.output_clip_gradient == "straight_through"
    assert base.loss.contrast_weight == 0 and base.loss.paper_peak_weight == 0
    save_config(base, root / "config.json")
    original = read(hard / "protocol.json")
    externals = {}
    for step, parents in [
        (
            2000,
            [
                (hard, ["control", "no_contrast", "ink_guard", "neighbor_guard"]),
                (ste, ["ste_control", "ste_ink_guard"]),
            ],
        ),
        (10000, [(continuation, ["control", "ink_guard", "neighbor_guard"])]),
    ]:
        externals[str(step)] = [
            dict(root=str(p), methods=names, source_sha256=sha(p / "source.json"))
            for p, names in parents
        ]
    counterparts = {
        "ste_no_contrast": dict(
            root=str(hard),
            method="no_contrast",
            changes={"model": {"output_clip_gradient": "straight_through"}},
        ),
        "ste_neighbor_guard": dict(
            root=str(hard),
            method="neighbor_guard",
            changes={"model": {"output_clip_gradient": "straight_through"}},
        ),
        "ste_ink_paper": dict(root=str(ste), method="ste_ink_guard", changes={}),
    }
    # Written before calibration/training; no development quality is used for weights.
    write(
        root / "plan.json",
        dict(
            methods=list(counterparts),
            tile=8,
            teacher_paper_max_opacity=0.02,
            calibration_seed=20262006,
            calibration_batches=12,
            depths=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            paper_median_parameter_gradient_ratio=0.1,
            paper_max_parameter_gradient_ratio=0.5,
            selection="Unchanged 2k screen and visual review; at most two passing arms ranked by "
            "sum of primary halo gains. Keep full states for 10k. Never extend failed arms.",
            hypothesis="Remove nonstationary contrast under identity clip gradients; "
            "compare existing "
            "neighbor losses and worst local teacher-paper excess opacity against STE ink.",
            constraints="No changed inference, data, recursive stage loss, optimizer, "
            "or acceptance gate.",
        ),
    )
    training, dev, final = split_experiment(discover(base.data), base.data, base.train.seed)
    assert dataset_description(discover(base.data), training, dev, final) == read(
        root / "dataset.json"
    )
    initial, _ = load_inference(original["initial"])
    model = create_model(base.model, rngs=nnx.Rngs(42))
    nnx.update(model, nnx.state(initial, nnx.Param))
    before = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(model))
    gradient = make_lineart_steps(base)[0]
    prepare_inputs = jax.jit(
        lambda x, k, n: training_inputs(
            x, k, base.diffusion, base.matching, levels=n, sketch=base.sketch
        ),
        static_argnums=2,
    )
    seed = 20262006
    stream = batches(
        training, base.data, batch_size=base.train.effective_batch_size, count=12, seed=seed
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
                    patch_coordinates(clean, seed=seed, step=step, size=64),
                )
            )
            levels = step // 2 + 1
            prepared = prepare_inputs(
                jax.device_put(clean), jax.random.fold_in(jax.random.PRNGKey(seed), step), levels
            )
            metrics, bg, norm = accumulate_generator(
                gradient, model, prepared, profiles, coordinates, None, 0, base.train
            )
            aux, pg, pnorm = accumulate_generator(
                paper_gradient, model, prepared, profiles, coordinates, None, 0, base.train
            )
            norm, pnorm = float(norm), float(pnorm)
            assert np.isfinite(norm) and np.isfinite(pnorm)
            row = dict(
                batch=step,
                levels=levels,
                base_loss=float(metrics["loss"]),
                base_norm=norm,
                paper_loss=float(aux["paper_peak"]),
                paper_norm=pnorm,
                ratio=pnorm / max(norm, 1e-12),
                cosine=float(tree_dot(bg, pg)) / max(norm * pnorm, 1e-12),
            )
            rows.append(row)
            print(row, flush=True)
    finally:
        stream.close()
    jax.tree.map(np.testing.assert_array_equal, before, nnx.state(model))
    ratios = [x["ratio"] for x in rows if x["paper_norm"] > 1e-12]
    assert ratios
    coefficient = float(min(0.1 / np.median(ratios), 0.5 / max(ratios)))
    configs["ste_ink_paper"] = base.model_copy(
        update={
            "loss": base.loss.model_copy(
                update={"paper_peak_weight": coefficient, "paper_peak_tile": 8}
            )
        }
    )
    counterparts["ste_ink_paper"]["changes"] = {"loss": {"paper_peak_weight": coefficient}}
    write(
        root / "calibration.json",
        dict(
            coefficients={"paper_peak": coefficient},
            batches=rows,
            calibration_seed=seed,
            generator_parameter_gradients=True,
            no_model_updates=True,
            plan_sha256=sha(root / "plan.json"),
        ),
    )
    protocol = original | dict(
        methods=list(configs),
        milestones=[2000],
        external_baselines=externals,
        config_counterparts=counterparts,
        plan_sha256=sha(root / "plan.json"),
        calibration="12 accumulated parameter gradients on training batches; "
        "paper median .1/max .5",
        optimizer="Reset optimizer/EMA from common 100k EMA; unchanged 10k cosine clock",
        next_phase="Apply inherited 2k screen and visual review before full-state 10k continuation",
    )
    write(root / "protocol.json", protocol)
    (root / "configs").mkdir()
    for name, config in configs.items():
        save_config(config, root / f"configs/{name}.json")
    print(dict(paper_coefficient=coefficient, methods=list(configs)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "hard", "ste", "continuation"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.root.resolve(), args.hard.resolve(), args.ste.resolve(), args.continuation.resolve()
    )
