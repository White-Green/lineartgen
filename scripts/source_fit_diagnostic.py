"""Fixed-pair fitting diagnostic; its checkpoints are never adoption candidates."""

import argparse
import fcntl
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from lineartgen_runtime.sampling import pyramid
from PIL import Image, ImageDraw

from lineartgen.config import load_config, save_config
from lineartgen.diffusion import recursive_forward_with_preclip
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.evaluation_sampling import runtime_inputs
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference
from lineartgen.models.widen import widen_unet
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm
from lineartgen.training.losses import image_loss

WORKTREES = Path("/home/whitegreen/projects/lineartgen/.worktrees")
TRAIN = (999, 1095, 1287, 1383)
PROBE = (807, 903, 1191, 1479)
METHODS = ("small", "wide")
STEPS = 1200
SEED = 2026101066


def prepare(root):
    assert not root.exists()
    parent = WORKTREES / "jax-source-interval-l1-20261010/runs/source-interval-l1-20261010"
    gate = WORKTREES / (
        "jax-width-corruption-interaction-20261010/runs/width-corruption-interaction-20261010"
    )
    verify(parent)
    initial = parent / "snapshots/step-4000/interval/raw"
    base = load_config(parent / "configs/interval.json")
    assert base.model.feature_input == "source" and base.model.sampling_mode == "one_pass"
    assert base.loss.pixel_loss == "interval_l1" and base.diffusion.scale_loss_multiplier == 0
    base = base.model_copy(
        update={
            "train": base.train.model_copy(
                update=dict(
                    steps=STEPS,
                    learning_rate_decay_steps=None,
                    warmup_steps=0,
                    learning_rate=5e-5,
                    minimum_learning_rate=1e-4,
                    batch_size=4,
                    accumulate_steps=1,
                    seed=SEED,
                )
            )
        }
    )
    assert base.train.peak_learning_rate == 1e-4
    wide = base.model_copy(
        update={
            "model": base.model.model_copy(
                update=dict(
                    channels=tuple(x * 2 for x in base.model.channels),
                    detail_channels=base.model.detail_channels * 2,
                    final_decoder_channels=base.model.final_decoder_channels * 2,
                    head_hidden_channels=2
                    * (base.model.head_hidden_channels or (1 + base.model.final_decoder_channels)),
                )
            )
        }
    )
    (root / "configs").mkdir(parents=True)
    for name, config in zip(METHODS, (base, wide), strict=True):
        save_config(config, root / f"configs/{name}.json")
    reference = read(parent / "reference.json")
    cases = read(Path(reference["root"]) / "cases.json")["cases"]
    arrays, bound, records = {}, {}, []
    for split, ids in [("train", TRAIN), ("probe", PROBE)]:
        for index in ids:
            case = cases[index]
            assert case["kind"] != "real" and case["family"] == "blur"
            assert case["strength"] == 0.5 and not case["partial"]
            path = Path(reference["root"]) / case["file"]
            assert sha(path) == reference["files"][case["file"]]
            bound[str(path)] = dict(path=str(path), sha256=sha(path))
            with np.load(path) as data:
                for key in data.files:
                    arrays[f"{index}-{key}"] = data[key]
                assert np.all(data["strength"] == 0.5)
            records.append(dict(case=index, split=split, specification=case))
    np.savez_compressed(root / "inputs.npz", **arrays)
    target_oracle(root)
    for split, ids in [("train", TRAIN), ("probe", PROBE)]:
        write(root / f"{split}.json", dict(cases=list(ids)))
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir() if p.is_file()},
            data=bound,
            arms=list(METHODS),
            cases=records,
            steps=STEPS,
            seed=SEED,
            inputs_sha256=sha(root / "inputs.npz"),
            target_oracle_sha256=sha(root / "target-oracle.json"),
            gate=str(gate),
            gate_source_sha256=sha(gate / "source.json"),
            parent=str(parent),
            reference=reference,
            crops={str(i): read(parent / "plan.json")["crops"][str(i)] for i in TRAIN + PROBE},
            plan_document_sha256=sha(Path("docs/source-fit-diagnostic-plan-20261010.md")),
            diagnostic_only=True,
            adoption=False,
            train_on_development_geometry=True,
            confirmation=False,
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(root / "inputs.npz") == plan["inputs_sha256"]
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    assert sha(Path(plan["gate"]) / "source.json") == plan["gate_source_sha256"]
    return plan


def prepared_inputs(clean, source, strength, depth):
    noisy, levels = runtime_inputs(source, strength, SEED + depth, depth)
    return jax.device_put(
        dict(
            noisy=noisy,
            levels=levels,
            guides=tuple(pyramid(source, depth)),
            target=clean,
        )
    )


def batch(root, ids, depth):
    with np.load(root / "inputs.npz") as data:
        clean = np.stack([data[f"{i}-image"] for i in ids])
        source = np.stack([data[f"{i}-source"] for i in ids])
        strength = np.stack([data[f"{i}-strength"] for i in ids])
    return prepared_inputs(clean, source, strength, depth)


def objective(model, prepared, config):
    predictions, physical = recursive_forward_with_preclip(
        model, prepared["noisy"], prepared["levels"], guides=prepared["guides"]
    )
    loss, parts = image_loss(
        predictions[0],
        prepared["target"],
        prepared["levels"][0],
        config.loss,
        config.diffusion.min_noise,
        preclip=physical[0],
    )
    return loss, dict(parts, loss=loss)


def make_arms(root):
    initial, _ = load_inference(read(root / "plan.json")["initial"])
    configs = {m: load_config(root / f"configs/{m}.json") for m in METHODS}
    wide = widen_unet(initial, configs["small"].model, configs["wide"].model, seed=SEED)
    arms = {
        m: create_arm(configs[m], model) for m, model in zip(METHODS, (initial, wide), strict=True)
    }
    for arm in arms.values():
        config = arm["config"]

        def make_gradient(config):
            @nnx.jit
            def gradient(model, prepared):
                (_, metrics), grads = nnx.value_and_grad(
                    lambda m: objective(m, prepared, config), has_aux=True
                )(model)
                return metrics, grads

            return gradient

        arm["gradient"] = make_gradient(config)
    return arms


@nnx.jit
def forward(model, prepared):
    return recursive_forward_with_preclip(
        model, prepared["noisy"], prepared["levels"], guides=prepared["guides"]
    )[0][0]


def fit_checks(rows):
    limits = dict(
        ink_f1_alpha50=(0.98, False),
        ink_deficit_fraction=(0.02, True),
        halo_opacity_excess=(0.003, True),
        profile_width_mae_px=(0.2, True),
        connection_cut_rate=(0.0, True),
        connection_bridge_rate=(0.0, True),
    )
    checks = []
    for row in rows:
        if row["split"] != "train":
            continue
        for metric, (threshold, lower) in limits.items():
            if row.get(metric) is not None:
                value = row[metric]
                checks.append(
                    dict(
                        method=row["method"],
                        case=row["case"],
                        metric=metric,
                        value=value,
                        threshold=threshold,
                        passed=value <= threshold if lower else value >= threshold,
                    )
                )
    return checks


def target_oracle(root):
    rows = []
    with np.load(root / "inputs.npz") as data:
        for split, ids in [("train", TRAIN), ("probe", PROBE)]:
            for index in ids:
                target, strength = data[f"{index}-image"], data[f"{index}-strength"]
                metrics = protected_metrics(target, target, strength)
                metrics |= halo_metrics(target, target, strength)
                metrics |= probe_metrics(
                    target, target, data[f"{index}-probes"], data[f"{index}-connections"]
                )
                rows.append(dict(method="teacher_identity", split=split, case=index, **metrics))
    checks = fit_checks(rows)
    assert all(c["passed"] for c in checks)
    write(root / "target-oracle.json", dict(rows=rows, checks=checks))


def evaluate(root, arms, step):
    out = root / f"evaluation/step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    rows, images = [], {}
    with np.load(root / "inputs.npz") as data:
        for split, ids in [("train", TRAIN), ("probe", PROBE)]:
            prepared = batch(root, ids, 3)
            for method, arm in arms.items():
                predictions = np.asarray(forward(arm["model"], prepared))
                assert np.isfinite(predictions).all()
                for index, prediction in zip(ids, predictions, strict=True):
                    target, strength = data[f"{index}-image"], data[f"{index}-strength"]
                    metrics = protected_metrics(prediction, target, strength)
                    metrics |= halo_metrics(prediction, target, strength)
                    metrics |= probe_metrics(
                        prediction, target, data[f"{index}-probes"], data[f"{index}-connections"]
                    )
                    assert all(v is None or np.isfinite(v) for v in metrics.values())
                    rows.append(dict(method=method, case=index, split=split, **metrics))
                    images[f"{method}-{index}"] = prediction
    np.savez_compressed(out / "predictions.npz", **images)
    checks = fit_checks(rows)
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            fit_passed={m: all(c["passed"] for c in checks if c["method"] == m) for m in METHODS},
            adoption=False,
        ),
    )
    render(root, step)


def render(root, step):
    plan = read(root / "plan.json")
    out = root / f"evaluation/step-{step}"
    columns = ("target", "source", "initial", *METHODS)
    with (
        np.load(root / "inputs.npz") as data,
        np.load(root / "evaluation/step-0/predictions.npz") as initial,
        np.load(out / "predictions.npz") as predictions,
    ):
        for split, ids in [("train", TRAIN), ("probe", PROBE)]:
            canvas = Image.new("L", (160 * len(columns), 184 * len(ids)), 255)
            for row, index in enumerate(ids):
                arrays = [
                    data[f"{index}-image"],
                    data[f"{index}-source"],
                    initial[f"small-{index}"],
                    *[predictions[f"{m}-{index}"] for m in METHODS],
                ]
                for col, (name, array) in enumerate(zip(columns, arrays, strict=True)):
                    pixels = np.rint(np.clip((array[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                    canvas.paste(
                        Image.fromarray(pixels).crop(plan["crops"][str(index)]),
                        (160 * col, 184 * row + 24),
                    )
                    ImageDraw.Draw(canvas).text(
                        (160 * col + 2, 184 * row + 4), f"{index} {name}", fill=0
                    )
            canvas.save(out / f"{split}-native.png")


def train(root):
    plan = check(root)
    gate = read(Path(plan["gate"]) / "final-audit.json")
    assert (
        gate["stage"] == "complete_diagnostic_insufficient" and gate["final_images_reviewed"] == 14
    )
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    prepared = {d: batch(root, TRAIN, d) for d in range(1, 7)}
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=4) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        if start == 0:
            errors = []
            for d, p in prepared.items():
                a, b = [np.asarray(forward(arm["model"], p)) for arm in arms.values()]
                error = float(np.max(np.abs(a - b)))
                assert error <= 1e-5
                errors.append(dict(depth=d, maximum_error=error))
            write(
                root / "initial-audit.json",
                dict(
                    all_depths_within_1e_minus_5=True,
                    errors=errors,
                    parameters={
                        m: sum(x.size for x in jax.tree.leaves(nnx.state(a["model"], nnx.Param)))
                        for m, a in arms.items()
                    },
                ),
            )
            snapshot(root, arms, 0)
            evaluate(root, arms, 0)
        for step in range(start, STEPS):
            depth = step % 6 + 1
            stats = {m: update(arms[m], prepared[depth]) for m in METHODS}
            complete = step + 1
            if complete % 25 == 0 or complete <= 6:
                write(out / "progress.json", dict(step=complete, depth=depth, metrics=stats))
                print(complete, depth, stats, flush=True)
            if complete in (6, 100, 400, STEPS):
                lineart_state.save(saved, arms, complete)
                restored = make_arms(root)
                assert lineart_state.restore(saved, restored, complete) == complete
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, complete),
                    lineart_state.payload(restored, complete),
                )
                write(out / f"restore-audit-{complete}.json", dict(all_state_exact=True))
                if complete != 6:
                    snapshot(root, arms, complete)
                    evaluate(root, arms, complete)
        bindings = {}
        for milestone in (100, 400, STEPS):
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, milestone) == milestone
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.clone(arm["model"])
                    if mode == "ema":
                        nnx.update(expected, arm["ema"])
                    folder = root / f"snapshots/step-{milestone}/{name}/{mode}"
                    actual, config = load_inference(folder)
                    assert config == arm["config"]
                    jax.tree.map(
                        np.testing.assert_array_equal,
                        nnx.state(expected, nnx.Param),
                        nnx.state(actual, nnx.Param),
                    )
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=12, bindings=bindings))
        write(
            root / "workflow-status.json",
            dict(stage="numerical_complete_pending_review", step=STEPS, adoption=False),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "train"])
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="train1200"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    else:
        prepare(root)
