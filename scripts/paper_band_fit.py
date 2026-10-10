"""Matched fixed-pair near-line paper-band loss diagnostic; never an adoption candidate."""

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
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import create_arm
from lineartgen.training.losses import image_loss

WORKTREES = Path("/home/whitegreen/projects/lineartgen/.worktrees")
TRAIN = (999, 1095, 1287, 1383)
PROBE = (807, 903, 1191, 1479)
METHODS = ("gap", "band")
STEPS = 1200
SEED = 2026101066


def prepare(root):
    import shutil

    assert not root.exists()
    parent = WORKTREES / "jax-gap-clearance-fit-20261010/runs/gap-clearance-fit-20261010"
    verify(parent)
    final = read(parent / "final-audit.json")
    assert final["stage"] == "complete_fitting_diagnostic" and final["final_images_reviewed"] == 2
    result = read(parent / "evaluation/step-1200/metrics.json")
    assert not result["fit_passed"]["gap"]
    assert any(
        not c["passed"]
        and c["method"] == "gap"
        and c["case"] in (1287, 1383)
        and c["metric"] in ("halo_opacity_excess", "connection_bridge_rate")
        for c in result["checks"]
    )
    plan = read(parent / "plan.json")
    config = load_config(parent / "configs/gap.json")
    config = config.model_copy(
        update={"loss": config.loss.model_copy(update={"paper_band_weight": 0.1})}
    )
    (root / "configs").mkdir(parents=True)
    save_config(config, root / "configs/band.json")
    for name in ("inputs.npz", "train.json", "probe.json", "target-oracle.json"):
        shutil.copyfile(parent / name, root / name)
    plan.update(
        arms=["band"],
        control_parent=str(parent),
        gate=str(parent),
        gate_source_sha256=sha(parent / "source.json"),
        plan_document_sha256=sha(Path("docs/paper-band-fit-plan-20261010.md")),
        control_files={
            str(p): sha(p)
            for step in (0, 100, 400, STEPS)
            for p in (parent / f"snapshots/step-{step}/gap/raw").iterdir()
            if p.is_file()
        },
    )
    write(root / "plan.json", plan)


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert sha(root / "inputs.npz") == plan["inputs_sha256"]
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    assert sha(Path(plan["gate"]) / "source.json") == plan["gate_source_sha256"]
    for path, digest in plan["control_files"].items():
        assert sha(Path(path)) == digest
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
    config = load_config(root / "configs/band.json")
    arms = {"band": create_arm(config, initial)}
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
    parent = Path(read(root / "plan.json")["control_parent"])
    control = load_inference(parent / f"snapshots/step-{step}/gap/raw")[0]
    arms = dict(gap=dict(model=control), **arms)
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
    expected_rows = [
        x
        for x in read(parent / f"evaluation/step-{step}/metrics.json")["rows"]
        if x["method"] == "gap"
    ]
    assert [x for x in rows if x["method"] == "gap"] == expected_rows
    with np.load(parent / f"evaluation/step-{step}/predictions.npz") as expected:
        for index in TRAIN + PROBE:
            np.testing.assert_array_equal(images[f"gap-{index}"], expected[f"gap-{index}"])
    write(out / "control-audit.json", dict(all_exact=True, cases=8))
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
                    initial[f"gap-{index}"],
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
    assert gate["stage"] == "complete_fitting_diagnostic" and gate["final_images_reviewed"] == 2
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
            initial, _ = load_inference(plan["initial"])
            for d, p in prepared.items():
                a = np.asarray(forward(initial, p))
                b = np.asarray(forward(arms["band"]["model"], p))
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
            stats = {m: update(arms[m], prepared[depth]) for m in arms}
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
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=6, bindings=bindings))
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
