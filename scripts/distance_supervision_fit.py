"""Coverage-distance supervision of the established one-pass source model."""

import argparse
import fcntl
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from edge_energy_fit import forward, prepared_inputs
from flax import nnx
from paper_band_fit import PROBE, TRAIN, fit_checks, target_oracle
from PIL import Image, ImageDraw

from lineartgen.config import load_config, save_config
from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.diffusion import recursive_forward_with_preclip
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.coverage_distance import coverage_distance
from lineartgen.training.distance_supervision import distance_supervision
from lineartgen.training.lineart_study import assert_initial_parameters, create_arm
from lineartgen.training.losses import image_loss

METHODS = ("control", "distance")
TRAINED = ("distance",)
STEPS = 1200
SEED = 2026101066
SAVED = (0, 6, 400, STEPS)


def arrays(root, ids):
    with np.load(root / "inputs.npz") as data:
        return tuple(
            jnp.asarray(np.stack([data[f"{i}-{field}"] for i in ids]))
            for field in ("image", "source", "strength")
        )


def audit_data(root):
    rows = []
    with np.load(root / "inputs.npz") as data:
        for i in TRAIN + PROBE:
            a, b, t = [data[f"{i}-{field}"] for field in ("image", "source", "strength")]
            assert a.shape == b.shape == t.shape == (512, 512, 1)
            assert np.isfinite(a).all() and np.isfinite(b).all() and np.all(t == 0.5)
            rows.append(
                dict(
                    case=i,
                    split="train" if i in TRAIN else "probe",
                    antialias_pixels=int(np.sum((a > -1) & (a < 1))),
                    minimum=float(a.min()),
                    maximum=float(a.max()),
                )
            )
    assert set(TRAIN).isdisjoint(PROBE)
    assert all(r["antialias_pixels"] > 0 for r in rows)
    write(
        root / "data-audit.json",
        dict(
            train=list(TRAIN),
            probe=list(PROBE),
            disjoint=True,
            inputs_sha256=sha(root / "inputs.npz"),
            rows=rows,
        ),
    )


def prepared_batch(clean, source, strength, depth):
    p = prepared_inputs(clean, source, strength, depth)
    p["distance"] = jax.numpy.asarray(np.stack([coverage_distance(x) for x in np.asarray(clean)]))
    return p


def objective(model, prepared, config, weight=0.1):
    predictions, physical = recursive_forward_with_preclip(
        model, prepared["noisy"], prepared["levels"], guides=prepared["guides"]
    )
    base, parts = image_loss(
        predictions[0],
        prepared["target"],
        prepared["levels"][0],
        config.loss,
        config.diffusion.min_noise,
        preclip=physical[0],
    )
    field = distance_supervision(
        physical[0],
        prepared["distance"],
        prepared["target"],
        prepared["levels"][0] > config.diffusion.min_noise,
    )
    loss = base + weight * field
    return loss, dict(parts, loss=loss, distance_supervision=field)


def make_arms(root):
    initial, _ = load_inference(root / "initial")
    config = load_config(root / "configs/distance.json")
    arm = create_arm(config, initial)
    assert_initial_parameters(initial, arm["model"])

    @nnx.jit
    def gradient(model, prepared):
        (_, metrics), grads = nnx.value_and_grad(
            lambda m: objective(m, prepared, config), has_aux=True
        )(model)
        return metrics, grads

    arm["gradient"] = gradient
    return {"distance": arm}


def prepare(root):
    assert not root.exists()
    (root / "configs").mkdir(parents=True)
    data_parent = Path("../jax-paper-band-fit-20261010/runs/paper-band-fit-20261010").resolve()
    parent = data_parent
    verify(parent)
    oracle = Path(
        "../jax-coverage-distance-oracle-20261011/runs/coverage-distance-oracle-20261011"
    ).resolve()
    verify(oracle)
    assert read(oracle / "final-audit.json")["round_trip_passed"]
    control, config = load_inference(parent / "snapshots/step-1200/band/raw")
    model, _ = load_inference(read(parent / "plan.json")["initial"])
    assert config.train.steps == STEPS and config.train.peak_learning_rate == 1e-4
    save_inference(control, config, root / "control", provenance="Frozen AD band1200 control")
    save_inference(model, config, root / "initial", provenance="Same R4000 initial weights as AD")
    assert read(root / "initial/metadata.json")["parameters"] == 130605
    for method in TRAINED:
        save_config(config, root / f"configs/{method}.json")
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(data_parent / name, root / name)
    target_oracle(root)
    audit_data(root)
    initial = root / "initial"
    write(
        root / "plan.json",
        dict(
            initial=str(initial),
            initial_files={p.name: sha(p) for p in initial.iterdir()},
            data={
                "fixed_pairs": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))
            },
            data_parent=str(data_parent),
            data_source_sha256=sha(data_parent / "source.json"),
            parent=str(parent),
            parent_source_sha256=sha(parent / "source.json"),
            parent_final_sha256=sha(parent / "final-audit.json"),
            oracle=str(oracle),
            oracle_final_sha256=sha(oracle / "final-audit.json"),
            control_files={str(p): sha(p) for p in (root / "control").iterdir()},
            control_metrics_sha256=sha(parent / "evaluation/step-1200/metrics.json"),
            crops=read(data_parent / "plan.json")["crops"],
            target_oracle_sha256=sha(root / "target-oracle.json"),
            steps=STEPS,
            seed=SEED,
            evaluation_seed_by_depth="2026101066 + depth",
            depths=list(range(1, 7)),
            adoption=False,
            plan_document_sha256=sha(Path("docs/distance-supervision-fit-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    assert (
        sha(Path("docs/distance-supervision-fit-plan-20261011.md")) == plan["plan_document_sha256"]
    )
    for prefix in ("parent", "data_parent"):
        verify(Path(plan[prefix]))
    assert sha(Path(plan["parent"]) / "source.json") == plan["parent_source_sha256"]
    assert sha(Path(plan["parent"]) / "final-audit.json") == plan["parent_final_sha256"]
    assert sha(Path(plan["data_parent"]) / "source.json") == plan["data_source_sha256"]
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    for p, digest in plan["control_files"].items():
        assert sha(Path(p)) == digest
    assert (
        sha(Path(plan["parent"]) / "evaluation/step-1200/metrics.json")
        == plan["control_metrics_sha256"]
    )
    verify(Path(plan["oracle"]))
    assert sha(Path(plan["oracle"]) / "final-audit.json") == plan["oracle_final_sha256"]
    return plan


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    data = arrays(root, TRAIN)
    prepared = {d: prepared_batch(*data, d) for d in range(1, 7)}
    control = load_inference(root / "control")[0]
    control_prediction = run_sample(control, *arrays(root, TRAIN + PROBE), 3)
    with np.load(
        Path(read(root / "plan.json")["parent"]) / "evaluation/step-1200/predictions.npz"
    ) as reference:
        for i, value in zip(TRAIN + PROBE, control_prediction, strict=True):
            np.testing.assert_array_equal(value, reference[f"band-{i}"])
    write(root / "control-preflight.json", dict(eight_outputs_exact=True))
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=4) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else 0
        if saved.latest_step() is None:
            for arm in arms.values():
                assert int(arm["optimizer"].step[...]) == 0
                jax.tree.map(
                    np.testing.assert_array_equal, arm["ema"], nnx.state(arm["model"], nnx.Param)
                )
                for n in (0, 100, 1199):
                    assert abs(float(arm["schedule"](n)) - 1e-4) < 1e-10
            write(
                root / "initial-audit.json",
                dict(
                    initial_parameters_exact=True,
                    optimizer_zero=True,
                    ema_equals_model=True,
                    constant_lr=1e-4,
                    inherited_from_R4000=True,
                ),
            )
            lineart_state.save(saved, arms, 0)
            snapshot(root, arms, 0)
            write(out / "restore-audit-0.json", dict(all_state_exact=True))
        prefix = []
        for step in range(start, STEPS):
            depth = step % 6 + 1
            p = prepared[depth]
            order = TRAINED
            stats = {name: update(arms[name], p) for name in order}
            complete = step + 1
            if complete <= 6:
                prefix.append(stats)
            if complete <= 6 or complete % 50 == 0:
                row = dict(step=complete, depth=depth, metrics=stats)
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete == 6:
                replay = make_arms(root)
                for n in range(6):
                    p = prepared[n % 6 + 1]
                    values = {name: update(replay[name], p) for name in TRAINED}
                    if start == 0:
                        assert values == prefix[n]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, 6),
                    lineart_state.payload(replay, 6),
                )
                write(
                    root / "prefix-audit.json",
                    dict(
                        updates=6,
                        all_state_exact=True,
                        metrics_exact=start == 0,
                        depths=list(range(1, 7)),
                    ),
                )
            if complete in SAVED:
                lineart_state.save(saved, arms, complete)
                restored = make_arms(root)
                assert lineart_state.restore(saved, restored, complete) == complete
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, complete),
                    lineart_state.payload(restored, complete),
                )
                write(out / f"restore-audit-{complete}.json", dict(all_state_exact=True))
                snapshot(root, arms, complete)
        bindings = {}
        for step in SAVED:
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, step) == step
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.clone(arm["model"])
                    if mode == "ema":
                        nnx.update(expected, arm["ema"])
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    actual, c = load_inference(folder)
                    assert c == arm["config"]
                    jax.tree.map(
                        np.testing.assert_array_equal,
                        nnx.state(expected, nnx.Param),
                        nnx.state(actual, nnx.Param),
                    )
                    bindings.update({str(p): sha(p) for p in folder.iterdir()})
        write(root / "snapshot-audit.json", dict(exports=8, raw_ema_exact=True, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def run_sample(model, clean, source, strength, depth):
    return np.asarray(forward(model, prepared_inputs(clean, source, strength, depth)))


def evaluate(root):
    plan = check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    models = {
        "distance": load_inference(root / f"snapshots/step-{STEPS}/distance/raw")[0],
        "control": load_inference(root / "control")[0],
    }
    initial = load_inference(root / "initial")[0]
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows, bindings, displays, partial_displays, replay = [], {}, {}, {}, []
    with np.load(root / "inputs.npz") as data:
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            clean, source, strength = arrays(root, ids)
            for depth in (1, 3, 6):
                active = dict(models)
                if depth == 3:
                    active["initial"] = initial
                for method, model in active.items():
                    prediction = run_sample(model, clean, source, strength, depth)
                    assert np.isfinite(prediction).all()
                    if method == "control" and depth == 3:
                        reference = Path(plan["parent"]) / "evaluation/step-1200/predictions.npz"
                        with np.load(reference) as a:
                            for i, value in zip(ids, prediction, strict=True):
                                np.testing.assert_array_equal(value, a[f"band-{i}"])
                                replay.append(i)
                    path = out / f"predictions/{split}-d{depth}-{method}.npz"
                    np.savez_compressed(path, prediction=prediction)
                    bindings[str(path)] = sha(path)
                    for i, value in zip(ids, prediction, strict=True):
                        target, level = data[f"{i}-image"], data[f"{i}-strength"]
                        metrics = protected_metrics(value, target, level) | halo_metrics(
                            value, target, level
                        )
                        metrics |= probe_metrics(
                            value, target, data[f"{i}-probes"], data[f"{i}-connections"]
                        )
                        metrics |= coverage_metrics(value, target, level)
                        assert all(v is None or np.isfinite(v) for v in metrics.values())
                        rows.append(
                            dict(method=method, case=i, split=split, depth=depth, **metrics)
                        )
                        if depth == 3 and method in METHODS:
                            displays[(method, i)] = value
        clean, source, strength = arrays(root, TRAIN)
        strength = strength.at[:, :, :256].set(0)
        for method, model in models.items():
            prediction = run_sample(model, clean, source, strength, 6)
            assert np.isfinite(prediction).all()
            mask = np.asarray(strength) == 0
            np.testing.assert_array_equal(prediction[mask], np.asarray(source)[mask])
            path = out / f"predictions/partial-d6-{method}.npz"
            np.savez_compressed(path, prediction=prediction)
            bindings[str(path)] = sha(path)
            for i, value in zip(TRAIN, prediction, strict=True):
                partial_displays[(method, i)] = value
        checks = [
            c
            for row in rows
            if row["depth"] == 3 and row["method"] in METHODS
            for c in fit_checks([row])
        ]
        guards = []
        for i in TRAIN + PROBE:
            lookup = {r["method"]: r for r in rows if r["case"] == i and r["depth"] == 3}
            for metric, limit, lower in (
                ("ink_f1_alpha50", 0.005, False),
                ("ink_deficit_fraction", 0.01, True),
            ):
                a, b = lookup["distance"][metric], lookup["control"][metric]
                guards.append(
                    dict(
                        case=i,
                        metric=metric,
                        distance=a,
                        control=b,
                        passed=a <= b + limit if lower else a >= b - limit,
                    )
                )
        assert len(rows) == 56 and len(checks) == 44 and len(guards) == 16 and len(bindings) == 16
        assert set(replay) == set(TRAIN + PROBE) and len(replay) == 8
        write(root / "replay-audit.json", dict(exact=True, cases=replay))
        write(
            out / "metrics.json",
            dict(
                rows=rows,
                checks=checks,
                retention_checks=guards,
                bindings=bindings,
                finite=True,
                protected_exact=True,
                adoption=False,
                fit_passed={
                    m: all(c["passed"] for c in checks if c["method"] == m) for m in METHODS
                },
                retention_passed=all(c["passed"] for c in guards),
            ),
        )
        records = []
        columns = ("target", "source", *METHODS)
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            for native in (True, False):
                canvas = Image.new("L", (160 * len(columns), 184 * len(ids)), 255)
                for row, i in enumerate(ids):
                    values = [
                        data[f"{i}-image"],
                        data[f"{i}-source"],
                        *[displays[(m, i)] for m in METHODS],
                    ]
                    for col, (label, value) in enumerate(zip(columns, values, strict=True)):
                        pixels = np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(
                            np.uint8
                        )
                        tile = Image.fromarray(pixels)
                        tile = (
                            tile.crop(plan["crops"][str(i)])
                            if native
                            else tile.resize((160, 160), Image.Resampling.LANCZOS)
                        )
                        canvas.paste(tile, (160 * col, 184 * row + 24))
                        ImageDraw.Draw(canvas).text(
                            (160 * col + 2, 184 * row + 4), f"{i} {label}", fill=0
                        )
                path = out / f"{split}-{'native' if native else 'overview'}.png"
                canvas.save(path)
                records.append(dict(file=path.name, sha256=sha(path)))
        canvas = Image.new("L", (160 * 4, 184 * 4), 255)
        for row, i in enumerate(TRAIN):
            target = np.where(
                np.asarray(strength[row]) > 0, data[f"{i}-image"], data[f"{i}-source"]
            )
            values = [target, data[f"{i}-source"], *[partial_displays[(m, i)] for m in METHODS]]
            for col, (label, value) in enumerate(
                zip(("masked_target", "source", *METHODS), values, strict=True)
            ):
                pixels = np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                canvas.paste(
                    Image.fromarray(pixels).crop(plan["crops"][str(i)]), (160 * col, 184 * row + 24)
                )
                ImageDraw.Draw(canvas).text((160 * col + 2, 184 * row + 4), f"{i} {label}", fill=0)
        path = out / "partial-native.png"
        canvas.save(path)
        records.append(dict(file=path.name, sha256=sha(path)))
        write(
            out / "render-manifest.json",
            dict(records=records, deterministic_source_model=True, crops_from_AD=True),
        )
    write(
        root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=STEPS)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "train", "evaluate"])
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
        {"prepare": prepare, "evaluate": evaluate}[args.action](root)
