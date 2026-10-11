"""Compare a full-resolution local CNN with a fresh U-Net on two-step restoration."""

import argparse
import fcntl
import hashlib
import json
import shutil
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import velocity as endpoint_velocity
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, snapshot, update, verify
from flax import nnx
from geometry_velocity_fit import prepared_batch
from paper_band_fit import PROBE, TRAIN, fit_checks, target_oracle
from PIL import Image, ImageDraw
from rollout_endpoint import inputs
from rollout_image_loss import gradient as image_gradient
from velocity_bridge_toy import sample as uniform_sample

from lineartgen.config import ModelConfig, load_config, save_config
from lineartgen.coverage_metrics import coverage_metrics
from lineartgen.evaluation import probe_metrics, protected_metrics
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models import create_model
from lineartgen.models.serialization import load_inference, save_inference
from lineartgen.training import checkpoints, lineart_state
from lineartgen.training.lineart_study import assert_initial_parameters
from lineartgen.training.steps import create_optimizer, make_steps

METHODS = ("reference", "unet", "local")
TRAINED = ("unet", "local")
START = 0
STEPS = 1200
SEED = 2026101093
EVAL_SEED = 2026102093
SAVED = (START, START + 6, 400, STEPS)
PARAMETERS = {"unet": 134688, "local": 15809}


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


def local_config():
    return ModelConfig(
        architecture="local",
        channels=(16,),
        convolutions_per_block=2,
        local_blocks=3,
        source_conditioning=True,
        progress_conditioning=True,
        progress_modulation="fourier_film",
        output_mode="residual",
    )


def fresh_model(c):
    model = create_model(c.model, rngs=nnx.Rngs(c.train.initialization_seed))
    if c.model.architecture == "unet":
        for block in (model.head, model.detail_head):
            block.layers[-1].kernel[...] = jnp.zeros_like(block.layers[-1].kernel[...])
            block.layers[-1].bias[...] = jnp.zeros_like(block.layers[-1].bias[...])
    return model


def make_arm(c, initial):
    model = nnx.clone(initial)
    optimizer, schedule = create_optimizer(model, c)
    arm = dict(
        config=c,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        ema=jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param)),
        update=make_steps(c)[1],
        critic=None,
    )
    arm["gradient"] = partial(
        image_gradient, loss_config=c.loss, min_noise=c.diffusion.min_noise, coefficient=0.1
    )
    return arm


def make_arms(root):
    arms = {}
    for name in TRAINED:
        c = load_config(root / f"configs/{name}.json")
        initial, saved = load_inference(root / f"initials/{name}")
        assert c == saved
        arm = make_arm(c, initial)
        assert_initial_parameters(initial, arm["model"])
        assert int(arm["optimizer"].step[...]) == 0
        arms[name] = arm
    return arms


def control_preflight(root):
    parent = Path(read(root / "plan.json")["parent"])
    reference = load_inference(root / "initial")[0]
    for steps, directory in ((2, "evaluation"), (32, "evaluation32")):
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            _, source, strength = arrays(root, ids)
            key = jax.random.fold_in(
                jax.random.PRNGKey(EVAL_SEED), 300 + int(split == "probe") * 10
            )
            value = run_sample(reference, source, strength, key, 3, steps=steps)
            with np.load(parent / f"{directory}/predictions/{split}-d3-r0-image.npz") as old:
                np.testing.assert_array_equal(value, old["physical"])
    write(root / "control-preflight.json", dict(sixteen_CH_outputs_exact=True))


def prepare(root):
    assert not root.exists()
    parent = Path("../jax-rollout-image-loss-20261011/runs/rollout-image-loss-20261011").resolve()
    from rollout_image_fit import check as parent_check

    parent_check(parent)
    assert read(parent / "final-audit.json")["stage"] == "complete_rollout_image_loss_diagnostic"
    reference, old = load_inference(parent / "snapshots/step-8400/image/raw")
    train = old.train.model_copy(
        update={
            "steps": STEPS,
            "learning_rate": 5e-5,
            "minimum_learning_rate": 1e-4,
            "warmup_steps": 0,
            "learning_rate_decay_steps": STEPS,
            "model_seed": 42,
        }
    )
    (root / "configs").mkdir(parents=True)
    bound = []
    for name in TRAINED:
        c = old.model_copy(
            update={"train": train, "model": old.model if name == "unet" else local_config()}
        )
        assert c.train.peak_learning_rate == 1e-4 and c.train.batch_size == 4
        save_config(c, root / f"configs/{name}.json")
        m = fresh_model(c)
        save_inference(m, c, root / f"initials/{name}", provenance="fresh seed42 zero output head")
        assert read(root / f"initials/{name}/metadata.json")["parameters"] == PARAMETERS[name]
        bound += list((root / f"initials/{name}").iterdir())
    save_inference(
        reference, old, root / "initial", provenance="CH image8400 reference; not matched training"
    )
    for name in ("inputs.npz", "train.json", "probe.json"):
        shutil.copyfile(parent / name, root / name)
    target_oracle(root)
    audit_data(root)
    bound += [
        parent / n
        for n in (
            "source.json",
            "final-audit.json",
            "evaluation/metrics.json",
            "evaluation32/metrics.json",
        )
    ]
    bound += list((parent / "snapshots/step-8400/image/raw").iterdir())
    bound += list((parent / "evaluation/predictions").glob("*-image.npz"))
    bound += list((parent / "evaluation32/predictions").glob("*-image.npz"))
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={
                "fixed_pairs": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))
            },
            parent=str(parent),
            bound_files={str(p): sha(p) for p in bound},
            crops=read(parent / "plan.json")["crops"],
            target_oracle_sha256=sha(root / "target-oracle.json"),
            start=START,
            steps=STEPS,
            saved=list(SAVED),
            seed=SEED,
            evaluation_seed=EVAL_SEED,
            depths=list(range(1, 7)),
            adoption=False,
            plan_document_sha256=sha(Path("docs/local-recursive-cnn-plan-20261011.md")),
        ),
    )


def check(root):
    verify(root)
    plan = read(root / "plan.json")
    from rollout_image_fit import check as parent_check

    parent_check(Path(plan["parent"]))
    assert sha(Path("docs/local-recursive-cnn-plan-20261011.md")) == plan["plan_document_sha256"]
    for path, digest in plan["bound_files"].items():
        assert sha(Path(path)) == digest
    assert sha(root / "target-oracle.json") == plan["target_oracle_sha256"]
    assert all(c["passed"] for c in read(root / "target-oracle.json")["checks"])
    return plan


def train(root):
    check(root)
    assert read(root / "test-results.json")["passed"]
    assert jax.devices()[0].platform != "cpu"
    arms = make_arms(root)
    data = arrays(root, TRAIN)
    control_preflight(root)
    out = root / "training"
    out.mkdir(exist_ok=True)
    with checkpoints.manager(out / "checkpoints", keep=4) as saved:
        start = lineart_state.restore(saved, arms) if saved.latest_step() is not None else START
        if saved.latest_step() is None:
            from anchored_endpoint import endpoints

            p = inputs(prepared_batch(*data, 0, 3)[0])
            values = []
            for name, arm in arms.items():
                assert int(arm["optimizer"].step[...]) == 0
                assert_initial_parameters(
                    load_inference(root / f"initials/{name}")[0], arm["model"]
                )
                for n in (0, 1, STEPS - 1):
                    assert abs(float(arm["schedule"](n)) - 1e-4) < 1e-10
                values.append(
                    endpoints(
                        arm["model"],
                        p["images"],
                        p["levels"],
                        p["guides"],
                        tuple(jnp.ones_like(x) for x in p["images"]),
                    )
                )
            jax.tree.map(np.testing.assert_array_equal, *values)
            write(
                root / "initial-audit.json",
                dict(
                    initial_functions_exact=True,
                    initial_parameters_exact=True,
                    optimizer_step=0,
                    parameters=PARAMETERS,
                    fresh_optimizer_and_ema=True,
                ),
            )
            lineart_state.save(saved, arms, START)
            restored = make_arms(root)
            assert lineart_state.restore(saved, restored, START) == START
            jax.tree.map(
                np.testing.assert_array_equal,
                lineart_state.payload(arms, START),
                lineart_state.payload(restored, START),
            )
            snapshot(root, arms, START)
            write(out / f"restore-audit-{START}.json", dict(all_state_exact=True))
        journal = out / "sampling.jsonl"
        records = (
            [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        )
        if any(r["step"] > start for r in records):
            write(out / f"resume-discarded-{start}.json", [r for r in records if r["step"] > start])
            records = [r for r in records if r["step"] <= start]
            journal.write_text("".join(json.dumps(r) + "\n" for r in records))
        assert [r["step"] for r in records] == list(range(START + 1, start + 1))
        prefix = []
        for step in range(start, STEPS):
            depth = step % 6 + 1
            batch, tau = prepared_batch(*data, step, depth)
            p = inputs(batch)
            order = TRAINED
            stats, elapsed = {}, {}
            for name in order:
                tick = time.perf_counter()
                stats[name] = update(arms[name], p)
                elapsed[name] = time.perf_counter() - tick
            complete = step + 1
            digest = hashlib.sha256()
            for x in jax.tree.leaves(p):
                digest.update(np.asarray(x).tobytes())
            with journal.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            step=complete,
                            depth=depth,
                            inputs_sha256=digest.hexdigest(),
                            elapsed_seconds=elapsed,
                            tau_mean=float(tau.mean()),
                        )
                    )
                    + "\n"
                )
            if complete <= START + 6:
                prefix.append(stats)
            if complete <= START + 6 or complete % 50 == 0:
                row = dict(step=complete, depth=depth, metrics=stats, tau_mean=float(tau.mean()))
                write(out / "progress.json", row)
                print(row, flush=True)
            if complete == START + 6:
                replay = make_arms(root)
                for n in range(START, START + 6):
                    p = inputs(prepared_batch(*data, n, n % 6 + 1)[0])
                    values = {name: update(replay[name], p) for name in TRAINED}
                    if start == START:
                        assert values == prefix[n - START]
                jax.tree.map(
                    np.testing.assert_array_equal,
                    lineart_state.payload(arms, START + 6),
                    lineart_state.payload(replay, START + 6),
                )
                write(
                    root / "prefix-audit.json",
                    dict(
                        updates=6,
                        all_state_exact=True,
                        metrics_exact=start == START,
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
        write(root / "snapshot-audit.json", dict(exports=16, raw_ema_exact=True, bindings=bindings))
    write(root / "workflow-status.json", dict(stage="training_complete", step=STEPS))


def run_sample(model, source, strength, key, depth, method="initial", steps=32):
    def predict(x, t, s, u):
        return endpoint_velocity(model, x, t, s, u, False)

    return np.asarray(uniform_sample(predict, source, strength, key, depth=depth, steps=steps))


def evaluate(root, steps=2):
    plan = check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    models = {m: load_inference(root / f"snapshots/step-{STEPS}/{m}/raw")[0] for m in TRAINED}
    models["reference"] = load_inference(root / "initial")[0]
    out = root / ("evaluation" if steps == 2 else "evaluation32")
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    reference_directory = "evaluation" if steps == 2 else "evaluation32"
    old_metrics = read(Path(plan["parent"]) / reference_directory / "metrics.json")
    old_rows = {
        (r["case"], r["depth"], r["draw"]): r for r in old_metrics["rows"] if r["method"] == "image"
    }
    rows = []
    bindings = {}
    replay = []
    displays = {}
    partial_displays = {}
    with np.load(root / "inputs.npz") as data:
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            clean, source, strength = arrays(root, ids)
            for depth in (1, 3, 6):
                for draw in range(4):
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(EVAL_SEED),
                        depth * 100 + int(split == "probe") * 10 + draw,
                    )
                    active_models = dict(models)
                    for method, model in active_models.items():
                        physical = run_sample(model, source, strength, key, depth, method, steps)
                        prediction = np.clip(physical, -1, 1)
                        assert np.isfinite(physical).all()
                        if method == "reference":
                            reference = (
                                Path(plan["parent"])
                                / reference_directory
                                / f"predictions/{split}-d{depth}-r{draw}-image.npz"
                            )
                            with np.load(reference) as a:
                                np.testing.assert_array_equal(physical, a["physical"])
                                np.testing.assert_array_equal(prediction, a["prediction"])
                            replay.append(str(reference))
                        path = out / f"predictions/{split}-d{depth}-r{draw}-{method}.npz"
                        np.savez_compressed(path, physical=physical, prediction=prediction)
                        bindings[str(path)] = sha(path)
                        for i, value in zip(ids, prediction, strict=True):
                            target, level = data[f"{i}-image"], data[f"{i}-strength"]
                            metrics = protected_metrics(value, target, level)
                            metrics |= halo_metrics(value, target, level)
                            metrics |= probe_metrics(
                                value, target, data[f"{i}-probes"], data[f"{i}-connections"]
                            )
                            metrics |= coverage_metrics(value, target, level)
                            assert all(v is None or np.isfinite(v) for v in metrics.values())
                            rows.append(
                                dict(
                                    method=method,
                                    case=i,
                                    split=split,
                                    depth=depth,
                                    draw=draw,
                                    **metrics,
                                )
                            )
                            if method == "reference":
                                assert {**rows[-1], "method": "image"} == old_rows[i, depth, draw]
                            if depth == 3 and method in METHODS:
                                displays[(method, i, draw)] = value
                    write(
                        out / "progress.json",
                        dict(split=split, depth=depth, draw=draw, rows=len(rows), total=288),
                    )
        clean, source, strength = arrays(root, TRAIN)
        strength = strength.at[:, :, :256].set(0)
        key = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), 9000)
        for method, model in models.items():
            physical = run_sample(model, source, strength, key, 6, method, steps)
            prediction = np.clip(physical, -1, 1)
            mask = np.asarray(strength) == 0
            assert np.isfinite(physical).all()
            np.testing.assert_array_equal(physical[mask], np.asarray(source)[mask])
            np.testing.assert_array_equal(prediction[mask], np.asarray(source)[mask])
            if method == "reference":
                reference = (
                    Path(plan["parent"]) / f"{reference_directory}/predictions/partial-d6-image.npz"
                )
                with np.load(reference) as a:
                    np.testing.assert_array_equal(physical, a["physical"])
                    np.testing.assert_array_equal(prediction, a["prediction"])
                replay.append(str(reference))
            path = out / f"predictions/partial-d6-{method}.npz"
            np.savez_compressed(path, physical=physical, prediction=prediction)
            bindings[str(path)] = sha(path)
            for i, value in zip(TRAIN, prediction, strict=True):
                partial_displays[(method, i)] = value
        assert len(replay) == 25
        write(
            root / f"replay-audit-{steps}.json",
            dict(exact=True, metric_rows_exact=96, files=replay),
        )
        checks = []
        for row in rows:
            if row["depth"] == 3 and row["method"] in METHODS:
                for c in fit_checks([row]):
                    checks.append(dict(c, depth=3, draw=row["draw"]))
        guards = []
        for i in TRAIN + PROBE:
            for draw in range(4):
                values = {
                    r["method"]: r
                    for r in rows
                    if r["case"] == i and r["depth"] == 3 and r["draw"] == draw
                }
                for compared, reference in (
                    ("unet", "reference"),
                    ("local", "reference"),
                    ("local", "unet"),
                ):
                    for metric, limit, lower in (
                        ("ink_f1_alpha50", 0.005, False),
                        ("ink_deficit_fraction", 0.01, True),
                    ):
                        a, b = values[compared][metric], values[reference][metric]
                        guards.append(
                            dict(
                                case=i,
                                draw=draw,
                                metric=metric,
                                method=compared,
                                reference=reference,
                                passed=a <= b + limit if lower else a >= b - limit,
                                value=a,
                                reference_value=b,
                            )
                        )
        assert (
            len(rows) == 288 and len(checks) == 264 and len(guards) == 192 and len(bindings) == 75
        )
        write(
            out / "metrics.json",
            dict(
                rows=rows,
                checks=checks,
                retention_checks=guards,
                fit_passed={
                    m: all(c["passed"] for c in checks if c["method"] == m) for m in METHODS
                },
                retention_passed=all(c["passed"] for c in guards),
                finite=True,
                protected_exact=True,
                bindings=bindings,
                adoption=False,
            ),
        )
        records = []
        columns = ("target", "source", *[f"{m}[{n}]" for m in METHODS for n in range(4)])
        for split, ids in (("train", TRAIN), ("probe", PROBE)):
            for native in (True, False):
                canvas = Image.new("L", (160 * len(columns), 184 * len(ids)), 255)
                for row, i in enumerate(ids):
                    values = [
                        data[f"{i}-image"],
                        data[f"{i}-source"],
                        *[displays[(m, i, n)] for m in METHODS for n in range(4)],
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
        canvas = Image.new("L", (160 * (2 + len(METHODS)), 184 * 4), 255)
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
            dict(
                records=records,
                all_four_fixed_draws=True,
                partial_one_draw_per_case=True,
                crops_from_AD=True,
            ),
        )
    write(
        root / "workflow-status.json",
        dict(stage=f"numerical{steps}_complete_pending_review", step=STEPS),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "freeze", "train", "evaluate"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sample-steps", type=int, choices=[2, 32], default=2)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "train":
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "workflow-status.json", dict(stage="fresh0to1200"))
            train(root)
    elif args.action == "freeze":
        freeze(root)
    elif args.action == "evaluate":
        evaluate(root, args.sample_steps)
    else:
        prepare(root)
