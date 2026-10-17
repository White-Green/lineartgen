"""Ablate geometric regularizers with the same informative pooled model and GO pair."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from opening_prior_objective import opening_consistency
from PIL import Image, ImageDraw
from plausible_study import digest, exact
from pooled_context_model import add_pool, load_field, pooled_cue
from pooled_context_model import save_field as save_pool
from quartic_field_objective import field_parts
from scale_style_balance_study import update
from subpixel_field_data import candidate_fields
from subpixel_field_model import coverage, field_forward

from lineartgen.config import load_config, save_config
from lineartgen.diffusion import source_guides
from lineartgen.training import checkpoints, lineart_state

GO = Path(
    "../jax-wide-context-identifiability-20261017/runs/wide-context-identifiability-20261017"
).resolve()
GP = Path("../jax-pooled-procedural-20261017/runs/pooled-procedural-20261017-retry1").resolve()
FW = Path("../jax-mixed-context-20261016/runs/mixed-context-20261016").resolve()
MODEL = FW / "snapshots/step-6000/mixed/raw"
GQ = Path(
    "../jax-pooled-collision-fit-20261017/runs/pooled-collision-fit-20261017-retry1"
).resolve()
GR = Path("../jax-coarse-collision-fit-20261017/runs/coarse-collision-fit-20261017").resolve()
DOC = Path("docs/collision-loss-ablation-plan-20261017.md")
METHODS = ("no_opening", "no_geometry")
SAVED = (0, 6, 400, 800, 1200)
QUERY = (128, 85)


def gradient_for(name):
    assert name in METHODS
    geometry_weight = 0.1 if name == "no_opening" else 0.0

    @nnx.jit
    def gradient(model, prepared, target, unused):
        def objective(m):
            fields = field_forward(m, prepared)
            parts = field_parts(
                fields, (target,), prepared.noise_levels, gradient_weight=geometry_weight
            )
            # Report opening for diagnosis, with no contribution to either loss.
            prior = opening_consistency(fields[0], target, prepared.noise_levels[0])
            parts |= dict(opening_consistency=prior)
            return parts["loss"], parts

        (_, parts), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        return parts, grads

    return gradient


@nnx.jit
def forward(model, prepared):
    field = field_forward(model, prepared)[0]
    return field, coverage(field)


def arms_from_root(root):
    cfg = load_config(root / "config.json")
    result = {}
    for name in METHODS:
        model, cc = load_field(root / f"initials/{name}")
        assert cc == cfg
        arm = make_arm(cfg, model)
        arm["gradient"] = gradient_for(name)
        result[name] = arm
    return result


def batch(root, depth=1):
    destination = jax.devices()[0]
    # Freeze the same CPU-generated batch used by prepare, then transfer unchanged.
    with jax.default_device(jax.devices("cpu")[0]):
        with np.load(root / "inputs.npz") as z:
            source, clean = z["source"], z["clean"]
            target = candidate_fields(z["base"], z["direction"], 1)[0]
        p = prepare_inputs(
            clean, source, np.full_like(source, 0.5), jax.random.PRNGKey(2026101704), depth
        )
        result = p, target, jnp.zeros(())
    return jax.device_put(result, destination)


def page(path, rows, labels):
    canvas = Image.new("RGB", (256 * len(rows[0]), 292 * len(rows)), "white")
    pen = ImageDraw.Draw(canvas)
    for i, (values, names) in enumerate(zip(rows, labels, strict=True)):
        for j, (v, name) in enumerate(zip(values, names, strict=True)):
            px = np.uint8(np.rint(np.clip((v[..., 0] + 1) * 127.5, 0, 255)))
            canvas.paste(Image.fromarray(px).convert("RGB"), (j * 256, i * 292 + 36))
            pen.text((j * 256 + 3, i * 292 + 6), name, fill="black")
    canvas.save(path)


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(GP / "completion.json")["stage"] == "complete_pooled_procedural"
    assert read(GQ / "completion.json")["stage"] == "complete_pooled_collision_fit"
    assert read(GR / "completion.json")["stage"] == "complete_coarse_collision_fit"
    assert read(GO / "completion.json")["stage"] == "complete_wide_context_identifiability"
    root.mkdir(parents=True, exist_ok=False)
    (root / "diagnostics").mkdir()
    shutil.copy2(GO / "inputs.npz", root / "inputs.npz")
    model, cfg = load_field(MODEL)
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=1200,
                    learning_rate_decay_steps=1200,
                    batch_size=2,
                    accumulate_steps=1,
                    learning_rate=2e-4,
                    minimum_learning_rate=2e-5,
                    warmup_steps=0,
                )
            )
        )
    )
    assert abs(cfg.train.peak_learning_rate - 2e-4 * np.sqrt(2)) < 1e-12
    save_config(cfg, root / "config.json")
    pooled = add_pool(model, cfg)
    state = nnx.to_pure_dict(nnx.state(pooled, nnx.Param))
    added = state.pop("pooled_stem")
    exact(state, nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    assert all(np.count_nonzero(np.asarray(v)) == 0 for v in jax.tree.leaves(added))
    for name in METHODS:
        m, save = pooled, save_pool
        assert sum(v.size for v in jax.tree.leaves(nnx.state(m, nnx.Param))) == 21778
        save(
            m,
            cfg,
            root / f"initials/{name}",
            provenance="GS loss ablation; untouched FW plus zero pooled cue; research only",
        )
        restored, cc = load_field(root / f"initials/{name}")
        assert cc == cfg
        exact(nnx.state(m, nnx.Param), nnx.state(restored, nnx.Param))
    with np.load(root / "inputs.npz") as z:
        source, clean = z["source"], z["clean"]
    p, target, _ = batch(root)
    assert digest((p, target)) == read(GQ / "initial-audit.json")["input_sha256"]
    inits = arms_from_root(root)
    exact(
        nnx.state(inits["no_opening"]["model"], nnx.Param),
        nnx.state(inits["no_geometry"]["model"], nnx.Param),
    )
    np.testing.assert_array_equal(np.asarray(coverage(target[:, 4])), clean)
    normalized = np.asarray(source_guides(model, p.guides)[0])
    sl = (slice(52, 205), slice(9, 162), slice(None))
    np.testing.assert_array_equal(source[(0, *sl)], source[(1, *sl)])
    np.testing.assert_array_equal(normalized[(0, *sl)], normalized[(1, *sl)])
    y, x = QUERY
    tq = np.asarray(target[:, :, y * 4 : (y + 1) * 4, x * 4 : (x + 1) * 4])
    assert (tq[0] >= 0.25).all() and (tq[1] <= -0.25).all()
    pc = np.asarray(pooled_cue(normalized, 0))
    pool_diff = float(np.max(abs(pc[(0, *sl)] - pc[(1, *sl)])))
    assert pool_diff > 0.1
    page(
        root / "teacher-preview.png",
        [[source[i], clean[i]] for i in range(2)],
        [["bar source", "bar center target"], ["fill source", "fill target"]],
    )
    write(
        root / "initial-audit.json",
        dict(
            old_arrays_exact=True,
            added_zero=True,
            raw_normalized_patches_exact=True,
            all_nine_teachers_disagree=True,
            pooled_patch_max_difference=pool_diff,
            input_sha256=digest((p, target)),
        ),
    )
    files = [p for p in root.rglob("*") if p.is_file()]
    files += [
        DOC.resolve(),
        GQ / "completion.json",
        GQ / "metrics.json",
        GQ / "initial-audit.json",
        GQ / "diagnostics/posthoc-review-erratum.json",
        GR / "completion.json",
        GR / "decision.json",
        GO / "inputs.npz",
        GO / "completion.json",
        GP / "completion.json",
        GP / "diagnostics/decision.json",
        FW / "completion.json",
    ]
    files += [GO / f"predictions-initial-{d}.npz" for d in (1, 3, 6)]
    files += list(MODEL.iterdir())
    files += [GQ / f"prediction-pooled-1200-depth{d}.npz" for d in (1, 3, 6)]
    files += [
        Path(f).resolve()
        for f in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if f.endswith((".py", ".toml")) or f == "uv.lock"
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            bindings={str(p.resolve()): sha(p) for p in files},
            archive_sha256=sha(root / "source.tar.gz"),
            steps=1200,
            saved=SAVED,
            parameters={name: 21778 for name in METHODS},
            outputs=60,
            pages=6,
            fitting_only=True,
            adoption=False,
        ),
    )


def check(root):
    p = read(root / "plan.json")
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f
    assert sha(root / "source.tar.gz") == p["archive_sha256"]


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            m = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(m, arm["ema"])
            save = save_pool
            save(
                m,
                arm["config"],
                root / f"snapshots/step-{step}/{name}/{mode}",
                provenance=f"GS fitting only {step} {mode}; never use as production initializer",
            )


def train(root, until):
    check(root)
    assert jax.devices()[0].platform != "cpu" and until in (400, 800, 1200)
    assert read(root / "test-results.json")["passed"]
    tr = read(root / "teacher-review.json")
    assert tr["reviewed"] and tr["sha256"] == sha(root / "teacher-preview.png")
    arms, b = arms_from_root(root), batch(root)
    assert digest(b[:2]) == read(root / "initial-audit.json")["input_sha256"]
    if until == 400:
        errors = []
        for d in (1, 3, 6):
            p = batch(root, d)[0]
            f0, o0 = forward(arms["no_opening"]["model"], p)
            f1, o1 = forward(arms["no_geometry"]["model"], p)
            np.testing.assert_allclose(f0, f1, atol=5e-6, rtol=1e-4)
            np.testing.assert_allclose(o0, o1, atol=5e-6, rtol=1e-4)
            errors.append(
                dict(
                    depth=d,
                    field=float(np.max(abs(np.asarray(f0 - f1)))),
                    pixels=float(np.max(abs(np.asarray(o0 - o1)))),
                )
            )
        checks, gradients = {}, {}
        for name in METHODS:
            parts, grads = arms[name]["gradient"](arms[name]["model"], *b)
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((parts, grads)))
            weight = 0.1 if name == "no_opening" else 0.0
            expected = (
                parts["distance"]
                + weight * parts["geometry_gradient"]
                + 0.25 * parts["sign_margin"]
            )
            np.testing.assert_allclose(parts["loss"], expected, atol=1e-7, rtol=1e-6)
            gn = (
                sum(
                    float(np.sum(np.asarray(v) ** 2)) for v in jax.tree.leaves(grads["pooled_stem"])
                )
                ** 0.5
            )
            assert gn > 0 and np.isfinite(gn)
            checks[name] = dict(pooled_gradient_norm=gn, loss_decomposition_passed=True)
            gradients[name] = grads
        difference = float(
            np.sqrt(
                sum(
                    float(np.sum((np.asarray(a) - np.asarray(b)) ** 2))
                    for a, b in zip(
                        jax.tree.leaves(gradients["no_opening"]),
                        jax.tree.leaves(gradients["no_geometry"]),
                        strict=True,
                    )
                )
            )
        )
        assert difference > 1e-8
        write(
            root / "preflight.json",
            dict(
                passed=True,
                initial_errors=errors,
                gradients=checks,
                gradient_difference_norm=difference,
            ),
        )
    out = root / "training"
    out.mkdir(exist_ok=True)
    journal = out / "metrics.jsonl"
    rows = []
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as manager:
        start = lineart_state.restore(manager, arms) if manager.latest_step() is not None else 0
        if journal.exists():
            rows = [json.loads(s) for s in journal.read_text().splitlines()]
            assert len(rows) == start, (
                "Preserve interruption evidence and explicitly audit replay before retry"
            )
        assert start < until
        if start == 0:
            lineart_state.save(manager, arms, 0)
            snapshot(root, arms, 0)
        for k in range(start, until):
            metrics, seconds = {}, {}
            for name in METHODS:
                tick = time.perf_counter()
                metrics[name] = update(arms[name], *b)
                seconds[name] = time.perf_counter() - tick
                assert all(np.isfinite(v) for v in metrics[name].values())
            row = dict(
                step=k + 1,
                metrics=metrics,
                seconds=seconds,
                input_sha256=read(root / "initial-audit.json")["input_sha256"],
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k == 5:
                replay = arms_from_root(root)
                for j in range(6):
                    for name in METHODS:
                        exact(update(replay[name], *b), rows[j]["metrics"][name])
                exact(lineart_state.payload(replay, 6), lineart_state.payload(arms, 6))
                write(root / "prefix-audit.json", dict(metrics_exact=True, full_state_exact=True))
            if k + 1 in SAVED:
                lineart_state.save(manager, arms, k + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(manager, restored, k + 1) == k + 1
                exact(lineart_state.payload(restored, k + 1), lineart_state.payload(arms, k + 1))
                snapshot(root, arms, k + 1)
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
            if (k + 1) % 50 == 0:
                print(dict(step=k + 1, loss={n: metrics[n]["loss"] for n in METHODS}), flush=True)
                write(out / "progress.json", dict(step=k + 1))
        assert digest(b[:2]) == read(root / "initial-audit.json")["input_sha256"]
        if until < 1200:
            return
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(manager, restored, step) == step
            for name in METHODS:
                for mode in ("raw", "ema"):
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    m, cfg = load_field(folder)
                    arm = restored[name]
                    assert cfg == arm["config"]
                    exact(
                        nnx.state(m, nnx.Param),
                        nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"],
                    )
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(
            root / "snapshot-audit.json",
            dict(full_states_exact=True, raw_ema_exports_exact=20, bindings=bindings),
        )


def evaluate(root):
    check(root)
    with np.load(root / "inputs.npz") as z:
        source, clean = z["source"], z["clean"]
    records, bindings, query, errors = [], {}, [], []
    for depth in (1, 3, 6):
        p = batch(root, depth)[0]
        for name in METHODS:
            values = {}
            for step in SAVED:
                folder = root / f"snapshots/step-{step}/{name}/raw"
                model, _ = load_field(folder)
                state = digest(nnx.state(model, nnx.Param))
                field, pred = forward(model, p)
                field, pred = np.asarray(field), np.asarray(pred)
                assert (
                    np.isfinite(field).all()
                    and np.isfinite(pred).all()
                    and pred.min() >= -1
                    and pred.max() <= 1
                )
                assert state == digest(nnx.state(model, nnx.Param))
                bindings.update({str(f): sha(f) for f in folder.iterdir() if f.is_file()})
                y, x = QUERY
                q = field[:, y * 4 : (y + 1) * 4, x * 4 : (x + 1) * 4]

                query.append(
                    dict(
                        method=name,
                        step=step,
                        depth=depth,
                        pixels=pred[:, y, x, 0].tolist(),
                        fields=q[..., 0].tolist(),
                        max_pair_difference=float(np.max(abs(q[0] - q[1]))),
                    )
                )
                file = root / f"prediction-{name}-{step}-depth{depth}.npz"
                np.savez_compressed(file, field=field, prediction=pred)
                bindings[str(file)] = sha(file)
                values[step] = pred
                if step == 0:
                    with np.load(GO / f"predictions-initial-{depth}.npz") as z:
                        np.testing.assert_allclose(pred, z["prediction"], atol=1e-4, rtol=0)
                        errors.append(
                            dict(
                                method=name,
                                depth=depth,
                                max_abs=float(np.max(abs(pred - z["prediction"]))),
                            )
                        )
            reference = GQ / f"prediction-pooled-1200-depth{depth}.npz"
            with np.load(reference) as z:
                baseline = z["prediction"]
            bindings[str(reference)] = sha(reference)
            show = (400, 800, 1200)
            file = root / f"native-{name}-depth-{depth}.png"
            page(
                file,
                [
                    [source[i], clean[i], baseline[i], *[values[s][i] for s in show]]
                    for i in range(2)
                ],
                [
                    [
                        f"{name} d{depth} " + ("bar" if i == 0 else "fill"),
                        "target",
                        "GQ full loss1200",
                        *[f"step{s}" for s in show],
                    ]
                    for i in range(2)
                ],
            )
            records.append(dict(file=file.name, sha256=sha(file)))
    write(
        root / "metrics.json",
        dict(
            outputs=60,
            reused_reference_outputs=6,
            states_unchanged=True,
            same_pooled_architecture=True,
            initial_cpu_gpu_errors=errors,
            query=query,
            bindings=bindings,
        ),
    )
    write(
        root / "manifest.json",
        dict(records=records, native_scale=True, predictions_unmodified=True),
    )


def finalize(root):
    check(root)
    assert read(root / "test-results.json")["passed"] and read(root / "preflight.json")["passed"]
    assert read(root / "prefix-audit.json")["full_state_exact"]
    snap = read(root / "snapshot-audit.json")
    assert snap["raw_ema_exports_exact"] == 20 and snap["full_states_exact"]
    for f, h in snap["bindings"].items():
        assert sha(Path(f)) == h
    rows = [json.loads(s) for s in (root / "training/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, 1201))
    for r in rows:
        assert r["input_sha256"] == read(root / "initial-audit.json")["input_sha256"]
        assert all(np.isfinite(v) for m in r["metrics"].values() for v in m.values())
    for s in SAVED[1:]:
        assert read(root / f"training/restore-{s}.json")["full_state_exact"]
    metrics = read(root / "metrics.json")
    assert (
        metrics["outputs"] == 60
        and metrics["states_unchanged"]
        and metrics["same_pooled_architecture"]
    )
    for f, h in metrics["bindings"].items():
        assert sha(Path(f)) == h
    records = read(root / "manifest.json")["records"]
    notes = read(root / "observations.json")
    assert len(records) == len(notes) == 6
    for r in records:
        assert notes[r["file"]] and sha(root / r["file"]) == r["sha256"]
    for phase in ("train-400", "train-800", "train-1200", "evaluate"):
        assert read(root / f"driver-{phase}.json")["returncode"] == 0
    summary = {}
    for name in METHODS:
        norms = [r["metrics"][name]["gradient_norm"] for r in rows]
        summary[name] = dict(
            clipped=sum(v > 1 for v in norms),
            max_norm=max(norms),
            mean_norm=float(np.mean(norms)),
            last300_update_seconds=float(np.mean([r["seconds"][name] for r in rows[-300:]])),
        )
    write(root / "diagnostics/integrity.json", summary)
    files = [p for p in root.glob("*.json") if p.name != "completion.json"]
    files += [root / "training/metrics.jsonl", root / "diagnostics/integrity.json"]
    files += list((root / "diagnostics").glob("*.json"))
    write(
        root / "completion.json",
        dict(
            stage="complete_collision_loss_ablation",
            fitting_only=True,
            adoption=False,
            fresh_confirmation=False,
            outputs=60,
            native_pages_reviewed=6,
            evidence={str(p.relative_to(root)): sha(p) for p in set(files)},
        ),
    )
    print(dict(complete=True, integrity=summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--until", type=int)
    a = parser.parse_args()
    if a.phase == "train":
        train(a.root.resolve(), a.until)
    else:
        globals()[a.phase](a.root.resolve())
