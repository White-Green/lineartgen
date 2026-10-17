"""HB remote-mask pair: whole-only versus support-aware consistent orientation."""

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ambiguous_shape_data import render
from campaign_protocol import read, sha, write
from consistent_axis_objective import candidate_mask
from context_dropout_objective import context_mask
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from local_axis_support import gradient_for, support_mask, support_safe
from PIL import Image, ImageDraw
from plausible_study import digest, exact
from pooled_axis_model import infer, load_field, save_field
from scale_style_balance_study import update
from subpixel_field_data import candidate_fields

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

GW = Path("../jax-pooled-axis-20261017/runs/pooled-axis-20261017-retry3").resolve()
METHODS = ("global", "local")
STEPS = 600
SAVED = (0, 18, 200, 400, 600)
DOC = Path("docs/local-axis-support-fit-plan-20261017.md")


def page(path, rows, labels):
    canvas = Image.new("RGB", (256 * len(rows[0]), 292 * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (values, names) in enumerate(zip(rows, labels, strict=True)):
        for j, (value, name) in enumerate(zip(values, names, strict=True)):
            px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
            canvas.paste(Image.fromarray(px).convert("RGB"), (256 * j, 292 * i + 36))
            draw.text((256 * j + 3, 292 * i + 5), name, fill="black")
    canvas.save(path)


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(GW / "completion.json")["stage"] == "complete_pooled_axis"
    root.mkdir(parents=True, exist_ok=False)
    (root / "teacher-preview").mkdir()
    model, cfg = load_field(GW / "initials/choice")
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=STEPS,
                    learning_rate_decay_steps=STEPS,
                    batch_size=2,
                    accumulate_steps=1,
                    learning_rate=1e-4,
                    minimum_learning_rate=1e-5 * np.sqrt(2),
                    warmup_steps=0,
                )
            )
        )
    )
    assert cfg.train.peak_learning_rate == 1e-4 * np.sqrt(2)
    assert sum(a.size for a in jax.tree.leaves(nnx.state(model, nnx.Param))) == 21890
    save_config(cfg, root / "config.json")
    for name in METHODS:
        save_field(
            model,
            cfg,
            root / f"initials/{name}",
            provenance="HC untouched GW initial choice; research only",
        )
        restored, cc = load_field(root / f"initials/{name}")
        exact(nnx.state(restored, nnx.Param), nnx.state(model, nnx.Param))
        assert cc == cfg
    manifest = []
    recipes = [
        dict(center=[208.0, 208.0], threshold=None, split="train", name="whole"),
        dict(center=[208.0, 208.0], threshold=153.6, split="train", name="partial"),
        dict(center=[208.0, 208.0], threshold=120.0, split="transfer", name="remote120"),
        dict(center=[208.0, 208.0], threshold=170.0, split="transfer", name="remote170"),
        dict(center=[192.0, 192.0], threshold=153.6, split="transfer", name="moved192"),
        dict(center=[128.0, 128.0], threshold=153.6, split="transfer", name="cut128"),
    ]
    yy, xx = np.mgrid[:256, :256]
    for split in ("train", "transfer"):
        data = []
        rr = [r for r in recipes if r["split"] == split]
        for r in rr:
            a = render(
                dict(
                    width=48.0,
                    length=48.0,
                    angle=0.0,
                    sigma=1.2,
                    opacity=0.4,
                    filled=False,
                    center=r["center"],
                )
            )
            t = (
                np.full((256, 256), 0.5, np.float32)
                if r["threshold"] is None
                else (xx + 0.3 * yy >= r["threshold"]).astype(np.float32) * 0.5
            )[..., None]
            a["strength"] = t
            targets = jnp.concatenate(
                tuple(
                    candidate_fields(jnp.asarray(a[k][None]), jnp.asarray(a[n][None]), 1)[0]
                    for k, n in (("base", "direction"), ("axis_base", "axis_direction"))
                ),
                axis=1,
            )
            safe = bool(support_safe(targets, jnp.asarray(t[None]))[0])
            assert safe == (r["name"] != "cut128")
            for policy, mask in (
                (
                    "global",
                    candidate_mask(
                        jnp.asarray(a["direction"][None]), jnp.asarray(t[None]), "consistent"
                    ),
                ),
                (
                    "local",
                    support_mask(jnp.asarray(a["direction"][None]), targets, jnp.asarray(t[None])),
                ),
            ):
                alt = bool(mask[0, 9])
                a[policy + "_clean"] = np.where(
                    t > 0, a["axis_clean" if alt else "clean"], a["source"]
                )
            r["support_safe"] = safe
            data.append(a)
        stacked = {k: np.stack([a[k] for a in data]) for k in data[0]}
        np.savez_compressed(root / f"{split}.npz", **stacked)
        write(root / f"{split}-recipes.json", rr)
        f = root / f"teacher-preview/{split}.png"
        page(
            f,
            [[a[k] for k in ("source", "global_clean", "local_clean")] for a in data],
            [[r["name"] + " source", "global permitted", "local permitted"] for r in rr],
        )
        manifest.append(dict(file=f.name, sha256=sha(f)))
    # The fitting source pair must be exactly HB's audited example.
    hb = Path("../jax-partial-axis-context-20261017/runs/partial-axis-context-20261017").resolve()
    with np.load(hb / "inputs.npz") as z, np.load(root / "train.npz") as a:
        for key in ("source", "clean", "base", "axis_base", "direction", "axis_direction"):
            np.testing.assert_array_equal(a[key][0], z[key])
            np.testing.assert_array_equal(a[key][1], z[key])
        np.testing.assert_array_equal(a["strength"][0], z["whole"])
        np.testing.assert_array_equal(a["strength"][1], z["partial"])
    write(root / "teacher-preview/manifest.json", dict(records=manifest, native_scale=True))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with (root / "source.tar.gz").open("wb") as f:
        subprocess.run(["git", "archive", "--format=tar.gz", source], stdout=f, check=True)
    files = [p for p in root.rglob("*") if p.is_file()] + [
        DOC,
        GW / "completion.json",
        GW / "diagnostics/supplemental-binding.json",
        hb / "inputs.npz",
        hb / "completion.json",
    ]
    files += list((GW / "initials/choice").glob("*"))
    files += [
        Path(p).resolve()
        for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if p.endswith((".py", ".toml")) or p == "uv.lock"
    ]
    write(
        root / "plan.json",
        dict(
            source=source,
            steps=STEPS,
            saved=SAVED,
            parameters=21890,
            bindings={str(p.resolve()): sha(p) for p in files},
        ),
    )
    print(dict(prepared=True, train=2, transfer=4, preview_pages=2), flush=True)


def check(root):
    p = read(root / "plan.json")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == p["source"]
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f


def arms_from_root(root):
    result = {}
    for name in METHODS:
        m, c = load_field(root / f"initials/{name}")
        assert (
            c == load_config(root / "config.json")
            and m.head.mode == "choice"
            and m.head.component == 0
        )
        result[name] = make_arm(c, m)
        result[name]["gradient"] = gradient_for(name)
    return result


def load_data(root):
    with np.load(root / "train.npz") as z:
        return {k: z[k] for k in z.files}


def batch(data, step):
    depth = step % 6 + 1
    ids = [0, 1]
    source = jnp.asarray(data["source"][ids])
    clean = jnp.asarray(data["clean"][ids])
    prepared = prepare_inputs(
        clean,
        source,
        jnp.asarray(data["strength"][ids]),
        jax.random.fold_in(jax.random.PRNGKey(2026101713), step),
        depth,
    )
    payload = (
        jnp.asarray(data["base"][ids]),
        jnp.asarray(data["axis_base"][ids]),
        jnp.asarray(data["axis_direction"][ids]),
        jnp.asarray(context_mask(step, depth, 2, enabled=True)),
    )
    return prepared, payload, jnp.asarray(data["direction"][ids])


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            m = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(m, arm["ema"])
            save_field(
                m,
                arm["config"],
                root / f"snapshots/step-{step}/{name}/{mode}",
                provenance=f"HC {name} step{step} {mode}; diagnostic, not production",
            )


def train(root):
    check(root)
    assert jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    preview = read(root / "teacher-preview/observations.json")
    for a in read(root / "teacher-preview/manifest.json")["records"]:
        assert preview[a["file"]]
    data = load_data(root)
    arms = arms_from_root(root)
    out = root / "training"
    out.mkdir(exist_ok=True)
    journal = out / "sampling.jsonl"
    rows = []
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as manager:
        start = lineart_state.restore(manager, arms) if manager.latest_step() is not None else 0
        if journal.exists():
            rows = [json.loads(s) for s in journal.read_text().splitlines()]
            assert len(rows) == start, "Archive and audit unsaved updates before resuming"
        assert start < STEPS
        if start == 0:
            initial = []
            for depth in range(1, 7):
                b = batch(data, depth - 1)
                a = np.asarray(infer(arms["global"]["model"], b[0]))
                bb = np.asarray(infer(arms["local"]["model"], b[0]))
                np.testing.assert_array_equal(a, bb)
                gradients = {}
                for name in METHODS:
                    parts, g = arms[name]["gradient"](arms[name]["model"], *b)
                    assert all(
                        np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((parts, g))
                    )
                    gradients[name] = float(
                        sum(np.square(np.asarray(v)).sum() for v in jax.tree.leaves(g)) ** 0.5
                    )
                    assert gradients[name] > 0
                initial.append(dict(depth=depth, outputs_exact=True, gradient_norms=gradients))
            write(root / "preflight.json", dict(passed=True, initial=initial))
            lineart_state.save(manager, arms, 0)
            snapshot(root, arms, 0)
        for k in range(start, STEPS):
            b = batch(data, k)
            metrics = {}
            seconds = {}
            for name in METHODS:
                tick = time.perf_counter()
                metrics[name] = update(arms[name], *b)
                seconds[name] = time.perf_counter() - tick
            row = dict(
                step=k + 1,
                depth=k % 6 + 1,
                indices=[0, 1],
                metrics=metrics,
                seconds=seconds,
                input_payload_sha256=digest(b),
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k == 17:
                replay = arms_from_root(root)
                for j in range(18):
                    for name in METHODS:
                        exact(update(replay[name], *batch(data, j)), rows[j]["metrics"][name])
                exact(lineart_state.payload(replay, 18), lineart_state.payload(arms, 18))
                write(
                    root / "prefix-audit.json",
                    dict(passed=True, updates=18, metrics_exact=True, full_state_exact=True),
                )
            if k + 1 in SAVED:
                lineart_state.save(manager, arms, k + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(manager, restored, k + 1) == k + 1
                exact(lineart_state.payload(restored, k + 1), lineart_state.payload(arms, k + 1))
                snapshot(root, arms, k + 1)
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
            if (k + 1) % 50 == 0:
                write(out / "progress.json", dict(step=k + 1))
                print(dict(step=k + 1), flush=True)
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(manager, restored, step) == step
            for name in METHODS:
                for mode in ("raw", "ema"):
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    m, c = load_field(folder)
                    expected = (
                        nnx.state(restored[name]["model"], nnx.Param)
                        if mode == "raw"
                        else restored[name]["ema"]
                    )
                    exact(nnx.state(m, nnx.Param), expected)
                    assert c == restored[name]["config"]
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(exports=20, raw_ema_exact=True, bindings=bindings))


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    paths = {"initial": root / "initials/global"} | {
        m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS
    }
    models = {m: load_field(p)[0] for m, p in paths.items()}
    states = {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
    records = []
    bindings = {}
    outputs = 0
    for split in ("train", "transfer"):
        p = root / f"{split}.npz"
        bindings[str(p)] = sha(p)
        with np.load(p) as z:
            data = {k: z[k] for k in z.files}
        count = len(data["source"])
        for depth in (1, 3, 6):
            predictions = {m: [] for m in models}
            for i in range(count):
                source = data["source"][i : i + 1]
                b = prepare_inputs(
                    source,
                    source,
                    data["strength"][i : i + 1],
                    jax.random.PRNGKey(2026101714),
                    depth,
                )
                for m, model in models.items():
                    value = np.asarray(infer(model, b))[0]
                    assert np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1
                    protected = data["strength"][i] == 0
                    np.testing.assert_array_equal(value[protected], source[0][protected])
                    predictions[m].append(value)
                    outputs += 1
            p = out / f"{split}-depth{depth}.npz"
            np.savez_compressed(p, **{m: np.stack(v) for m, v in predictions.items()})
            bindings[str(p)] = sha(p)
            for start in range(0, count, 4):
                ids = range(start, min(start + 4, count))
                rows = []
                labels = []
                for i in ids:
                    values = [data["source"][i]]
                    names = [f"{split}{i} d{depth} source"]
                    for policy in METHODS:
                        values.append(data[policy + "_clean"][i])
                        names.append(policy + " permitted")
                    for m in models:
                        values.append(predictions[m][i])
                        names.append(m)
                    rows.append(values)
                    labels.append(names)
                p = out / f"native-{split}-{depth}-{start}.png"
                page(p, rows, labels)
                records.append(dict(file=p.name, sha256=sha(p)))
    assert outputs == 54 and len(records) == 6
    assert states == {m: digest(nnx.state(v, nnx.Param)) for m, v in models.items()}
    for folder in paths.values():
        bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
    write(out / "manifest.json", dict(records=records, native_scale=True))
    write(out / "metrics.json", dict(outputs=outputs, states_unchanged=True, bindings=bindings))
    print(dict(outputs=outputs, pages=len(records)), flush=True)


def finalize(root):
    check(root)
    for name in ("test-results", "preflight", "prefix-audit"):
        assert read(root / f"{name}.json")["passed"]
    a = read(root / "snapshot-audit.json")
    assert a["exports"] == 20 and a["raw_ema_exact"]
    for f, h in a["bindings"].items():
        assert sha(Path(f)) == h
    for step in SAVED[1:]:
        assert read(root / f"training/restore-{step}.json")["full_state_exact"]
    rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert len(rows) == STEPS
    for k, a in enumerate(rows):
        assert a["step"] == k + 1 and a["depth"] == k % 6 + 1 and a["indices"] == [0, 1]
        assert all(np.isfinite(v) for row in a["metrics"].values() for v in row.values())
    for folder, count in (("teacher-preview", 2), ("evaluation", 6)):
        records = read(root / f"{folder}/manifest.json")["records"]
        notes = read(root / f"{folder}/observations.json")
        assert len(records) == len(notes) == count
        for r in records:
            assert notes[r["file"]] and sha(root / folder / r["file"]) == r["sha256"]
    ev = read(root / "evaluation/metrics.json")
    assert ev["outputs"] == 54 and ev["states_unchanged"]
    for f, h in ev["bindings"].items():
        assert sha(Path(f)) == h
    assert read(root / "decision.json")["adoption"] is False
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".jsonl", ".png", ".py", ".log")
        and p.name != "completion.json"
        and "checkpoints" not in p.parts
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_local_axis_support_fit",
            steps=STEPS,
            outputs=54,
            teacher_pages_reviewed=2,
            output_pages_reviewed=6,
            adoption=False,
            fresh_confirmation=False,
            evidence={str(p.relative_to(root)): sha(p) for p in files},
        ),
    )
    print(dict(complete=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "train", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "train":
        with (root / "process.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root)
    else:
        globals()[a.action](root)
