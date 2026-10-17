"""GZ: replay GX fitting examples while interleaving ordinary and fill learning."""

import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from consistent_axis_fit import batch as focus_batch
from consistent_axis_fit import page
from consistent_axis_objective import assignment_gradient
from consistent_broad_objective import axis_gradient
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from plausible_study import digest, exact
from pooled_axis_model import infer, load_field, save_field
from pooled_axis_study import GK, GO, GW, GX
from pooled_axis_study import batch as broad_batch
from pooled_axis_study import load_data as broad_data
from scale_style_balance_study import update

from lineartgen.config import save_config
from lineartgen.training import checkpoints, lineart_state

GY = Path("../jax-consistent-axis-broad-20261017/runs/consistent-axis-broad-20261017").resolve()
STEPS = 2400
SAVED = (0, 36, 600, 1200, 1800, 2400)
DOC = Path("docs/axis-replay-mix-plan-20261017.md")


def background_indices():
    rows = [json.loads(x) for x in (GW / "training/sampling.jsonl").read_text().splitlines()]
    return [i for i, r in enumerate(rows) if r["audit"]["family"] != "ambiguous"][:1200]


def select(step, indices):
    i = step // 2
    return (
        ("focus", i, i % 6 + 1) if step % 2 == 0 else ("background", indices[i], indices[i] % 6 + 1)
    )


def check(root):
    p = read(root / "plan.json")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == p["source"]
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(GX / "completion.json")["stage"] == "complete_consistent_axis_fit"
    assert not read(GY / "review-completion.json")["strict_protocol_passed"]
    root.mkdir(parents=True, exist_ok=False)
    (root / "diagnostics").mkdir()
    model, cfg = load_field(GX / "initials/consistent")
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(update=dict(steps=STEPS, learning_rate_decay_steps=STEPS))
        )
    )
    assert (
        cfg.train.peak_learning_rate == 2e-4
        and cfg.train.batch_size == 4
        and cfg.train.accumulate_steps == 1
    )
    assert sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))) == 21890
    save_config(cfg, root / "config.json")
    save_field(model, cfg, root / "initial", provenance="GZ untouched GX/GW initial; research only")
    mm, cc = load_field(root / "initial")
    exact(nnx.state(model, nnx.Param), nnx.state(mm, nnx.Param))
    assert cc == cfg
    for split in ("train", "transfer", "anchors"):
        shutil.copy2(GX / f"{split}.npz", root / f"{split}.npz")
    ids = background_indices()
    assert len(ids) == 1200
    write(root / "background-indices.json", ids)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), source], check=True
    )
    files = [p for p in root.rglob("*") if p.is_file()]
    files += [
        DOC.resolve(),
        GX / "completion.json",
        GX / "diagnostics/supplemental-binding.json",
        GX / "training/sampling.jsonl",
        GY / "review-completion.json",
        GK / "inputs.npz",
        GO / "inputs.npz",
    ]
    files += list((GX / "initials/consistent").iterdir())
    files += list((GX / "evaluation").glob("*.npz"))
    files += list((GY / "diagnostics/geometry").glob("*.npz")) + list(
        (GY / "diagnostics/collision").glob("*.npz")
    )
    files += list((GY / "scene512").glob("*.npz"))
    files += [Path(f) for f in read(GW / "plan.json")["bindings"]]
    files += [
        Path(f).resolve()
        for f in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if f.endswith((".py", ".toml")) or f == "uv.lock"
    ]
    write(
        root / "plan.json",
        dict(
            source=source,
            steps=STEPS,
            focus_updates=1200,
            background_updates=1200,
            saved=SAVED,
            parameters=21890,
            bindings={str(p.resolve()): sha(p) for p in files},
        ),
    )
    print(dict(prepared=True, bindings=len(read(root / "plan.json")["bindings"])), flush=True)


def arm(root):
    m, c = load_field(root / "initial")
    a = make_arm(c, m)
    assert m.head.mode == "choice" and m.head.component == 0
    return a


def data(root):
    with np.load(root / "train.npz") as z:
        focus = {k: z[k] for k in z.files}
    return dict(
        focus=focus,
        broad=broad_data(),
        indices=read(root / "background-indices.json"),
        gx_rows=[json.loads(s) for s in (GX / "training/sampling.jsonl").read_text().splitlines()],
    )


def batch(d, k):
    family, i, depth = select(k, d["indices"])
    if family == "focus":
        b = focus_batch(d["focus"], i)
        assert digest(b) == d["gx_rows"][i]["input_payload_sha256"]
    else:
        b = broad_batch(d["broad"], i)
    assert len(b[0].guides) == depth
    return b, dict(family=family, original_step=i, depth=depth, input_payload_sha256=digest(b))


def gradients():
    return dict(focus=assignment_gradient("consistent"), background=axis_gradient())


def snapshot(root, a, step):
    for mode in ("raw", "ema"):
        m = nnx.clone(a["model"])
        if mode == "ema":
            nnx.update(m, a["ema"])
        save_field(
            m,
            a["config"],
            root / f"snapshots/step-{step}/{mode}",
            provenance=f"GZ mixed step{step} {mode}; not production",
        )


def train(root):
    check(root)
    assert jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    a = arm(root)
    arms = {"mixed": a}
    d = data(root)
    grad = gradients()
    out = root / "training"
    out.mkdir(exist_ok=True)
    journal = out / "sampling.jsonl"
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as manager:
        start = lineart_state.restore(manager, arms) if manager.latest_step() is not None else 0
        rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        assert len(rows) == start, "Preserve and audit lost updates before resuming"
        if start == 0:
            old, cfg = load_field(GX / "initials/consistent")
            control = make_arm(cfg, old)
            control["gradient"] = grad["focus"]
            exact(nnx.state(old, nnx.Param), nnx.state(a["model"], nnx.Param))
            rates = []
            for i in range(1200):
                aa = float(a["schedule"](2 * i))
                bb = float(control["schedule"](i))
                np.testing.assert_allclose(aa, bb, atol=2e-11, rtol=1e-6)
                rates.append(abs(aa - bb))
            differences = []
            for i in range(18):
                m = update(control, *focus_batch(d["focus"], i))
                oldm = d["gx_rows"][i]["metrics"]["consistent"]
                for key, v in m.items():
                    np.testing.assert_allclose(v, oldm[key], atol=1e-6, rtol=2e-5)
                differences.append(max(abs(v - oldm[key]) for key, v in m.items()))
            for k in range(12):
                b, info = batch(d, k)
                parts, g = grad[info["family"]](a["model"], *b)
                assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((parts, g)))
                assert sum(float(np.square(np.asarray(x)).sum()) for x in jax.tree.leaves(g)) > 0
            write(
                root / "preflight.json",
                dict(
                    passed=True,
                    initial_state_exact=True,
                    focus_schedule_max_abs=max(rates),
                    gx18_metric_max_abs=max(differences),
                    cross_run_atol=1e-6,
                    cross_run_rtol=2e-5,
                ),
            )
            lineart_state.save(manager, arms, 0)
            snapshot(root, a, 0)
        for k in range(start, STEPS):
            b, info = batch(d, k)
            a["gradient"] = grad[info["family"]]
            tick = time.perf_counter()
            metrics = update(a, *b)
            seconds = time.perf_counter() - tick
            row = dict(
                step=k + 1,
                **info,
                metrics=metrics,
                seconds=seconds,
                learning_rate=float(a["schedule"](k)),
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k == 35:
                re = arm(root)
                for j in range(36):
                    rb, ri = batch(d, j)
                    re["gradient"] = grad[ri["family"]]
                    exact(update(re, *rb), rows[j]["metrics"])
                exact(lineart_state.payload({"mixed": re}, 36), lineart_state.payload(arms, 36))
                write(
                    root / "prefix-audit.json",
                    dict(passed=True, updates=36, metrics_exact=True, full_state_exact=True),
                )
            if k + 1 in SAVED:
                lineart_state.save(manager, arms, k + 1)
                restored = {"mixed": arm(root)}
                assert lineart_state.restore(manager, restored, k + 1) == k + 1
                exact(lineart_state.payload(restored, k + 1), lineart_state.payload(arms, k + 1))
                snapshot(root, a, k + 1)
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
            if (k + 1) % 50 == 0:
                write(out / "progress.json", dict(step=k + 1))
                print(dict(step=k + 1), flush=True)
        bindings = {}
        for step in SAVED:
            restored = {"mixed": arm(root)}
            assert lineart_state.restore(manager, restored, step) == step
            for mode in ("raw", "ema"):
                folder = root / f"snapshots/step-{step}/{mode}"
                m, c = load_field(folder)
                expected = (
                    nnx.state(restored["mixed"]["model"], nnx.Param)
                    if mode == "raw"
                    else restored["mixed"]["ema"]
                )
                exact(nnx.state(m, nnx.Param), expected)
                assert c == a["config"]
                bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(exports=12, raw_ema_exact=True, bindings=bindings))
    print(dict(training_complete=True), flush=True)


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    m, _ = load_field(root / f"snapshots/step-{STEPS}/raw")
    state = digest(nnx.state(m, nnx.Param))
    records = []
    bindings = {}
    count = 0
    references = 0

    def predict(source, target, strength, seed, depth):
        nonlocal count
        p = prepare_inputs(target, source, strength, jax.random.PRNGKey(seed), depth)
        v = np.asarray(infer(m, p))
        assert np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1
        np.testing.assert_array_equal(v[strength == 0], source[strength == 0])
        count += len(source)
        return v

    def save(name, values):
        p = out / name
        np.savez_compressed(p, **values)
        bindings[str(p)] = sha(p)

    def draw(name, rows, labels):
        p = out / name
        page(p, rows, labels)
        records.append(dict(file=name, sha256=sha(p)))

    for split in ("train", "transfer", "anchors"):
        with np.load(root / f"{split}.npz") as z:
            d = {k: z[k] for k in z.files}
        for depth in (1, 3, 6):
            pred = np.concatenate(
                [
                    predict(v[None], v[None], np.full_like(v[None], 0.5), 2026101711, depth)
                    for v in d["source"]
                ]
            )
            ref = GX / f"evaluation/{split}-depth{depth}.npz"
            bindings[str(ref)] = sha(ref)
            with np.load(ref) as z:
                initial = z["initial"]
                focused = z["consistent"]
                references += len(initial) + len(focused)
            save(f"{split}-depth{depth}.npz", dict(initial=initial, focused=focused, mixed=pred))
            if depth == 3:
                continue
            for start in range(0, len(pred), 4):
                rows = []
                labels = []
                for i in range(start, min(start + 4, len(pred))):
                    row = [d["source"][i]]
                    names = [f"{split}{i} d{depth} source"]
                    if split != "anchors":
                        row.append(d["preferred_clean"][i])
                        names.append("admissible target")
                    rows.append(row + [initial[i], focused[i], pred[i]])
                    labels.append(names + ["initial", "GX focused1200", "GZ mixed2400"])
                draw(f"native-{split}-{depth}-{start}.png", rows, labels)
    for group, path in [("geometry", GK / "inputs.npz"), ("collision", GO / "inputs.npz")]:
        bindings[str(path)] = sha(path)
        with np.load(path) as z:
            source = z["source"]
            target = z["target"] if group == "geometry" else z["clean"]
        if group == "geometry":
            source = source[:, 0]
            target = target[:, 0]
        for depth in (1, 3, 6):
            pred = np.concatenate(
                [
                    predict(
                        s[None],
                        t[None],
                        np.full_like(s[None], 0.5),
                        2026101701 if group == "geometry" else 2026101704,
                        depth,
                    )
                    for s, t in zip(source, target, strict=True)
                ]
            )
            old = []
            if group == "geometry":
                for i in range(8):
                    ref = GY / f"diagnostics/{group}/predictions-{depth}-{i}.npz"
                    bindings[str(ref)] = sha(ref)
                    with np.load(ref) as z:
                        old.append(z["consistent"][0])
                old = np.stack(old)
            else:
                ref = GY / f"diagnostics/{group}/predictions-{depth}.npz"
                bindings[str(ref)] = sha(ref)
                with np.load(ref) as z:
                    old = z["consistent"]
            references += len(old)
            save(f"{group}-depth{depth}.npz", dict(broad=old, mixed=pred))
            if depth == 3:
                continue
            for start in range(0, len(pred), 4):
                ids = range(start, min(start + 4, len(pred)))
                draw(
                    f"native-{group}-{depth}-{start}.png",
                    [[source[i], target[i], old[i], pred[i]] for i in ids],
                    [
                        [f"{group}{i} d{depth} source", "target", "GY broad6000", "GZ mixed2400"]
                        for i in ids
                    ],
                )
    from PIL import Image, ImageDraw

    yy, xx = np.indices((512, 512))
    for i in range(8):
        inp = GY / f"scene512/input-{i}.npz"
        bindings[str(inp)] = sha(inp)
        with np.load(inp) as z:
            source = z["source"]
            quantized = z["quantized"]
            clean = z["clean"]
        for mode in ("whole", "partial"):
            strength = (
                np.full_like(source, 0.5)
                if mode == "whole"
                else np.where((xx > 512 * 0.6 - yy * 0.25)[..., None], 0.5, 0).astype(np.float32)
            )
            v = {
                encoding: predict(s[None], clean[None], strength[None], 2026101319, 6)[0]
                for encoding, s in [("float", source), ("uint8", quantized)]
            }
            ref = GY / f"scene512/predictions-{i}-{mode}-0.npz"
            bindings[str(ref)] = sha(ref)
            with np.load(ref) as z:
                old = z["consistent_uint8"]
                references += 1
            save(
                f"scene-{i}-{mode}.npz",
                dict(broad_uint8=old, mixed_float=v["float"], mixed_uint8=v["uint8"]),
            )
            if mode == "partial" and i not in (1, 3, 5, 7):
                continue
            canvas = Image.new("RGB", (2048, 548), "white")
            pen = ImageDraw.Draw(canvas)
            for j, (label, value) in enumerate(
                zip(
                    ("source uint8", "GY broad uint8", "GZ mixed uint8", "GZ mixed float"),
                    (quantized, old, v["uint8"], v["float"]),
                    strict=True,
                )
            ):
                px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(Image.fromarray(px).convert("RGB"), (j * 512, 36))
                pen.text((j * 512 + 3, 5), f"scene{i} {mode} {label}", fill="black")
            p = out / f"native-scene-{i}-{mode}.png"
            canvas.save(p)
            records.append(dict(file=p.name, sha256=sha(p)))
    assert count == 134 and references == 190 and len(records) == 30
    assert state == digest(nnx.state(m, nnx.Param))
    write(out / "manifest.json", dict(records=records, native_scale=True))
    write(
        out / "metrics.json",
        dict(
            new_outputs=count,
            archived_reference_outputs=references,
            states_unchanged=True,
            protected_exact=True,
            bindings=bindings,
        ),
    )
    print(dict(outputs=count, references=references, pages=len(records)), flush=True)


def finalize(root):
    check(root)
    for name in ("test-results", "preflight", "prefix-audit"):
        assert read(root / f"{name}.json")["passed"]
    a = read(root / "snapshot-audit.json")
    assert a["exports"] == 12 and a["raw_ema_exact"]
    for p, h in a["bindings"].items():
        assert sha(Path(p)) == h
    for s in SAVED[1:]:
        assert read(root / f"training/restore-{s}.json")["full_state_exact"]
    rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert len(rows) == STEPS
    ids = read(root / "background-indices.json")
    gx = [json.loads(s) for s in (GX / "training/sampling.jsonl").read_text().splitlines()]
    for k, r in enumerate(rows):
        f, i, d = select(k, ids)
        assert (r["family"], r["original_step"], r["depth"]) == (f, i, d)
        if f == "focus":
            assert r["input_payload_sha256"] == gx[i]["input_payload_sha256"]
        assert all(np.isfinite(v) for v in r["metrics"].values())
    ev = read(root / "evaluation/metrics.json")
    assert ev["new_outputs"] == 134 and ev["states_unchanged"] and ev["protected_exact"]
    for p, h in ev["bindings"].items():
        assert sha(Path(p)) == h
    records = read(root / "evaluation/manifest.json")["records"]
    notes = read(root / "evaluation/observations.json")
    assert len(records) == len(notes) == 30
    for r in records:
        assert notes[r["file"]] and sha(root / "evaluation" / r["file"]) == r["sha256"]
    assert not read(root / "decision.json")["adoption"]
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".jsonl", ".py", ".log")
        and "checkpoints" not in p.parts
        and p.name not in ("completion.json", "supplemental-binding.json")
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_axis_replay_mix",
            steps=STEPS,
            focus_updates=1200,
            background_updates=1200,
            output_pages_reviewed=30,
            new_outputs=134,
            archived_reference_outputs=190,
            adoption=False,
            fresh_confirmation=False,
            evidence={str(p.relative_to(root)): sha(p) for p in files},
        ),
    )
    print(dict(complete=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "train", "evaluate", "finalize"))
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "train":
        with (root / "process.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root)
    else:
        globals()[a.action](root)
