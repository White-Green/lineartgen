"""HA: matched simultaneous mean versus conflict-projected task gradients."""

import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from axis_replay_mix import GK, GO, GX, GY, batch, data, gradients
from campaign_protocol import read, sha, write
from cue_capacity import make_arm, prepare_inputs
from flax import nnx
from pair_gradient_projection import combine
from plausible_study import digest, exact
from pooled_axis_model import infer, load_field, save_field

from lineartgen.config import save_config
from lineartgen.training import checkpoints, lineart_state

GZ = Path("../jax-axis-replay-mix-20261017/runs/axis-replay-mix-20261017").resolve()
METHODS = ("average", "projected")
STEPS = 1200
SAVED = (0, 18, 400, 800, 1200)
DOC = Path("docs/axis-gradient-projection-plan-20261017.md")


def check(root):
    p = read(root / "plan.json")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == p["source"]
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h, f


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(GZ / "completion.json")["stage"] == "complete_axis_replay_mix"
    root.mkdir(parents=True, exist_ok=False)
    (root / "diagnostics").mkdir()
    model, cfg = load_field(GZ / "initial")
    cfg = cfg.model_copy(
        update=dict(
            train=cfg.train.model_copy(
                update=dict(
                    steps=STEPS,
                    learning_rate_decay_steps=STEPS,
                    batch_size=8,
                    accumulate_steps=1,
                    minimum_learning_rate=1e-5 * np.sqrt(8),
                )
            )
        )
    )
    assert cfg.train.peak_learning_rate == 1e-4 * np.sqrt(8)
    assert sum(v.size for v in jax.tree.leaves(nnx.state(model, nnx.Param))) == 21890
    save_config(cfg, root / "config.json")
    for name in METHODS:
        save_field(
            model,
            cfg,
            root / f"initials/{name}",
            provenance="HA untouched GZ/GX/GW initial; research only",
        )
        restored, c = load_field(root / f"initials/{name}")
        exact(nnx.state(model, nnx.Param), nnx.state(restored, nnx.Param))
        assert c == cfg
    for name in ("train.npz", "transfer.npz", "anchors.npz", "background-indices.json"):
        shutil.copy2(GZ / name, root / name)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), source], check=True
    )
    files = [p for p in root.rglob("*") if p.is_file()] + [
        DOC.resolve(),
        GZ / "completion.json",
        GZ / "diagnostics/supplemental-binding.json",
        GZ / "training/sampling.jsonl",
    ]
    files += [Path(p) for p in read(GZ / "plan.json")["bindings"]]
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
            effective_batch=8,
            bindings={str(p.resolve()): sha(p) for p in files},
        ),
    )
    print(dict(prepared=True, bindings=len(read(root / "plan.json")["bindings"])), flush=True)


def arms_from_root(root):
    arms = {}
    for name in METHODS:
        model, cfg = load_field(root / f"initials/{name}")
        assert model.head.mode == "choice" and model.head.component == 0
        arms[name] = make_arm(cfg, model)
    return arms


def paired_update(a, batches, grad, method):
    f, gf = grad["focus"](a["model"], *batches[0])
    b, gb = grad["background"](a["model"], *batches[1])
    g, stats = combine(gf, gb, project=method == "projected")
    metrics = (
        {f"focus_{k}": float(v) for k, v in f.items()}
        | {f"background_{k}": float(v) for k, v in b.items()}
        | {k: float(v) for k, v in stats.items()}
    )
    metrics["loss"] = float((f["loss"] + b["loss"]) * 0.5)
    assert all(np.isfinite(v) for v in metrics.values())
    _, a["ema"] = a["update"](a["model"], a["optimizer"], g, a["ema"])
    jax.block_until_ready(a["ema"])
    return metrics


def snapshot(root, arms, step):
    for name, a in arms.items():
        for mode in ("raw", "ema"):
            m = nnx.clone(a["model"])
            if mode == "ema":
                nnx.update(m, a["ema"])
            save_field(
                m,
                a["config"],
                root / f"snapshots/step-{step}/{name}/{mode}",
                provenance=f"HA {name} step{step} {mode}; research only",
            )


def train(root):
    check(root)
    assert jax.devices()[0].platform != "cpu"
    assert read(root / "test-results.json")["passed"]
    arms = arms_from_root(root)
    d = data(root)
    grad = gradients()
    previous = [json.loads(s) for s in (GZ / "training/sampling.jsonl").read_text().splitlines()]

    def pair(k):
        results = []
        info = []
        for j in (2 * k, 2 * k + 1):
            v, r = batch(d, j)
            assert r["input_payload_sha256"] == previous[j]["input_payload_sha256"]
            results.append(v)
            info.append(r)
        assert info[0]["depth"] == info[1]["depth"] == k % 6 + 1
        return results, info

    out = root / "training"
    out.mkdir(exist_ok=True)
    journal = out / "sampling.jsonl"
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as manager:
        start = lineart_state.restore(manager, arms) if manager.latest_step() is not None else 0
        rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        assert len(rows) == start, "Preserve and audit unsaved rows before resuming"
        if start == 0:
            exact(
                nnx.state(arms["average"]["model"], nnx.Param),
                nnx.state(arms["projected"]["model"], nnx.Param),
            )
            initial = []
            for k in range(6):
                bs, _ = pair(k)
                for b in bs:
                    np.testing.assert_array_equal(
                        np.asarray(infer(arms["average"]["model"], b[0])),
                        np.asarray(infer(arms["projected"]["model"], b[0])),
                    )
                f, gf = grad["focus"](arms["average"]["model"], *bs[0])
                b, gb = grad["background"](arms["average"]["model"], *bs[1])
                _, stats = combine(gf, gb, project=True)
                initial.append({a: float(v) for a, v in stats.items()})
                assert all(np.isfinite(v) for v in initial[-1].values())
            write(
                root / "preflight.json",
                dict(
                    passed=True, initial_state_exact=True, initial_pixels_exact=True, depths=initial
                ),
            )
            lineart_state.save(manager, arms, 0)
            snapshot(root, arms, 0)
        for k in range(start, STEPS):
            bs, info = pair(k)
            metrics = {}
            seconds = {}
            for name in METHODS:
                tick = time.perf_counter()
                metrics[name] = paired_update(arms[name], bs, grad, name)
                seconds[name] = time.perf_counter() - tick
            row = dict(step=k + 1, depth=k % 6 + 1, inputs=info, metrics=metrics, seconds=seconds)
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k == 17:
                replay = arms_from_root(root)
                for i in range(18):
                    rb, _ = pair(i)
                    for name in METHODS:
                        exact(paired_update(replay[name], rb, grad, name), rows[i]["metrics"][name])
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
                    assert c == arms[name]["config"]
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(exports=20, raw_ema_exact=True, bindings=bindings))
    print(dict(training_complete=True), flush=True)


def native_page(path, rows, labels):
    from PIL import Image, ImageDraw

    h, w = rows[0][0].shape[:2]
    canvas = Image.new("RGB", (w * len(rows[0]), (h + 36) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (values, names) in enumerate(zip(rows, labels, strict=True)):
        for j, (v, name) in enumerate(zip(values, names, strict=True)):
            assert v.shape[:2] == (h, w)
            px = np.uint8(np.rint(np.clip((v[..., 0] + 1) * 127.5, 0, 255)))
            canvas.paste(Image.fromarray(px).convert("RGB"), (j * w, i * (h + 36) + 36))
            draw.text((j * w + 3, i * (h + 36) + 5), name, fill="black")
    canvas.save(path)


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    models = {name: load_field(root / f"snapshots/step-{STEPS}/{name}/raw")[0] for name in METHODS}
    states = {name: digest(nnx.state(m, nnx.Param)) for name, m in models.items()}
    records = []
    bindings = {}
    count = 0
    references = 0

    def predict(source, target, strength, seed, depth):
        nonlocal count
        prepared = prepare_inputs(target, source, strength, jax.random.PRNGKey(seed), depth)
        result = {}
        for name, m in models.items():
            v = np.asarray(infer(m, prepared))
            assert np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1
            np.testing.assert_array_equal(v[strength == 0], source[strength == 0])
            count += len(source)
            result[name] = v
        return result

    def save(name, values):
        p = out / name
        np.savez_compressed(p, **values)
        bindings[str(p)] = sha(p)

    def draw(name, rows, labels):
        p = out / name
        native_page(p, rows, labels)
        records.append(dict(file=name, sha256=sha(p)))

    def collect(source, target, seed, depth):
        values = [
            predict(s[None], t[None], np.full_like(s[None], 0.5), seed, depth)
            for s, t in zip(source, target, strict=True)
        ]
        return {m: np.concatenate([v[m] for v in values]) for m in METHODS}

    for split in ("train", "transfer", "anchors"):
        with np.load(root / f"{split}.npz") as z:
            d = {k: z[k] for k in z.files}
        source = d["source"]
        for depth in (1, 3, 6):
            pred = collect(source, source, 2026101711, depth)
            ref = GX / f"evaluation/{split}-depth{depth}.npz"
            bindings[str(ref)] = sha(ref)
            with np.load(ref) as z:
                focused = z["consistent"]
                references += len(focused)
            save(f"{split}-depth{depth}.npz", dict(focused=focused, **pred))
            if depth == 3:
                continue
            for start in range(0, len(source), 4):
                rows = []
                labels = []
                for i in range(start, min(start + 4, len(source))):
                    values = [source[i]]
                    names = [f"{split}{i} d{depth} source"]
                    if split != "anchors":
                        values.append(d["preferred_clean"][i])
                        names.append("admissible target")
                    rows.append(values + [focused[i], *[pred[m][i] for m in METHODS]])
                    labels.append(names + ["GX focused1200", "HA average1200", "HA projected1200"])
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
            pred = collect(source, target, 2026101701 if group == "geometry" else 2026101704, depth)
            if group == "geometry":
                old = []
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
            save(f"{group}-depth{depth}.npz", dict(broad=old, **pred))
            if depth == 3:
                continue
            for start in range(0, len(source), 4):
                ids = range(start, min(start + 4, len(source)))
                draw(
                    f"native-{group}-{depth}-{start}.png",
                    [[source[i], target[i], old[i], *[pred[m][i] for m in METHODS]] for i in ids],
                    [
                        [f"{group}{i} d{depth} source", "target", "GY broad6000", *METHODS]
                        for i in ids
                    ],
                )
    yy, xx = np.indices((512, 512))
    for i in range(8):
        ip = GY / f"scene512/input-{i}.npz"
        bindings[str(ip)] = sha(ip)
        with np.load(ip) as z:
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
                enc: predict(s[None], clean[None], strength[None], 2026101319, 6)
                for enc, s in [("float", source), ("uint8", quantized)]
            }
            ref = GY / f"scene512/predictions-{i}-{mode}-0.npz"
            bindings[str(ref)] = sha(ref)
            with np.load(ref) as z:
                old = z["consistent_uint8"]
                references += 1
            values = {f"{m}_{enc}": vv[m][0] for enc, vv in v.items() for m in METHODS}
            save(f"scene-{i}-{mode}.npz", dict(broad_uint8=old, **values))
            if mode == "partial" and i not in (1, 3, 5, 7):
                continue
            draw(
                f"native-scene-{i}-{mode}.png",
                [
                    [
                        quantized,
                        old,
                        values["average_uint8"],
                        values["projected_uint8"],
                        values["projected_float"],
                    ]
                ],
                [
                    [
                        f"scene{i} {mode} source uint8",
                        "GY broad uint8",
                        "HA average uint8",
                        "HA projected uint8",
                        "HA projected float",
                    ]
                ],
            )
    assert count == 268 and references == 118 and len(records) == 30
    assert states == {name: digest(nnx.state(m, nnx.Param)) for name, m in models.items()}
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
    print(dict(new_outputs=count, references=references, pages=len(records)), flush=True)


def finalize(root):
    check(root)
    for name in ("test-results", "preflight", "prefix-audit"):
        assert read(root / f"{name}.json")["passed"]
    audit = read(root / "snapshot-audit.json")
    assert audit["exports"] == 20 and audit["raw_ema_exact"]
    for p, h in audit["bindings"].items():
        assert sha(Path(p)) == h
    for s in SAVED[1:]:
        assert read(root / f"training/restore-{s}.json")["full_state_exact"]
    rows = [json.loads(s) for s in (root / "training/sampling.jsonl").read_text().splitlines()]
    assert len(rows) == STEPS
    previous = [json.loads(s) for s in (GZ / "training/sampling.jsonl").read_text().splitlines()]
    for k, r in enumerate(rows):
        assert r["step"] == k + 1 and r["depth"] == k % 6 + 1
        for j in (0, 1):
            assert (
                r["inputs"][j]["input_payload_sha256"]
                == previous[2 * k + j]["input_payload_sha256"]
            )
        assert all(np.isfinite(v) for m in r["metrics"].values() for v in m.values())
    ev = read(root / "evaluation/metrics.json")
    assert ev["new_outputs"] == 268 and ev["states_unchanged"] and ev["protected_exact"]
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
            stage="complete_axis_gradient_projection",
            steps=STEPS,
            trained_arms=2,
            new_outputs=268,
            archived_reference_outputs=118,
            output_pages_reviewed=30,
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
