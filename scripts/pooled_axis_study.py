"""HH coherent retained axes, including partial legacy cases; archived HD control."""

import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from axis_family_data import axis_payload
from campaign_protocol import read, sha, write
from coherent_broad_objective import axis_gradient, selected_original
from cue_capacity import make_arm
from flax import nnx
from plausible_study import digest, exact
from pooled_axis_batches import batch as previous_batch
from pooled_axis_batches import load_data as previous_load_data
from pooled_axis_model import load_field, save_field
from pooled_procedural_study import GO as GO
from procedural_shape_study import AMBIGUITY as AMBIGUITY
from procedural_shape_study import DIAGNOSTIC as DIAGNOSTIC
from procedural_shape_study import EVAL_SEED as EVAL_SEED
from procedural_shape_study import GK as GK
from procedural_shape_study import MODELREF as MODELREF
from procedural_shape_study import select
from scale_style_balance_study import update

from lineartgen.config import load_config, save_config
from lineartgen.training import checkpoints, lineart_state

GP = Path("../jax-pooled-procedural-20261017/runs/pooled-procedural-20261017-retry1").resolve()
GT = Path("../jax-pooled-noopening-20261017/runs/pooled-noopening-20261017").resolve()
GU = Path("../jax-one-sided-opening-20261017/runs/one-sided-opening-20261017").resolve()
GW = Path("../jax-pooled-axis-20261017/runs/pooled-axis-20261017-retry3").resolve()
GX = Path("../jax-consistent-axis-fit-20261017/runs/consistent-axis-fit-20261017").resolve()
GY = Path("../jax-consistent-axis-broad-20261017/runs/consistent-axis-broad-20261017").resolve()
HC = Path("../jax-local-axis-support-fit-20261017/runs/local-axis-support-fit-20261017").resolve()
HD = Path("../jax-supported-axis-broad-20261017/runs/supported-axis-broad-20261017").resolve()
HF = Path("../jax-coherent-mask-fit-20261017/runs/coherent-mask-fit-20261017").resolve()
HG = Path("../jax-mask-context-fit-20261017/runs/mask-context-fit-20261017").resolve()
DOC = Path("docs/coherent-axis-broad-plan-20261017.md")
METHODS = ("consistent",)
REPORT_METHODS = ("winner", "consistent")
LABELS = dict(initial="FW mixed6000", winner="HD supported6000", consistent="HH retained6000")
STEPS = 6000
SAVED = (0, 6, 36, *range(1000, 6001, 1000))


def arms_from_root(root):
    arms = {}
    for name in METHODS:
        model, cfg = load_field(root / f"initials/{name}")
        assert (
            cfg == load_config(root / "config.json")
            and model.head.mode == "choice"
            and model.head.component == 0
        )
        arms[name] = make_arm(cfg, model)
        arms[name]["gradient"] = axis_gradient()
    return arms


def load_data():
    data = previous_load_data()
    data["gw_rows"] = [
        json.loads(s) for s in (GW / "training/sampling.jsonl").read_text().splitlines()
    ]
    return data


def batch(data, step):
    p, c, d = previous_batch(data, step)
    old = data["gw_rows"][step]
    assert digest((p, c[0], d)) == old["input_sha256"]
    assert digest(c) == old["candidate_payload_sha256"]
    assert np.asarray(c[3]).tolist() == old["context_keep"]
    assert data["last_audit"] == old["audit"]
    old_allowed = np.asarray(c[6]).copy()
    original_sha = digest(c)
    old_fields = tuple(np.asarray(v) for v in c[:6])
    records = None
    family = data["last_audit"]["family"]
    if family == "ambiguous":
        c = (*c[:6], jax.numpy.ones(4, dtype=bool))
    elif family == "legacy":
        axis, records = axis_payload(
            data, select(step), data["axis_widths"][step]["records"], c[:4], d, allow_partial=True
        )
        c = (*c[:4], *axis)
    allowed = np.asarray(c[6])
    promoted = ~old_allowed & allowed
    assert np.all(~old_allowed | allowed)
    for i in range(4):
        np.testing.assert_array_equal(c[i], old_fields[i])
    for i in (4, 5):
        np.testing.assert_array_equal(np.asarray(c[i])[~promoted], old_fields[i][~promoted])
    data["coherent_extension"] = dict(
        original_candidate_payload_sha256=original_sha,
        first_four_payload_sha256=digest(c[:4]),
        old_allowed=old_allowed.tolist(),
        allowed=allowed.tolist(),
        promoted=promoted.tolist(),
        axis_records=records,
        selected_original=np.asarray(
            selected_original(c[0], d, c[4], c[5], p.noise_levels[0], c[6])
        ).tolist(),
    )
    return p, c, d


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(GW / "completion.json")["stage"] == "complete_pooled_axis"
    assert read(HG / "decision.json")["mechanistic_fit_pass"]
    assert read(HD / "completion.json")["stage"] == "complete_supported_axis_broad"
    assert read(GY / "review-completion.json")["strict_protocol_passed"] is False
    root.mkdir(parents=True, exist_ok=False)
    (root / "diagnostics").mkdir()
    model, cfg = load_field(GW / "initials/choice")
    assert (
        cfg.train.steps == 6000
        and cfg.train.peak_learning_rate == 2e-4
        and cfg.train.batch_size == 4
    )
    assert sum(v.size for v in jax.tree.leaves(nnx.state(model, nnx.Param))) == 21890
    save_field(
        model,
        cfg,
        root / "initials/consistent",
        provenance="HH untouched GW initial choice; research only",
    )
    restored, cc = load_field(root / "initials/consistent")
    exact(nnx.state(model, nnx.Param), nnx.state(restored, nnx.Param))
    assert cc == cfg
    save_config(cfg, root / "config.json")
    shutil.copytree(GW / "reference", root / "reference")
    for name in (
        "train.npz",
        "data.json",
        "stress-plan.json",
        "data",
        "stress-data",
        "teacher-preview",
        "source-preview",
        "width-range-audit.json",
        "width-plan.json",
        "probe.json",
    ):
        (root / name).symlink_to((GW / name).resolve(), target_is_directory=(GW / name).is_dir())
    (root / "scene512").mkdir()
    for name in ["plan.json"] + [f"input-{i}.npz" for i in range(8)]:
        shutil.copy2(GW / "scene512" / name, root / "scene512" / name)
    (root / "diagnostics/orientation").mkdir()
    shutil.copy2(
        GW / "diagnostics/orientation/square48-input.npz",
        root / "diagnostics/orientation/square48-input.npz",
    )
    (root / "snapshots/step-6000").mkdir(parents=True)
    (root / "snapshots/step-6000/winner").symlink_to(
        (HD / "snapshots/step-6000/consistent").resolve(), target_is_directory=True
    )
    (root / "mask-probes").mkdir()
    for split in ("train", "transfer"):
        shutil.copy2(HC / f"{split}.npz", root / f"mask-probes/{split}.npz")
    (root / "coherent-probes").mkdir()
    for split in ("train", "transfer"):
        shutil.copy2(HF / f"{split}.npz", root / f"coherent-probes/{split}.npz")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    files = [p for p in root.rglob("*") if p.is_file()]
    files += [Path(f) for f in read(GW / "plan.json")["bindings"]]
    files += [
        GW / "completion.json",
        GW / "diagnostics/supplemental-binding.json",
        GW / "training/sampling.jsonl",
        GX / "completion.json",
        GX / "decision.json",
        GX / "diagnostics/supplemental-binding.json",
        GY / "review-completion.json",
        GY / "diagnostics/decision.json",
        GY / "diagnostics/supplemental-binding.json",
        HC / "completion.json",
        HC / "decision.json",
        HC / "supplemental-binding.json",
        HD / "completion.json",
        HD / "supplemental-binding.json",
        HD / "training/sampling.jsonl",
        HF / "completion.json",
        HF / "supplemental-binding.json",
        HG / "completion.json",
        HG / "decision.json",
        HG / "supplemental-binding.json",
        HG / "diagnostics/legacy-permission-audit.json",
        DOC.resolve(),
    ]
    files += list((GW / "initials/choice").iterdir()) + list(
        (HD / "snapshots/step-6000/consistent/raw").iterdir()
    )
    files += [
        Path(f).resolve()
        for f in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if f.endswith((".py", ".toml")) or f == "uv.lock"
    ]
    write(
        root / "plan.json",
        dict(
            commit=commit,
            bindings={str(p.resolve()): sha(p) for p in files},
            archive_sha256=sha(root / "source.tar.gz"),
            steps=STEPS,
            saved=SAVED,
            parameters=21890,
            trained_arms=1,
            control="Archived HD supported raw6000; key winner denotes baseline only",
            prior_mode="none",
            adoption=False,
            fresh_confirmation=False,
        ),
    )


def check(root):
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        == read(root / "plan.json")["commit"]
    )
    for f, h in read(root / "plan.json")["bindings"].items():
        assert sha(Path(f)) == h, f
    assert sha(root / "source.tar.gz") == read(root / "plan.json")["archive_sha256"]


def snapshot(root, arms, step):
    for name, arm in arms.items():
        for mode in ("raw", "ema"):
            model = nnx.clone(arm["model"])
            if mode == "ema":
                nnx.update(model, arm["ema"])
            save_field(
                model,
                arm["config"],
                root / f"snapshots/step-{step}/{name}/{mode}",
                provenance=f"HH retained axis {name} {step} {mode}; research only",
            )


def train(root, until):
    check(root)
    assert jax.devices()[0].platform != "cpu" and until in SAVED[3:]
    assert read(root / "test-results.json")["passed"] and read(root / "preflight.json")["passed"]
    preview = read(root / "support-preview/observations.json")
    for r in read(root / "support-preview/manifest.json")["records"]:
        assert preview[r["file"]] and sha(root / "support-preview" / r["file"]) == r["sha256"]
    data, arms = load_data(), arms_from_root(root)
    out = root / "training"
    out.mkdir(exist_ok=True)
    rows = []
    with checkpoints.manager(out / "checkpoints", keep=len(SAVED)) as manager:
        start = lineart_state.restore(manager, arms) if manager.latest_step() is not None else 0
        journal = out / "sampling.jsonl"
        if journal.exists():
            rows = [json.loads(s) for s in journal.read_text().splitlines()]
            assert len(rows) == start, (
                "Preserve interruption evidence and explicitly audit replay before retry"
            )
            assert [r["step"] for r in rows] == list(range(1, start + 1))
        if start == 0:
            lineart_state.save(manager, arms, 0)
            snapshot(root, arms, 0)
        assert start < until
        for k in range(start, until):
            p, c, d = batch(data, k)
            metrics, seconds = {}, {}
            for name in METHODS:
                tick = time.perf_counter()
                metrics[name] = update(arms[name], p, c, d)
                seconds[name] = time.perf_counter() - tick
                assert all(np.isfinite(v) for v in metrics[name].values())
            row = dict(
                step=k + 1,
                **select(k),
                metrics=metrics,
                seconds=seconds,
                input_sha256=digest((p, c[0], d)),
                candidate_payload_sha256=digest(c),
                context_keep=np.asarray(c[3]).tolist(),
                audit=data["last_audit"],
                coherent_extension=data["coherent_extension"],
                consistent_original=data["coherent_extension"]["selected_original"],
            )
            rows.append(row)
            with journal.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if k == 35:
                replay = arms_from_root(root)
                for j in range(36):
                    for name in METHODS:
                        exact(update(replay[name], *batch(data, j)), rows[j]["metrics"][name])
                exact(lineart_state.payload(replay, 36), lineart_state.payload(arms, 36))
                write(
                    root / "prefix-audit.json",
                    dict(passed=True, metrics_exact=True, full_state_exact=True, steps=36),
                )
            if k + 1 in SAVED:
                lineart_state.save(manager, arms, k + 1)
                restored = arms_from_root(root)
                assert lineart_state.restore(manager, restored, k + 1) == k + 1
                exact(lineart_state.payload(restored, k + 1), lineart_state.payload(arms, k + 1))
                snapshot(root, arms, k + 1)
                write(out / f"restore-{k + 1}.json", dict(full_state_exact=True))
            if k < 36 or (k + 1) % 100 == 0:
                write(out / "progress.json", dict(step=k + 1, metrics=metrics))
                print(dict(step=k + 1), flush=True)
        if until < STEPS:
            return
        bindings = {}
        for step in SAVED:
            restored = arms_from_root(root)
            assert lineart_state.restore(manager, restored, step) == step
            for name, arm in restored.items():
                for mode in ("raw", "ema"):
                    expected = nnx.state(arm["model"], nnx.Param) if mode == "raw" else arm["ema"]
                    folder = root / f"snapshots/step-{step}/{name}/{mode}"
                    model, cfg = load_field(folder)
                    assert cfg == arm["config"]
                    exact(expected, nnx.state(model, nnx.Param))
                    bindings.update({str(p): sha(p) for p in folder.iterdir() if p.is_file()})
        write(root / "snapshot-audit.json", dict(raw_ema_exact=True, exports=18, bindings=bindings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--until", type=int, default=STEPS)
    a = parser.parse_args()
    root = a.root.resolve()
    if a.action == "prepare":
        prepare(root)
    else:
        with (root / "process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(root, a.until)
