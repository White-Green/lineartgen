"""Inspect predetermined broad examples from exact completed training batches."""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from context_dropout_objective import dropout_forward
from flax import nnx
from physical_field_model import infer, load_field
from PIL import Image, ImageDraw
from plausible_study import digest
from subpixel_field_data import central_field
from subpixel_field_model import coverage
from wide_emphasis_study import batch, load_data

PARENT = Path("../jax-wide-emphasis-20261016/runs/wide-emphasis-20261016").resolve()
DOC = Path("docs/wide-training-audit-plan-20261016.md")
MODELS = {
    "FN": PARENT / "reference",
    "FO": PARENT / "initials/uniform",
    "uniform": PARENT / "snapshots/step-6000/uniform/raw",
    "emphasized": PARENT / "snapshots/step-6000/emphasized/raw",
}


def select():
    selections = read(PARENT / "sampling-plan.json")
    widths = read(PARENT / "width-plan.json")
    records = read(PARENT / "data.json")["records"]
    cells = {}
    for selection, row in zip(selections, widths, strict=True):
        assert selection["step"] == row["step"]
        for width_record in row["records"]:
            item = width_record["item"]
            primary = width_record["primary"]
            width = primary["new"]
            if primary["index"] < 464 or width == primary["old"] or selection["modes"][item] != 1:
                continue
            paths = records[primary["index"]]["metadata"]["path_geometry"]
            assert len(paths) == 1
            points = paths[0]["points"]
            length = sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:], strict=True))
            aspect = width / length
            width_bin = 0 if width < 40 else 1 if width < 52 else 2
            aspect_bin = int(aspect >= 0.5)
            composed = int(selection["overlays"][item]["enabled"])
            key = (width_bin, aspect_bin, composed)
            cells.setdefault(key, []).append(
                dict(
                    step=selection["step"],
                    item=item,
                    index=primary["index"],
                    width=width,
                    length=length,
                    aspect=aspect,
                    depth=selection["depth"],
                    composed=bool(composed),
                )
            )
    assert set(cells) == {(w, a, c) for w in range(3) for a in range(2) for c in range(2)}
    return [
        dict(cell=list(key), population=len(rows), **rows[0]) for key, rows in sorted(cells.items())
    ]


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_wide_emphasis"
    root.mkdir(parents=True, exist_ok=False)
    files = [
        DOC.resolve(),
        *[
            PARENT / name
            for name in (
                "completion.json",
                "supplemental-binding.json",
                "diagnostics/decision.json",
                "data.json",
                "sampling-plan.json",
                "width-plan.json",
                "training/sampling.jsonl",
            )
        ],
        *[f for folder in MODELS.values() for f in folder.iterdir() if f.is_file()],
    ]
    for name in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        if name.endswith((".py", ".toml")) or name == "uv.lock":
            files.append(Path(name).resolve())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit],
        check=True,
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=select(),
            models={m: str(p) for m, p in MODELS.items()},
            bindings={str(f): sha(f) for f in files},
            source_archive_sha256=sha(root / "source.tar.gz"),
            training=False,
            fresh_confirmation=False,
            adoption=False,
            selected_outputs=60,
            full_batch_outputs=240,
            native_pages=6,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for file, expected in plan["bindings"].items():
        assert sha(Path(file)) == expected, file
    assert sha(root / "source.tar.gz") == plan["source_archive_sha256"]
    assert plan["records"] == select()
    return plan


@nnx.jit
def training_prediction(model, prepared, keep):
    field = dropout_forward(model, prepared, keep)[0]
    return jnp.where(prepared.noise_levels[0] > 0, coverage(field), prepared.guides[0])


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    models = {m: load_field(Path(p))[0] for m, p in plan["models"].items()}
    states = {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    data = load_data(PARENT)
    journal = [
        json.loads(line) for line in (PARENT / "training/sampling.jsonl").read_text().splitlines()
    ]
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    rows, bindings = [], {}
    started = time.monotonic()
    for case, record in enumerate(plan["records"]):
        step, item = record["step"], record["item"]
        prepared, bases, directions = batch(data, step, "emphasized")
        expected = journal[step]
        assert digest((prepared, bases[0], directions)) == expected["input_sha256"]["emphasized"]
        assert digest(bases[:3]) == expected["candidate_payload_sha256"]["emphasized"]
        assert np.asarray(bases[3]).tolist() == expected["context_keep"]
        assert np.asarray(bases[4]).tolist() == expected["teacher_audit"]["example_weights"]
        assert data["last_audit"]["changed_examples"][item]
        assert np.all(np.asarray(prepared.noise_levels[0][item]) == 0.5)
        teacher = jnp.where(bases[2][:, None, None], bases[1], bases[0])
        oracle = coverage(central_field(teacher.astype(jnp.float32))[..., None])
        oracle = jnp.where(prepared.noise_levels[0] > 0, oracle, prepared.guides[0])
        values = {m: np.asarray(infer(model, prepared)) for m, model in models.items()}
        values["emphasized_training"] = np.asarray(
            training_prediction(models["emphasized"], prepared, bases[3])
        )
        no_drop = np.asarray(
            training_prediction(models["emphasized"], prepared, jnp.ones_like(bases[3]))
        )
        np.testing.assert_allclose(no_drop, values["emphasized"], atol=5e-6, rtol=1e-4)
        maximum = float(abs(no_drop - values["emphasized"]).max())
        if record["depth"] == 1:
            np.testing.assert_array_equal(values["emphasized_training"], values["emphasized"])
        for value in (*values.values(), np.asarray(oracle)):
            assert value.shape == (4, 256, 256, 1)
            assert np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1
        file = out / f"case-{case:02d}.npz"
        np.savez_compressed(
            file,
            source=np.asarray(prepared.guides[0]),
            oracle=np.asarray(oracle),
            **values,
        )
        bindings[file.name] = sha(file)
        rows.append(
            dict(
                case=case,
                **record,
                inputs_and_teachers_exact=True,
                keep=np.asarray(bases[3])[:, item, 0, 0, 0].tolist(),
                no_drop_parity_max=maximum,
            )
        )
        print(
            dict(case=case, step=step, width=record["width"], seconds=time.monotonic() - started),
            flush=True,
        )
    assert states == {m: digest(nnx.state(model, nnx.Param)) for m, model in models.items()}
    pages = []
    labels = ("source", "oracle", "FN", "FO", "uniform", "emphasized", "emphasized_training")
    for first in range(0, 12, 2):
        canvas = Image.new("RGB", (256 * len(labels), 2 * 292), "white")
        pen = ImageDraw.Draw(canvas)
        for row, case in enumerate((first, first + 1)):
            r = rows[case]
            title = (
                f"case{case} step{r['step']} item{r['item']} width{r['width']:.1f} "
                f"ratio{r['aspect']:.3f} depth{r['depth']} composed{r['composed']}"
            )
            pen.text((4, row * 292 + 2), title, fill="black")
            with np.load(out / f"case-{case:02d}.npz") as arrays:
                for col, label in enumerate(labels):
                    value = arrays[label][r["item"], ..., 0]
                    pixels = np.rint(np.clip((value + 1) * 127.5, 0, 255)).astype(np.uint8)
                    canvas.paste(
                        Image.fromarray(pixels).convert("RGB"), (col * 256, row * 292 + 36)
                    )
                    pen.text((col * 256 + 4, row * 292 + 18), label, fill="black")
        dest = out / f"native-{first}-{first + 1}.png"
        canvas.save(dest)
        pages.append(dict(file=dest.name, sha256=sha(dest)))
    write(out / "manifest.json", dict(records=pages))
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            selected_outputs=60,
            full_batch_outputs=240,
            states_unchanged=True,
            bindings=bindings,
            seconds=time.monotonic() - started,
        ),
    )


def finalize(root):
    check(root)
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    metrics = read(root / "evaluation/metrics.json")
    assert len(metrics["rows"]) == 12 and metrics["states_unchanged"]
    for file, expected in metrics["bindings"].items():
        assert sha(root / "evaluation" / file) == expected
    observations = read(root / "observations.json")
    records = read(root / "evaluation/manifest.json")["records"]
    assert len(records) == len(observations) == 6
    for record in records:
        assert sha(root / "evaluation" / record["file"]) == record["sha256"]
        assert record["file"] in observations
    write(
        root / "visual-review.json",
        dict(reviewed=True, native_scale=True, files=records, observations=observations),
    )
    files = (
        "plan.json",
        "driver-evaluate.json",
        "evaluation/metrics.json",
        "evaluation/manifest.json",
        "observations.json",
        "visual-review.json",
        "decision.json",
    )
    write(
        root / "completion.json",
        dict(
            stage="complete_wide_training_audit",
            adoption=False,
            fresh_confirmation=False,
            selected_outputs=60,
            full_batch_outputs=240,
            native_pages_reviewed=6,
            evidence={f: sha(root / f) for f in files},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "finalize": finalize}[args.phase](
        args.root.resolve()
    )
