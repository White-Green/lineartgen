"""Construct a broad-stroke/fill pair indistinguishable within the finest CNN reach."""

import argparse
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from physical_field_model import infer, load_field
from PIL import Image, ImageDraw
from plausible_study import digest
from scipy.ndimage import gaussian_filter

from lineartgen.diffusion import source_guides

PARENT = Path("../jax-wide-emphasis-20261016/runs/wide-emphasis-20261016").resolve()
DOC = Path("docs/local-ambiguity-plan-20261016.md")
MODELS = {
    "FN": PARENT / "reference",
    "FO": PARENT / "initials/uniform",
    "uniform": PARENT / "snapshots/step-6000/uniform/raw",
    "emphasized": PARENT / "snapshots/step-6000/emphasized/raw",
}
POINT = (128, 33)
RADIUS = 43


def inputs():
    alpha = np.zeros((2, 256, 256, 1), np.float32)
    alpha[0, 48:208, 32:94] = 1
    alpha[1, 32:224, 32:224] = 1
    source = np.stack(
        [1 - 0.8 * gaussian_filter(a, sigma=(1.2, 1.2, 0), mode="constant", cval=0) for a in alpha]
    )
    clean = np.ones_like(alpha)
    clean[0, 48:208, 62:64] = -1
    clean[1, 32:224, 32:224] = -1
    assert source.dtype == clean.dtype == np.float32
    assert clean[0, *POINT, 0] == 1 and clean[1, *POINT, 0] == -1
    return source, clean


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    source, clean = inputs()
    np.savez_compressed(root / "inputs.npz", source=source, clean=clean)
    files = [DOC.resolve(), root / "inputs.npz", PARENT / "completion.json"]
    files += [f for folder in MODELS.values() for f in folder.iterdir() if f.is_file()]
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
            point_yx=POINT,
            normalized_guide_radius=RADIUS,
            models={m: str(p) for m, p in MODELS.items()},
            bindings={str(p): sha(p) for p in files},
            archive_sha256=sha(root / "source.tar.gz"),
            outputs=8,
            native_pages=2,
            training=False,
            fresh_confirmation=False,
            adoption=False,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for file, expected in plan["bindings"].items():
        assert sha(Path(file)) == expected, file
    assert sha(root / "source.tar.gz") == plan["archive_sha256"]
    return plan


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform == "cpu"
    with np.load(root / "inputs.npz") as z:
        source, clean = z["source"], z["clean"]
    parts = [
        prepare_inputs(
            clean[i : i + 1],
            source[i : i + 1],
            np.full_like(source[i : i + 1], 0.5),
            jax.random.PRNGKey(2026101615),
            1,
        )
        for i in range(2)
    ]
    prepared = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), *parts)
    y, x = POINT
    box = (slice(y - RADIUS, y + RADIUS + 1), slice(max(0, x - RADIUS), x + RADIUS + 1))
    # The remaining columns lie outside the canvas and have identical convolution padding.
    for tensor in (
        prepared.guides[0],
        prepared.noisy[0],
        prepared.noise_levels[0],
        prepared.progress[0],
    ):
        np.testing.assert_array_equal(np.asarray(tensor)[0][box], np.asarray(tensor)[1][box])
    values = dict(source=source, clean=clean)
    rows = []
    started = time.monotonic()
    for method, path in plan["models"].items():
        model, cfg = load_field(Path(path))
        assert tuple(cfg.model.local_dilations) == (1, 4, 16)
        assert 1 + 2 * sum(cfg.model.local_dilations) == RADIUS
        assert len(model.blocks) == 3 and model.stem.dilation == 1
        assert not getattr(model, "uses_raw_guide", False)
        before = digest(nnx.state(model, nnx.Param))
        normalized = np.asarray(source_guides(model, prepared.guides)[0])
        np.testing.assert_array_equal(normalized[0][box], normalized[1][box])
        value = np.asarray(infer(model, prepared))
        assert digest(nnx.state(model, nnx.Param)) == before
        assert np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1
        difference = float(abs(value[0, y, x, 0] - value[1, y, x, 0]))
        np.testing.assert_allclose(value[0, y, x], value[1, y, x], atol=1e-6, rtol=0)
        values[method] = value
        rows.append(
            dict(
                model=method,
                normalized_patch_exact=True,
                states_unchanged=True,
                prediction_at_point=value[:, y, x, 0].tolist(),
                teacher_at_point=clean[:, y, x, 0].tolist(),
                prediction_difference=difference,
            )
        )
    np.savez_compressed(root / "predictions.npz", **values)
    records = []
    for name, crop in (
        ("native-full.png", (0, 0, 256, 256)),
        ("native-detail.png", (24, 112, 80, 144)),
    ):
        w, h = crop[2] - crop[0], crop[3] - crop[1]
        stride = max(w, 110)
        canvas = Image.new("RGB", (stride * len(values), 2 * (h + 38)), "white")
        pen = ImageDraw.Draw(canvas)
        for row in range(2):
            pen.text(
                (3, row * (h + 38) + 2), "broad62px" if row == 0 else "filled192px", fill="black"
            )
            for col, (label, value) in enumerate(values.items()):
                gray = np.rint(np.clip((value[row, ..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                canvas.paste(
                    Image.fromarray(gray).crop(crop).convert("RGB"),
                    (col * stride, row * (h + 38) + 38),
                )
                pen.text((col * stride + 3, row * (h + 38) + 20), label, fill="black")
        dest = root / name
        canvas.save(dest)
        records.append(dict(file=name, sha256=sha(dest), crop=crop, native_scale=True))
    write(root / "manifest.json", dict(records=records))
    write(
        root / "metrics.json",
        dict(
            rows=rows,
            outputs=8,
            seconds=time.monotonic() - started,
            bindings={"predictions.npz": sha(root / "predictions.npz")},
            interpretation=(
                "Constructed local ambiguity under the current thin-stroke/fill teacher policy; "
                "not a prevalence estimate or proof of the cause of all observed failures."
            ),
        ),
    )
    print(rows, flush=True)


def finalize(root):
    check(root)
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    rows = read(root / "metrics.json")["rows"]
    assert len(rows) == 4 and all(r["states_unchanged"] for r in rows)
    for file, expected in read(root / "metrics.json")["bindings"].items():
        assert sha(root / file) == expected
    notes = read(root / "observations.json")
    records = read(root / "manifest.json")["records"]
    assert len(notes) == len(records) == 2
    for r in records:
        assert sha(root / r["file"]) == r["sha256"] and r["file"] in notes
    write(
        root / "visual-review.json",
        dict(reviewed=True, native_scale=True, files=records, observations=notes),
    )
    names = (
        "plan.json",
        "driver-evaluate.json",
        "metrics.json",
        "manifest.json",
        "visual-review.json",
        "decision.json",
    )
    write(
        root / "completion.json",
        dict(
            stage="complete_local_ambiguity",
            outputs=8,
            native_pages_reviewed=2,
            adoption=False,
            fresh_confirmation=False,
            evidence={f: sha(root / f) for f in names},
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
