"""Fixed FH weights: native field/depth/coarse-context diagnosis."""

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
from subpixel_field_model import DISTANCE_RANGE, coverage, unpack

from lineartgen.diffusion import source_guides

PARENT = Path("../jax-fresh-aligned-20261015/runs/fresh-aligned-20261015").resolve()
MODEL = PARENT / "snapshots/step-24000/aligned/raw"
DOC = Path("docs/aligned-recursion-plan-20261016.md")
ENCODINGS = ("float", "uint8")
MODES = ("normal", "zero")
DEPTHS = tuple(range(1, 7))


@nnx.jit(static_argnames=("zero",))
def diagnostic_infer(model, prepared, *, zero):
    guides = source_guides(model, prepared.guides)
    g = guides[-1]
    insert = jnp.zeros((g.shape[0], g.shape[1] // 2, g.shape[2] // 2, 1), g.dtype)
    for level in reversed(range(len(guides))):
        head = model(
            prepared.noisy[level],
            prepared.noise_levels[level],
            insert,
            guides[level],
            prepared.progress[level],
            pyramid_level=level,
            return_network_output=True,
        )
        insert = jnp.clip(head.mean(-1, keepdims=True), -1, 1)
        if zero:
            insert = jnp.zeros_like(insert)
    field = unpack(head) * DISTANCE_RANGE
    image = jnp.where(prepared.noise_levels[0] > 0, coverage(field), prepared.guides[0])
    return image, field


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(PARENT / "completion.json")["stage"] == "complete_fresh_aligned"
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for i, box in [(1, (116, 146, 236, 290)), (2, (65, 160, 190, 235)), (7, (292, 140, 390, 238))]:
        records.append(
            dict(
                name=f"scene{i}",
                index=i,
                size=512,
                crop=box,
                depth=6,
                input=str(PARENT / f"scene512/input-{i}.npz"),
                comparison=str(PARENT / f"scene512/predictions-{i}-whole-0.npz"),
            )
        )
    for name, index, split, file, box in [
        ("probe37", 37, "new_probe", "new_probe/037.npz", (40, 88, 112, 161)),
        ("oldtrain36", 36, "old_train", "expanded/288.npz", (45, 105, 165, 170)),
    ]:
        records.append(
            dict(
                name=name,
                index=index,
                split=split,
                size=256,
                crop=box,
                depth=3,
                input=str(PARENT / "data" / file),
                comparison=str(
                    PARENT / f"evaluation/predictions/{split}-{index:03d}-whole05-0.npz"
                ),
            )
        )
    bindings = {}
    for r in records:
        for k in ("input", "comparison"):
            bindings[r[k]] = sha(Path(r[k]))
    for f in MODEL.iterdir():
        if f.is_file():
            bindings[str(f)] = sha(f)
    for f in (DOC.resolve(), PARENT / "completion.json", PARENT / "diagnostics/decision.json"):
        bindings[str(f)] = sha(f)
    for name in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        if name.endswith((".py", ".toml")) or name == "uv.lock":
            f = Path(name).resolve()
            bindings[str(f)] = sha(f)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            model=str(MODEL),
            bindings=bindings,
            source_archive_sha256=sha(root / "source.tar.gz"),
            outputs=120,
            native_pages=15,
            training=False,
            fresh_confirmation=False,
            adoption=False,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for p, h in plan["bindings"].items():
        assert sha(Path(p)) == h, p
    assert sha(root / "source.tar.gz") == plan["source_archive_sha256"]
    return plan


def inputs(record):
    with np.load(record["input"]) as z:
        source = z["source"]
        clean = z["clean"] if record["size"] == 512 else z["variants"][13]
        quantized = (
            np.rint(np.clip((source + 1) * 127.5, 0, 255)).astype(np.uint8).astype(np.float32)
            / 127.5
            - 1
        )
        if record["size"] == 512:
            np.testing.assert_array_equal(quantized, z["quantized"])
    if record["size"] == 512:
        key = jax.random.fold_in(
            jax.random.fold_in(jax.random.PRNGKey(2026101319), record["index"]), 0
        )
    else:
        key = jax.random.PRNGKey(2026101503)
        for v in (3 if record["split"] == "new_probe" else 0, record["index"], 1, 0):
            key = jax.random.fold_in(key, v)
    return source, quantized, clean, key


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    model, _ = load_field(plan["model"])
    before = digest(nnx.state(model, nnx.Param))
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    bindings, rows = {}, []
    compared = 0
    started = time.perf_counter()
    for r in plan["records"]:
        source, quantized, clean, key = inputs(r)
        for encoding, src in (("float", source), ("uint8", quantized)):
            for depth in DEPTHS:
                prepared = prepare_inputs(
                    clean[None], src[None], np.full_like(src[None], 0.5), key, depth
                )
                baseline = np.asarray(infer(model, prepared))[0]
                if depth == r["depth"] and (r["size"] == 512 or encoding == "float"):
                    with np.load(r["comparison"]) as z:
                        np.testing.assert_array_equal(
                            baseline, z["aligned_" + encoding if r["size"] == 512 else "aligned"]
                        )
                    compared += 1
                arrays = {}
                for mode in MODES:
                    value, field = diagnostic_infer(model, prepared, zero=mode == "zero")
                    value, field = np.asarray(value)[0], np.asarray(field)[0]
                    assert np.isfinite(value).all() and np.isfinite(field).all()
                    assert value.min() >= -1 and value.max() <= 1
                    arrays[mode] = value
                    arrays[mode + "_field"] = field
                np.testing.assert_allclose(arrays["normal"], baseline, atol=5e-6, rtol=1e-4)
                if depth == 1:
                    np.testing.assert_array_equal(arrays["normal"], arrays["zero"])
                    np.testing.assert_array_equal(arrays["normal_field"], arrays["zero_field"])
                f = out / f"{r['name']}-{encoding}-depth{depth}.npz"
                np.savez_compressed(f, **arrays)
                bindings[str(f)] = sha(f)
                rows.append(
                    dict(
                        case=r["name"],
                        encoding=encoding,
                        depth=depth,
                        baseline_max_abs=float(np.max(np.abs(arrays["normal"] - baseline))),
                    )
                )
        print(dict(case=r["name"], outputs=24), flush=True)
    assert compared == 8
    assert before == digest(nnx.state(model, nnx.Param))
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            outputs=120,
            existing_predictions_exact=8,
            normal_parity_passed=True,
            depth1_modes_exact=True,
            states_unchanged=True,
            seconds=time.perf_counter() - started,
            bindings=bindings,
        ),
    )
    render(root)


def pixels(value):
    return Image.fromarray(
        np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
    ).convert("RGB")


def render(root):
    plan = check(root)
    out = root / "evaluation"
    records = []
    for r in plan["records"]:
        box = r["crop"]
        w, h = box[2] - box[0], box[3] - box[1]
        for encoding in ENCODINGS:
            canvas = Image.new("RGB", (6 * max(w, 125), 2 * (h + 30)), "white")
            pen = ImageDraw.Draw(canvas)
            bound = {}
            for col, depth in enumerate(DEPTHS):
                f = out / f"{r['name']}-{encoding}-depth{depth}.npz"
                bound[str(f)] = sha(f)
                with np.load(f) as z:
                    for row, mode in enumerate(MODES):
                        x, y = col * max(w, 125), row * (h + 30)
                        pen.text((x + 2, y + 3), f"{mode} depth{depth}", fill="black")
                        canvas.paste(pixels(z[mode]).crop(box), (x, y + 25))
            f = out / f"detail-{r['name']}-{encoding}.png"
            canvas.save(f)
            records.append(
                dict(file=f.name, sha256=sha(f), bindings=bound, crop=box, native_scale=True)
            )
        size = r["size"]
        canvas = Image.new("RGB", (3 * size, size + 30), "white")
        pen = ImageDraw.Draw(canvas)
        source = inputs(r)[1]
        f = out / f"{r['name']}-uint8-depth6.npz"
        with np.load(f) as z:
            for col, (label, value) in enumerate(
                (
                    ("uint8 source", source),
                    ("normal depth6", z["normal"]),
                    ("zero depth6", z["zero"]),
                )
            ):
                pen.text((col * size + 4, 5), label, fill="black")
                canvas.paste(pixels(value), (col * size, 30))
        f_out = out / f"full-{r['name']}.png"
        canvas.save(f_out)
        records.append(
            dict(
                file=f_out.name,
                sha256=sha(f_out),
                bindings={str(f): sha(f), r["input"]: sha(Path(r["input"]))},
                native_scale=True,
            )
        )
    write(
        out / "manifest.json", dict(records=records, thresholding=False, contrast_rescaling=False)
    )


def finalize(root):
    check(root)
    out = root / "evaluation"
    m = read(out / "metrics.json")
    assert m["outputs"] == 120 and m["existing_predictions_exact"] == 8
    assert m["normal_parity_passed"] and m["depth1_modes_exact"] and m["states_unchanged"]
    for p, h in m["bindings"].items():
        assert sha(Path(p)) == h
    review = read(out / "visual-review.json")
    records = read(out / "manifest.json")["records"]
    assert review["reviewed"] and len(records) == len(review["observations"]) == 15
    for r in records:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
        for p, h in r["bindings"].items():
            assert sha(Path(p)) == h
    assert read(root / "driver-evaluate.json")["returncode"] == 0
    files = [
        "plan.json",
        "driver-evaluate.json",
        "evaluation/metrics.json",
        "evaluation/manifest.json",
        "evaluation/visual-review.json",
        "diagnostics/decision.json",
    ]
    write(
        root / "completion.json",
        dict(
            stage="complete_aligned_recursion",
            outputs=120,
            native_pages_reviewed=15,
            training=False,
            adoption=False,
            fresh_confirmation=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "evaluate", "finalize"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
