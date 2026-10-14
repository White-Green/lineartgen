"""Fixed-model transfer assessment on established development probes and scenes."""

import argparse
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from campaign_protocol import read, sha, write
from contour_coverage_study import mask
from cue_capacity import prepare_inputs
from flax import nnx
from PIL import Image, ImageDraw
from plausible_study import digest
from subpixel_field_model import infer, load_field

DQ = Path("../jax-opaque-geometry-20261014/runs/opaque-geometry-20261014").resolve()
DU = Path("../jax-field-source-support-20261014/runs/field-source-support-20261014").resolve()
DY = Path("../jax-region-sign-geometry-20261014/runs/region-sign-geometry-20261014").resolve()
PATHS = dict(
    baseline=DU / "snapshots/step-6000/control/raw",
    control=DY / "snapshots/step-3000/control/raw",
    regions=DY / "snapshots/step-3000/regions/raw",
)
DOC = Path("docs/geometry-transfer-plan-20261014.md")
SEED = 2026101423


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(DQ / "completion.json")["stage"] == "complete_opaque_geometry_oracle"
    assert read(DY / "completion.json")["stage"] == "complete_region_sign_geometry"
    root.mkdir(parents=True, exist_ok=False)
    records = read(DQ / "plan.json")["records"]
    assert len(records) == 24
    bound = {}
    for r in records:
        for p in (DQ / r["file"], DQ / "outputs" / f"{r['name']}.npz"):
            bound[str(p)] = sha(p)
    for path in PATHS.values():
        bound.update({str(p): sha(p) for p in path.iterdir() if p.is_file()})
    for p in (
        DQ / "completion.json",
        DU / "completion.json",
        DY / "completion.json",
        DY / "evaluation/visual-review.json",
        DOC,
    ):
        bound[str(p.resolve())] = sha(p)
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    for name in tracked:
        if name.endswith((".py", ".toml")) or name == "uv.lock":
            p = Path(name).resolve()
            bound[str(p)] = sha(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            models={k: str(v) for k, v in PATHS.items()},
            bindings=bound,
            source_archive_sha256=sha(root / "source.tar.gz"),
            depths=[3, 6],
            modes=["whole", "partial"],
            encodings=["float", "uint8"],
            seed=SEED,
            outputs=576,
            native_pages=32,
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


def evaluate(root):
    plan = check(root)
    models = {name: load_field(path)[0] for name, path in PATHS.items()}
    states = {name: digest(nnx.state(model, nnx.Param)) for name, model in models.items()}
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    rows = []
    bindings = {}
    start = time.perf_counter()
    for index, r in enumerate(plan["records"]):
        with np.load(DQ / r["file"]) as z:
            source = z["source"]
            clean = z["clean"]
        quantized = (
            np.rint(np.clip((source + 1) * 127.5, 0, 255)).astype(np.uint8).astype(np.float32)
            / 127.5
            - 1
        )
        for depth in plan["depths"]:
            for mode in plan["modes"]:
                t = mask(r["size"], mode)
                arrays = {}
                for encoding, src in (("float", source), ("uint8", quantized)):
                    p = prepare_inputs(
                        clean[None],
                        src[None],
                        t[None],
                        jax.random.fold_in(jax.random.PRNGKey(SEED), index),
                        depth,
                    )
                    for name, model in models.items():
                        pred = np.asarray(infer(model, p))[0]
                        assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
                        np.testing.assert_array_equal(pred[t == 0], src[t == 0])
                        arrays[f"{name}_{encoding}"] = pred
                path = out / f"{r['name']}-d{depth}-{mode}.npz"
                np.savez_compressed(path, **arrays)
                bindings[str(path)] = sha(path)
                rows.append(
                    dict(
                        name=r["name"],
                        depth=depth,
                        mode=mode,
                        outputs=6,
                        protected_exact=True,
                        quantization_output_rms={
                            name: float(
                                np.sqrt(
                                    np.mean(
                                        (arrays[f"{name}_float"] - arrays[f"{name}_uint8"]) ** 2
                                    )
                                )
                            )
                            for name in models
                        },
                        source_already_quantized=bool(np.array_equal(source, quantized)),
                    )
                )
        print(dict(case=index + 1, total=24, name=r["name"]), flush=True)
    assert sum(r["outputs"] for r in rows) == 576
    assert states == {name: digest(nnx.state(model, nnx.Param)) for name, model in models.items()}
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            outputs=576,
            bindings=bindings,
            states_unchanged=True,
            evaluation_seconds=time.perf_counter() - start,
        ),
    )
    render(root)


def render(root):
    plan = check(root)
    out = root / "evaluation"
    records = []
    for depth in plan["depths"]:
        for mode in plan["modes"]:
            for group, size, perpage in (("probe", 256, 4), ("scene", 512, 2)):
                items = [r for r in plan["records"] if r["name"].startswith(group)]
                for first in range(0, len(items), perpage):
                    selected = items[first : first + perpage]
                    canvas = Image.new("RGB", (5 * size, 34 + len(selected) * (size + 24)), "white")
                    draw = ImageDraw.Draw(canvas)
                    for col, label in enumerate(
                        (
                            "uint8 rough source",
                            "one permitted opaque geometry",
                            "DU before focused fits",
                            "DY shared RMS control",
                            "DY near/far RMS",
                        )
                    ):
                        draw.text((col * size + 6, 8), label, fill="black")
                    predictions = {}
                    for row, r in enumerate(selected):
                        with np.load(DQ / r["file"]) as z:
                            source = z["source"]
                        source = (
                            np.rint(np.clip((source + 1) * 127.5, 0, 255))
                            .astype(np.uint8)
                            .astype(np.float32)
                            / 127.5
                            - 1
                        )
                        with np.load(DQ / "outputs" / f"{r['name']}.npz") as z:
                            clean = z["fine"]
                        t = mask(size, mode)
                        clean = np.where(t > 0, clean, source)
                        path = out / f"{r['name']}-d{depth}-{mode}.npz"
                        with np.load(path) as z:
                            arrays = [source, clean] + [z[f"{name}_uint8"] for name in PATHS]
                        predictions[str(path)] = sha(path)
                        yy = 34 + row * (size + 24)
                        draw.text(
                            (6, yy),
                            f"{r['name']} {r['kind']} / {mode} / depth{depth}",
                            fill="black",
                        )
                        for col, a in enumerate(arrays):
                            pixels = np.uint8(np.round(np.clip((a[..., 0] + 1) * 127.5, 0, 255)))
                            canvas.paste(
                                Image.fromarray(pixels).convert("RGB"), (col * size, yy + 20)
                            )
                    path = out / f"native-{group}-d{depth}-{mode}-{first}.png"
                    canvas.save(path)
                    records.append(dict(file=path.name, sha256=sha(path), predictions=predictions))
    assert len(records) == 32
    write(out / "manifest.json", dict(records=records))


def finalize(root):
    check(root)
    out = root / "evaluation"
    metrics = read(out / "metrics.json")
    assert metrics["outputs"] == 576 and metrics["states_unchanged"]
    assert all(r["protected_exact"] for r in metrics["rows"])
    for p, h in metrics["bindings"].items():
        assert sha(Path(p)) == h
    manifest = read(out / "manifest.json")["records"]
    review = read(out / "visual-review.json")
    assert review["reviewed"] and len(review["observations"]) == len(manifest) == 32
    for r in manifest:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
        for p, h in r["predictions"].items():
            assert sha(Path(p)) == h
    write(
        root / "completion.json",
        dict(
            stage="complete_geometry_transfer",
            outputs=576,
            native_pages_reviewed=32,
            training=False,
            adoption=False,
            fresh_confirmation=False,
            evidence={
                str(p.relative_to(root)): sha(p)
                for p in (
                    root / "plan.json",
                    out / "metrics.json",
                    out / "manifest.json",
                    out / "visual-review.json",
                )
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.root.resolve())
