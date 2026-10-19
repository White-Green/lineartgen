"""Review deployed graph weights on existing development cases and a sealed final set."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from attraction_stress import FACTORS, conditions, key, quantize
from lineartgen_runtime.graph import GraphLineartModel, render
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, label


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics(image):
    ink = image[..., 0] < 0
    return dict(
        ink_pixels=int(ink.sum()),
        components=int(label(ink, np.ones((3, 3)))[1]),
        holes=int(label(binary_fill_holes(ink) & ~ink)[1]),
        intermediate_pixels=int(((image > -0.9) & (image < 0.9)).sum()),
    )


def float_infer(model, source, depth, mode, level):
    size = source.shape[0]
    yy, xx = np.mgrid[:size, :size]
    active = xx > 0.6 * size - 0.25 * yy if mode == "partial" else np.ones((size, size), bool)
    noise = active.astype(np.float32)[..., None] * level
    padded = np.pad(source, ((32, 32), (32, 32), (0, 0)), constant_values=1)[None]
    levels = np.pad(
        noise, ((32, 32), (32, 32), (0, 0)), constant_values=0 if mode == "partial" else level
    )[None]
    heads = model.heads(padded, levels, depth)
    image = render(heads[0][0])[32:-32, 32:-32]
    image = np.where(active[..., None], image, source)
    np.testing.assert_array_equal(image[~active], source[~active])
    assert np.isfinite(image).all() and image.min() >= -1 and image.max() <= 1
    return image


def panel(path, rows, labels):
    size = rows[0][1][0].shape[0]
    canvas = Image.new("RGB", (size * len(labels), (size + 28) * len(rows)), "#ddd")
    draw = ImageDraw.Draw(canvas)
    for row, (name, arrays) in enumerate(rows):
        for col, (title, array) in enumerate(zip(labels, arrays, strict=True)):
            draw.text((col * size + 3, row * (size + 28) + 3), f"{name} {title}", fill="black")
            gray = np.rint((array[..., 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(gray).convert("RGB"), (col * size, row * (size + 28) + 28))
    canvas.save(path)


def broad_pages(inputs, output):
    folder = output / "native"
    folder.mkdir(exist_ok=True)
    records = []

    def page(name, cases, depth=3, mode="whole", encoding="float", strength=0.5, deep=False):
        rows = []
        for case in cases:
            with (
                np.load(inputs / "inputs" / (case + ".npz")) as inp,
                np.load(inputs / "outputs" / (case + ".npz")) as old,
                np.load(output / "outputs" / (case + ".npz")) as new,
            ):
                arrays = [inp["source"]]
                if deep:
                    arrays += [new[key("preview", d)] for d in (1, 3, 6)]
                else:
                    arrays += [
                        old[key("control", depth, mode, encoding, strength)],
                        new[key("preview", depth, mode, encoding, strength)],
                    ]
            rows.append((case, arrays))
        labels = ["source"] + (["d1", "d3", "d6"] if deep else ["initial16200", "preview"])
        path = folder / (name + ".png")
        panel(path, rows, labels)
        records.append(dict(file=path.name, sha256=sha(path), cases=cases, native_pixels=True))

    for i in range(0, 64, 4):
        page(f"rough-{i:02d}", [f"stress-{j:02d}-rough" for j in range(i, i + 4)])
    selected = (12, 24, 36, 37, 38, 56, 59, 62)
    for style in ("base", "faint", "dark", "sharp", "soft"):
        for i in (0, 4):
            page(f"{style}-{i}", [f"stress-{j:02d}-{style}" for j in selected[i : i + 4]])
    for i in (0, 4):
        page(
            f"partial-rough-{i}",
            [f"stress-{j:02d}-rough" for j in selected[i : i + 4]],
            mode="partial",
        )
    for i in range(58, 64):
        for j in (0, 3):
            page(f"width-{i}-{j}", [f"width-{i}-{f}" for f in FACTORS[j : j + 3]])
    page("width-depth", [f"width-{i}-2.0" for i in (59, 62, 63)], deep=True)
    for mode in ("whole", "partial"):
        for i in range(0, 8, 2):
            page(
                f"scenes-{mode}-{i}",
                [f"scene-{i}", f"scene-{i + 1}"],
                depth=6,
                mode=mode,
                encoding="uint8",
            )
    extras = [r["name"] for r in read(inputs / "plan.json")["records"] if r["group"] == "extra"]
    for i in range(0, len(extras), 4):
        page(f"extra-{i}", extras[i : i + 4])
        page(f"extra-depth-{i}", extras[i : i + 4], deep=True)
    for level in (0.25, 0.75):
        page(
            f"strength-{level}",
            [f"strength-probe-{i}" for i in (37, 56, 59, 62)],
            strength=level,
        )
    assert len(records) == 57
    write(folder / "manifest.json", dict(records=records, visually_reviewed=False))


def broad(model, inputs, output, limit):
    records = read(inputs / "plan.json")["records"]
    rows, files = [], {}
    for i, r in enumerate(records[:limit]):
        source_path = inputs / r["file"]
        assert sha(source_path) == r["sha256"]
        path = output / "outputs" / (r["name"] + ".npz")
        summary = path.with_suffix(".json")
        if path.exists() and summary.exists():
            saved = read(summary)
            assert sha(path) == saved["sha256"]
            rows += saved["rows"]
        else:
            with np.load(source_path) as z:
                source = z["source"]
            arrays, case_rows = {}, []
            for depth, mode, encoding, level in conditions(r["group"]):
                src = source if encoding == "float" else quantize(source)
                before = time.monotonic()
                image = float_infer(model, src, depth, mode, level)
                name = key("preview", depth, mode, encoding, level)
                arrays[name] = image
                case_rows.append(
                    dict(
                        name=r["name"],
                        condition=name,
                        seconds=time.monotonic() - before,
                        protected_exact=True,
                        **diagnostics(image),
                    )
                )
            np.savez_compressed(path, **arrays)
            write(summary, dict(sha256=sha(path), rows=case_rows))
            rows += case_rows
        files[path.name] = sha(path)
        write(output / "progress.json", dict(cases=i + 1, outputs=len(rows), expected=1180))
    if limit is None:
        assert len(rows) == 1180
        broad_pages(inputs, output)
    write(output / "evaluation.json", dict(records=rows, files=files, protected_exact=True))


def heldout(model, inputs, output, selection):
    selected = read(selection)
    assert selected["checkpoint_sha256"] == model.metadata["checkpoint_sha256"]
    assert selected["heldout_seen_before_selection"] is False
    manifest = read(inputs / "manifest.json")
    results = []
    for r in manifest["records"]:
        path = inputs / r["file"]
        assert sha(path) == r["sha256"]
        with np.load(path) as z:
            source = z["source"]
        n = r["size"]
        scribble = np.zeros((n, n, 4), np.uint8)
        scribble[..., 3] = np.rint((1 - source[..., 0]) * 127.5).astype(np.uint8)
        source = 1 - 2 * scribble[..., 3:4].astype(np.float32) / 255
        options = dict(
            scribble=scribble.tobytes(),
            lineart=bytes(n * n * 4),
            width=n,
            height=n,
            strength=0.5,
            seed=42,
        )
        arrays, case_rows = {}, []
        for depth in (1, 3, 6):
            before = time.monotonic()
            raw = model.infer(**options, denoise_steps=depth)
            bgra = np.frombuffer(raw, np.uint8).reshape(n, n, 4)
            assert not bgra[..., :3].any()
            image = 1 - 2 * bgra[..., 3:4].astype(np.float32) / 255
            arrays[f"d{depth}"] = image
            case_rows.append(
                dict(depth=depth, seconds=time.monotonic() - before, **diagnostics(image))
            )
        yy, xx = np.mgrid[:n, :n]
        protected = xx <= 0.6 * n - 0.25 * yy
        lineart = np.zeros_like(scribble)
        lineart[protected, :3] = 255 - scribble[protected, 3:4]
        lineart[protected, 3] = 255
        raw = model.infer(**(options | dict(lineart=lineart.tobytes())), denoise_steps=3)
        bgra = np.frombuffer(raw, np.uint8).reshape(n, n, 4)
        np.testing.assert_array_equal(bgra[protected, 3], scribble[protected, 3])
        arrays["partial"] = 1 - 2 * bgra[..., 3:4].astype(np.float32) / 255
        filename = f"{r['index']:02d}"
        np.savez_compressed(output / "outputs" / (filename + ".npz"), **arrays)
        panel(
            output / (filename + ".png"),
            [(r["kind"], [source, *[arrays[f"d{d}"] for d in (1, 3, 6)]])],
            ["source", "d1", "d3", "d6"],
        )
        panel(
            output / (filename + "-partial.png"),
            [(r["kind"], [source, arrays["partial"]])],
            ["source", "d3 partial"],
        )
        results.append(
            dict(index=r["index"], kind=r["kind"], whole=case_rows, protected_exact=True)
        )
    write(
        output / "evaluation.json",
        dict(records=results, selection_sha256=sha(selection), visually_reviewed=False),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("broad", "heldout"))
    for name in ("assets", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "outputs").mkdir(exist_ok=True)
    binding = dict(
        model_sha256=sha(args.assets / "graph-model.json"),
        inputs=str(args.inputs.resolve()),
        phase=args.phase,
        limit=args.limit,
    )
    if (args.output / "binding.json").exists():
        assert read(args.output / "binding.json") == binding
    else:
        write(args.output / "binding.json", binding)
    runtime = GraphLineartModel(args.assets / "graph-model.json", provider="cpu")
    if args.phase == "broad":
        broad(runtime, args.inputs, args.output, args.limit)
    else:
        if args.selection is None:
            parser.error("heldout requires a recorded checkpoint selection before viewing outputs")
        heldout(runtime, args.inputs, args.output, args.selection)
