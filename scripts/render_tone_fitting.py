"""Fixed native crops for the tone-weight fitting comparison."""

import argparse
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import verify
from PIL import Image, ImageDraw


def render(root):
    verify(root)
    plan = read(root / "plan.json")
    final = root / "evaluation/step-1200"
    original = root / "evaluation/reference-1200"
    out = final / "visual-review"
    out.mkdir(exist_ok=True)
    cache, files = {}, {}

    def prediction(name, mode, split, steps, partial, pair, strength):
        folder = original if name == "warm_detail16" else final
        key = folder / f"{name}-{mode}-{split}-n{steps}-{'partial' if partial else 'whole'}.npz"
        if key not in cache:
            cache[key] = np.load(key)["images"]
        return cache[key][pair + (57 if strength == 0.5 else 0)]

    def sheet(filename, specifications, columns):
        canvas = Image.new("RGB", ((len(columns) + 2) * 160, len(specifications) * 180), "white")
        draw = ImageDraw.Draw(canvas)
        crops = []
        for row, spec in enumerate(specifications):
            split, pair, mode, strength, partial = (
                spec[k] for k in ("split", "pair", "mode", "strength", "partial")
            )
            data = np.load(root / f"{split}.npz")
            clean, source = data["clean"][pair], data["source"][pair]
            meta = read(root / f"{split}.json")[pair]
            ys, xs = np.where(clean[..., 0] < 0.96)
            if len(ys):
                y, x = (int(np.clip(np.mean(a) - 80, 0, 352)) for a in (ys, xs))
            else:
                y = x = 96 if partial else 176
            target = clean.copy()
            if partial:
                target[:128] = source[:128]
                target[:, :170] = source[:, :170]
            images = [target, source] + [
                prediction(n, mode, split, steps, partial, pair, strength) for n, steps in columns
            ]
            labels = ["target", "source"] + [
                f"{n.removeprefix('warm_tone_')} n{steps}" for n, steps in columns
            ]
            for col, (label, image) in enumerate(zip(labels, images, strict=True)):
                gray = np.uint8(np.rint(np.clip((image[..., 0] + 1) * 127.5, 0, 255)))
                canvas.paste(
                    Image.fromarray(gray[y : y + 160, x : x + 160]).convert("RGB"),
                    (col * 160, row * 180 + 20),
                )
                prefix = meta["kind"]
                if prefix == "blank":
                    prefix += f" {strength} {'P' if partial else 'W'}"
                draw.text((col * 160 + 2, row * 180 + 2), f"{prefix} {label}", fill="black")
            crops.append(dict(spec, crop=[x, y, x + 160, y + 160]))
        path = out / filename
        canvas.save(path)
        files[filename] = dict(
            sha256=sha(path),
            columns=columns,
            crops=crops,
            inspected=False,
            learned_float_inputs=True,
            actual_bgra_api=False,
        )

    main = [("warm_detail16", 1), ("warm_tone_sqrt", 1), ("warm_tone_plain", 1)]
    for split, family, strength, mode in (
        ("train", "blur", 0.1, "raw"),
        ("train", "blur", 0.5, "raw"),
        ("probe", "blur", 0.1, "raw"),
        ("probe", "blur", 0.5, "raw"),
        ("probe", "tone", 0.1, "raw"),
        ("probe", "clean", 0.1, "raw"),
        ("probe", "blur", 0.1, "ema"),
    ):
        offset = {"clean": 0, "tone": 19, "blur": 38}[family]
        specs = [
            dict(split=split, pair=offset + i, mode=mode, strength=strength, partial=False)
            for i in range(9)
        ]
        sheet(f"{split}-{family}-{strength}-{mode}.png", specs, main)
    repeated = [
        (n, steps)
        for n in ("warm_detail16", "warm_tone_sqrt", "warm_tone_plain")
        for steps in (1, 2, 5)
    ]
    for strength in (0.1, 0.5):
        specs = [
            dict(split="probe", pair=38 + i, mode="raw", strength=strength, partial=False)
            for i in range(9)
        ]
        sheet(f"probe-blur-{strength}-repeat.png", specs, repeated)
        specs = [dict(x, partial=True) for x in specs]
        sheet(f"probe-blur-{strength}-partial.png", specs, main)
    blanks = [
        dict(split="probe", pair=56, mode="raw", strength=strength, partial=partial)
        for partial in (False, True)
        for strength in (0.1, 0.5)
    ]
    sheet("probe-blank-repeat.png", blanks, repeated)
    assert len(files) == 12
    write(
        out / "manifest.json",
        dict(
            files=files,
            plan_sha256=sha(root / "plan.json"),
            script_sha256=sha(__file__),
            inspection_complete=False,
            declared_primary=plan["primary"],
            adoption=False,
        ),
    )
    print(dict(native_sheets=len(files), inspection_complete=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    render(p.parse_args().root.resolve())
