"""Freeze and inspect proposed training teachers before any new model training."""

import argparse
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from axis_family_data import MINIMUM_WIDTH_LENGTH_RATIO, axis_payload, axis_spec
from campaign_protocol import read, sha, write
from choice_readout_study import batch, check, load_data
from PIL import Image, ImageDraw
from subpixel_field_data import candidate_fields
from subpixel_field_model import coverage

ROOT = Path("../jax-choice-readout-20261016/runs/choice-readout-20261016-retry2").resolve()
DOC = Path("docs/axis-family-preview-plan-20261016.md")
p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
r = p.parse_args().root.resolve()
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
check(ROOT)
r.mkdir(parents=True, exist_ok=False)
data = load_data(ROOT)
plan = read(ROOT / "sampling-plan.json")
widths = read(ROOT / "width-plan.json")
rows = []
selected = {}
for step, (selection, wr) in enumerate(zip(plan, widths, strict=True)):
    for item, index in enumerate(selection["indices"]):
        spec = axis_spec(
            data["records"][index],
            wr["records"][item]["primary"]["new"],
            overlay=selection["overlays"][item]["enabled"],
        )
        if spec["allowed"]:
            row = dict(
                step=step,
                item=item,
                index=index,
                depth=selection["depth"],
                width=spec["width"],
                length=spec["length"],
                ratio=spec["ratio"],
            )
            rows.append(row)
            selected.setdefault(selection["depth"], row)
assert len(rows) == 243 and len(selected) == 6
write(
    r / "eligibility.json",
    dict(
        ratio=MINIMUM_WIDTH_LENGTH_RATIO, examples=rows, selected=[selected[d] for d in range(1, 7)]
    ),
)
tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
files = [Path(f).resolve() for f in tracked if f.endswith((".py", ".toml")) or f == "uv.lock"]
files += [DOC.resolve(), ROOT / "plan.json", ROOT / "source.json", r / "eligibility.json"]
prior = Path("../jax-axis-supervision-20261016/runs/axis-supervision-20261016-retry1")
files += [prior / "completion.json", prior / "decision.json", prior / "observations.json"]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
subprocess.run(
    ["git", "archive", "--format=tar.gz", "-o", str(r / "source.tar.gz"), commit], check=True
)
write(
    r / "plan.json",
    dict(
        commit=commit,
        bindings={str(p.resolve()): sha(p) for p in files},
        archive_sha256=sha(r / "source.tar.gz"),
        teacher_only=True,
        model_training=False,
        adoption=False,
        selection="First eligible example at each depth",
    ),
)
images = []
bindings = {}
for depth in range(1, 7):
    row = selected[depth]
    step = row["step"]
    item = row["item"]
    prepared, bases, directions = batch(data, step, "choice")
    axis, audit = axis_payload(data, plan[step], widths[step]["records"], bases, directions)
    assert audit[item]["allowed"]
    original = candidate_fields(bases[0], directions, depth)[0]
    alternative = candidate_fields(axis[0], axis[1], depth)[0]
    target = np.asarray(coverage(original[:, 4]))[item]
    other = np.asarray(coverage(alternative[:, 4]))[item]
    source = np.asarray(prepared.guides[0])[item]
    mask = np.asarray(prepared.noise_levels[0])[item]
    assert np.isfinite(other).all() and not np.array_equal(other, target)
    path = r / f"depth-{depth}.npz"
    np.savez_compressed(
        path,
        source=source,
        original=target,
        alternative=other,
        mask=mask,
        original_field=np.asarray(original[item, 4]),
        alternative_field=np.asarray(alternative[item, 4]),
    )
    bindings[str(path)] = sha(path)
    images.append((row, [source, target, other, 1 - 2 * (mask > 0).astype(np.float32)]))
pages = []
for start in (0, 2, 4):
    canvas = Image.new("RGB", (1024, 584), "white")
    pen = ImageDraw.Draw(canvas)
    for rr, (row, values) in enumerate(images[start : start + 2]):
        pen.text(
            (4, rr * 292 + 4),
            f"depth{row['depth']} / step{row['step']} item{row['item']} ratio{row['ratio']:.3f}",
            fill="black",
        )
        for col, (name, value) in enumerate(
            zip(["source", "original", "perpendicular", "active mask"], values, strict=True)
        ):
            pen.text((col * 256 + 4, rr * 292 + 20), name, fill="black")
            px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
            canvas.paste(Image.fromarray(px).convert("RGB"), (col * 256, rr * 292 + 36))
    path = r / f"native-depth-{start + 1}-{start + 2}.png"
    canvas.save(path)
    pages.append(dict(file=path.name, sha256=sha(path)))
write(
    r / "manifest.json", dict(records=pages, native_scale=True, arrays=bindings, teacher_only=True)
)
print(dict(eligible=243, selected=list(selected.values()), pages=3), flush=True)
