"""Continuous-width synthetic rectangles with exact R1 graph teachers."""

from pathlib import Path

import numpy as np
from attraction_graph import decode, logits_from_target, target
from campaign_protocol import read, sha, write
from centerline_grid import render
from learned_centerline_study import crop
from procedural_shape_data import render as render_source

SEED = 2026101814
COUNT = 1024
CACHE = Path("/tmp/lineart-continuous-width-cache-20261018")


def record(index):
    if not 0 <= index < COUNT:
        raise ValueError(index)
    rng = np.random.default_rng([SEED, index])
    if index >= 896:
        width, length = map(int, rng.integers(128, 209, size=2))
        left, top = int(rng.integers(8, 249 - width)), int(rng.integers(8, 249 - length))
        return dict(
            index=index,
            kind="fill",
            filled=True,
            width=width,
            length=length,
            angle=0.0,
            center=[left + width / 2, top + length / 2],
            sigma=1.2,
            opacity=0.4,
        )
    low, high = (4, 16) if index < 512 else ((16, 48) if index < 768 else (48, 96))
    while True:
        width = float(rng.uniform(low, high))
        length = float(rng.uniform(max(32, 1.05 * width), 208))
        kind = int(rng.integers(4))
        angle = float(rng.integers(4) * np.pi / 2)
        if kind == 1:
            angle += float(rng.uniform(-np.pi / 60, np.pi / 60))
        elif kind >= 2:
            angle = float(rng.uniform(-np.pi, np.pi))
        c, s = abs(np.cos(angle)), abs(np.sin(angle))
        rx, ry = (c * width + s * length) / 2, (s * width + c * length) / 2
        if max(rx, ry) <= 120:
            break
    return dict(
        index=index,
        kind=f"bar{low}-{high}",
        filled=False,
        width=width,
        length=length,
        angle=angle,
        center=[float(rng.uniform(8 + rx, 248 - rx)), float(rng.uniform(8 + ry, 248 - ry))],
        sigma=1.2,
        opacity=0.4,
    )


def geometry(r):
    empty_s = np.empty((0, 2, 2), np.float32)
    empty_w = np.empty((0, 2), np.float32)
    empty_r = np.empty((0, 4), np.float32)
    center = np.array(r["center"], np.float32)
    if r["filled"]:
        half = np.array([r["width"], r["length"]], np.float32) / 2
        rect = np.concatenate([center - half, center + half])[None]
        return empty_s, empty_w, rect
    axis = np.array([np.sin(r["angle"]), np.cos(r["angle"])], np.float32)
    reach = r["length"] / 2 - 1
    return (
        np.stack([center - axis * reach, center + axis * reach])[None],
        np.full((1, 2), 2, np.float32),
        empty_r,
    )


def build(root=CACHE):
    assert not root.exists()
    root.mkdir()
    (root / "inputs").mkdir()
    recipe = Path(__file__)
    (root / "recipe.py").write_bytes(recipe.read_bytes())
    write(
        root / "plan.json",
        dict(
            seed=SEED,
            count=COUNT,
            code_sha256=sha(recipe),
            training_updates=0,
            model_predictions=0,
            groups={"bar4-16": 512, "bar16-48": 256, "bar48-96": 128, "fill": 128},
        ),
    )
    rows = []
    for i in range(COUNT):
        r = record(i)
        source = render_source(r)["source"]
        s, w, rect = geometry(r)
        data = dict(source=source, segments=s, widths=w, rectangles=rect)
        stages = []
        for level in range(6):
            t, stats = target(s + 32, w, rect + 32, 320, level, base_radius=1)
            assert not stats["unrepresented_paths"]
            data[f"smalltarget{level}"] = t
            stages.append(stats)
        oracle, _ = decode(logits_from_target(data["smalltarget0"], base_radius=1), base_radius=1)
        reference = render(s, w, rect, 256)
        np.testing.assert_allclose(crop(oracle), reference, atol=1e-4, rtol=1e-4)
        data["graph_oracle"] = oracle
        p = root / "inputs" / f"{i:04d}.npz"
        np.savez_compressed(p, **data)
        rows.append(
            dict(
                r,
                file=str(p.relative_to(root)),
                sha256=sha(p),
                stages=stages,
                roundtrip_max=float(abs(crop(oracle) - reference).max()),
            )
        )
        if (i + 1) % 64 == 0:
            print(dict(prepared=i + 1, total=COUNT), flush=True)
    write(
        root / "manifest.json",
        dict(
            records=rows,
            count=COUNT,
            max_roundtrip_error=max(x["roundtrip_max"] for x in rows),
            training_updates=0,
            model_predictions=0,
        ),
    )


def load():
    result = []
    for r in read(CACHE / "manifest.json")["records"]:
        p = CACHE / r["file"]
        assert sha(p) == r["sha256"]
        with np.load(p) as z:
            result.append({k: z[k] for k in z.files})
    return result


if __name__ == "__main__":
    build()
