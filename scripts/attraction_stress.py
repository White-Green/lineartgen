"""ID: fixed HY weights on established broad synthetic stress inputs."""

import argparse
import datetime
import itertools
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from attraction_graph import decode
from broad_attraction_study import RADII, to_image
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from learned_centerline_model import infer_heads, load_weights
from learned_centerline_study import SEED, crop, halo, make_model, strength
from PIL import Image, ImageDraw
from plausible_study import digest
from projected_edge_gate import gate
from rough_family_data import rough_source

HY = Path("../jax-broad-attraction-20261018/runs/broad-attraction-20261018").resolve()
HI = Path("../jax-learned-coarse-axis-20261017/runs/learned-coarse-axis-20261017").resolve()
EL = Path("../jax-width-response-20261015/runs/width-response-20261015").resolve()
GK = Path("../jax-bar-fill-transfer-20261017/runs/bar-fill-transfer-20261017").resolve()
GO = Path(
    "../jax-wide-context-identifiability-20261017/runs/wide-context-identifiability-20261017"
).resolve()
DOC = Path("docs/attraction-stress-plan-20261018.md")
FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
METHODS = ("control", "sparse")


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def quantize(a):
    return (np.rint((a + 1) * 127.5) / 127.5 - 1).astype(np.float32)


def conditions(group):
    if group == "stress":
        return [(3, m, "float", 0.5) for m in ("whole", "partial")]
    if group == "rough_uint8":
        return [(3, "whole", "uint8", 0.5)]
    if group in ("width", "extra"):
        return [(d, "whole", "float", 0.5) for d in (1, 3, 6)] + [(3, "partial", "float", 0.5)]
    if group == "scene":
        return list(itertools.product((3, 6), ("whole", "partial"), ("float", "uint8"), (0.5,)))
    if group == "strength":
        return list(itertools.product((3,), ("whole", "partial"), ("float",), (0.25, 0.75)))
    raise ValueError(group)


def key(method, depth=3, mode="whole", encoding="float", level=0.5):
    return f"{method}-d{depth}-{mode}-{encoding}-t{level}"


def prepare(root):
    assert not root.exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    for f, h in read(HY / "completion.json")["evidence"].items():
        assert sha(HY / f) == h
    root.mkdir(parents=True)
    (root / "inputs").mkdir()
    bindings = {}
    records = []

    def bind(p):
        bindings[str(p.resolve())] = sha(p)

    def add(name, source, group, **metadata):
        source = np.asarray(source, np.float32)
        assert source.shape in ((256, 256, 1), (512, 512, 1))
        assert np.isfinite(source).all() and source.min() >= -1 and source.max() <= 1
        p = root / "inputs" / (name + ".npz")
        np.savez_compressed(p, source=source)
        records.append(
            dict(name=name, group=group, file=str(p.relative_to(root)), sha256=sha(p), **metadata)
        )

    for p in [
        HY / "completion.json",
        HY / "supplemental-binding.json",
        HY / "config.json",
        HY / "initial.npz",
        HI / "stress-plan.json",
        HI / "scene512/plan.json",
        GK / "plan.json",
        GK / "inputs.npz",
        GO / "plan.json",
        GO / "inputs.npz",
    ]:
        bind(p)
    for method in METHODS:
        bind(HY / f"snapshots/step-6000/{method}/raw.npz")
    stress = read(HI / "stress-plan.json")
    for r in stress["records"]:
        p = HI / r["file"]
        bind(p)
        with np.load(p) as z:
            mixture = z["mixture"]
        for style, options in stress["corruptions"].items():
            source = rough_source(mixture, options[r["index"]])
            add(
                f"stress-{r['index']:02d}-{style}",
                source,
                "stress",
                index=r["index"],
                style=style,
                kind=r["kind"],
            )
            if style == "rough":
                add(f"uint8-{r['index']:02d}", source, "rough_uint8", index=r["index"])
    for i in range(58, 64):
        for factor in FACTORS:
            p = EL / f"evaluation/wide-{i}-factor-{factor}.npz"
            bind(p)
            with np.load(p) as z:
                add(f"width-{i}-{factor}", z["source"], "width", index=i, factor=factor)
    for r in read(HI / "scene512/plan.json")["records"]:
        p = HI / "scene512" / r["file"]
        bind(p)
        assert sha(p) == r["sha256"]
        with np.load(p) as z:
            np.testing.assert_array_equal(quantize(z["source"]), z["quantized"])
            add(f"scene-{r['index']}", z["source"], "scene", index=r["index"], style=r["condition"])
    with np.load(GK / "inputs.npz") as z:
        for i, c in enumerate(read(GK / "plan.json")["cases"]):
            add("extra-" + c["name"], z["source"][i, 0], "extra", case=c)
    with np.load(GO / "inputs.npz") as z:
        for i, name in enumerate(("wide87", "fill160")):
            add("extra-" + name, z["source"][i], "extra")
    for r in read(HY / "plan.json")["records"]:
        p = HY / r["file"]
        bind(p)
        with np.load(p) as z:
            add("strength-" + r["name"], z["source"], "strength")
    for p in HY.joinpath("outputs").glob("*.npz"):
        bind(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    count = sum(len(conditions(r["group"])) * 2 for r in records)
    assert len(records) == 527 and count == 2360, (len(records), count)
    code = [
        "scripts/attraction_stress.py",
        "scripts/broad_attraction_study.py",
        "scripts/learned_centerline_model.py",
        "scripts/learned_coarse_axis_model.py",
        "scripts/learned_centerline_study.py",
        "scripts/attraction_graph.py",
        "scripts/projected_edge_gate.py",
        "scripts/cue_capacity.py",
        "scripts/rough_family_data.py",
    ]
    write(
        root / "plan.json",
        dict(
            commit=commit,
            created_at=now(),
            records=records,
            bindings=bindings,
            code={p: sha(Path(p)) for p in code},
            document_sha256=sha(DOC),
            archive_sha256=sha(root / "source.tar.gz"),
            outputs=count,
            training_updates=0,
            fresh_used=False,
            adoption=False,
            models="HY raw6000 R4/R1",
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for f, h in plan["bindings"].items():
        assert sha(Path(f)) == h, f
    for f, h in plan["code"].items():
        assert sha(Path(f)) == h
    for r in plan["records"]:
        assert sha(root / r["file"]) == r["sha256"]
    assert sha(DOC) == plan["document_sha256"]
    assert sha(root / "source.tar.gz") == plan["archive_sha256"]
    return plan


def prepared(source, depth, mode, level=0.5, target=None):
    mask = strength(source.shape[0], mode == "partial") * (level / 0.5)
    src = halo(source)
    p = prepare_inputs(
        np.ones_like(src)[None] if target is None else target[None],
        src[None],
        halo(mask, 0 if mode == "partial" else level)[None],
        jax.random.PRNGKey(SEED + 1),
        depth,
    )
    return p, mask


def infer(model, source, depth, mode, level, radius):
    p, mask = prepared(source, depth, mode, level)
    h = np.asarray(infer_heads(model, p)[0])[0]
    assert np.isfinite(h).all()
    gated, rejected = gate(h, base_radius=radius)
    image, stats = decode(gated, base_radius=radius)
    image = np.where(mask > 0, crop(image), source)
    np.testing.assert_array_equal(image[mask == 0], source[mask == 0])
    assert np.isfinite(image).all() and image.min() >= -1 and image.max() <= 1
    return image, h, dict(stats, rejected_edges=rejected, protected_exact=True)


def models():
    result = {}
    for method in METHODS:
        m, _ = make_model(HY)
        load_weights(m, HY / f"snapshots/step-6000/{method}/raw.npz")
        result[method] = m
    return result


def preflight(root):
    check(root)
    ms = models()
    rows = []
    for name in ("probe-37", "probe-62", "scene-7"):
        with np.load(HY / "inputs" / (name + ".npz")) as z:
            source, teacher = z["source"], z["graph_oracle"]
        for method, model in ms.items():
            image, h, _ = infer(model, source, 3, "whole", 0.5, RADII[method])
            p, _ = prepared(source, 3, "whole", target=teacher)
            np.testing.assert_array_equal(h, np.asarray(infer_heads(model, p)[0])[0])
            with np.load(HY / "outputs" / (name + ".npz")) as z:
                previous_h, previous = z[method + "-head-d3"], z[method + "-d3"]
            np.testing.assert_allclose(h, previous_h, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(image, previous, atol=1e-5, rtol=1e-5)
            rows.append(
                dict(
                    name=name,
                    method=method,
                    teacher_isolation_exact=True,
                    head_max_difference=float(abs(h - previous_h).max()),
                    image_max_difference=float(abs(image - previous).max()),
                )
            )
    write(root / "preflight.json", dict(passed=True, rows=rows, atol=1e-5, rtol=1e-5))


def evaluate(root):
    plan = check(root)
    assert read(root / "preflight.json")["passed"]
    ms = models()
    states = {k: digest(nnx.state(m, nnx.Param)) for k, m in ms.items()}
    out = root / "outputs"
    out.mkdir()
    rows, outputs = [], {}
    for i, r in enumerate(plan["records"]):
        with np.load(root / r["file"]) as z:
            source = z["source"]
        arrays = {}
        for depth, mode, encoding, level in conditions(r["group"]):
            src = source if encoding == "float" else quantize(source)
            for method, model in ms.items():
                image, _, stats = infer(model, src, depth, mode, level, RADII[method])
                k = key(method, depth, mode, encoding, level)
                arrays[k] = image
                rows.append(dict(name=r["name"], condition=k, graph=stats))
        p = out / (r["name"] + ".npz")
        np.savez_compressed(p, **arrays)
        outputs[p.name] = sha(p)
        if (i + 1) % 16 == 0:
            status = dict(cases=i + 1, model_outputs=len(rows), total=plan["outputs"])
            write(root / "progress.json", status)
            print(status, flush=True)
    assert len(rows) == plan["outputs"]
    assert states == {k: digest(nnx.state(m, nnx.Param)) for k, m in ms.items()}
    write(
        root / "evaluation.json",
        dict(
            count=len(rows),
            records=rows,
            outputs=outputs,
            unchanged_states=True,
            protected_exact=True,
            training_updates=0,
            fresh_used=False,
        ),
    )
    pages(root)


def pages(root):
    out = root / "native"
    out.mkdir()
    records = []

    def page(name, cases, cols=None):
        if cols is None:
            cols = [("source", None), ("HY R4", key("control")), ("HY R1", key("sparse"))]
        with np.load(root / "inputs" / (cases[0] + ".npz")) as z:
            n = z["source"].shape[0]
        canvas = Image.new("RGB", (n * len(cols), 24 + (n + 20) * len(cases)), "#ddd")
        draw = ImageDraw.Draw(canvas)
        for col, (label, _) in enumerate(cols):
            draw.text((col * n + 3, 4), label, fill="black")
        for row, case in enumerate(cases):
            with (
                np.load(root / "inputs" / (case + ".npz")) as inp,
                np.load(root / "outputs" / (case + ".npz")) as pred,
            ):
                for col, (_, k) in enumerate(cols):
                    a = inp["source"] if k is None else pred[k]
                    canvas.paste(to_image(a), (col * n, 44 + row * (n + 20)))
            draw.text((3, 24 + row * (n + 20)), case, fill="black")
        p = out / (name + ".png")
        canvas.save(p)
        records.append(
            dict(file=p.name, sha256=sha(p), native_pixels=True, cases=cases, columns=cols)
        )

    for i in range(0, 64, 4):
        page(f"rough-{i:02d}", [f"stress-{j:02d}-rough" for j in range(i, i + 4)])
    selected = (12, 24, 36, 37, 38, 56, 59, 62)
    for style in ("base", "faint", "dark", "sharp", "soft"):
        for i in range(0, 8, 4):
            page(f"{style}-{i}", [f"stress-{j:02d}-{style}" for j in selected[i : i + 4]])
    for i in range(0, 8, 4):
        page(
            f"partial-rough-{i}",
            [f"stress-{j:02d}-rough" for j in selected[i : i + 4]],
            [("source", None)] + [(m, key(m, mode="partial")) for m in METHODS],
        )
    for i in range(58, 64):
        for j in (0, 3):
            page(f"width-{i}-{j}", [f"width-{i}-{f}" for f in FACTORS[j : j + 3]])
    deep = [("source", None)] + [(f"R1 d{d}", key("sparse", d)) for d in (1, 3, 6)]
    page("width-depth", [f"width-{i}-2.0" for i in (59, 62, 63)], deep)
    for mode in ("whole", "partial"):
        for i in range(0, 8, 2):
            page(
                f"scenes-{mode}-{i}",
                [f"scene-{j}" for j in (i, i + 1)],
                [("source", None)] + [(m + " uint8 d6", key(m, 6, mode, "uint8")) for m in METHODS],
            )
    extras = [r["name"] for r in read(root / "plan.json")["records"] if r["group"] == "extra"]
    for i in range(0, len(extras), 4):
        page(f"extra-{i}", extras[i : i + 4])
        page(f"extra-depth-{i}", extras[i : i + 4], deep)
    for t in (0.25, 0.75):
        page(
            f"strength-{t}",
            ["strength-probe-" + x for x in ("37", "56", "59", "62")],
            [("source", None)] + [(m, key(m, level=t)) for m in METHODS],
        )
    assert len(records) == 57, len(records)
    write(out / "manifest.json", records)


def finalize(root):
    p = check(root)
    e = read(root / "evaluation.json")
    m = read(root / "native/manifest.json")
    obs = read(root / "native/observations.json")
    assert len(m) == 57 and set(obs) == {r["file"] for r in m} and all(obs.values())
    assert not read(root / "decision.json")["adoption"]
    files = [
        "plan.json",
        "preflight.json",
        "evaluation.json",
        "decision.json",
        "native/manifest.json",
        "native/observations.json",
    ]
    for f, h in e["outputs"].items():
        assert sha(root / "outputs" / f) == h
        files.append("outputs/" + f)
    for r in m:
        assert sha(root / "native" / r["file"]) == r["sha256"]
        files.append("native/" + r["file"])
    write(
        root / "completion.json",
        dict(
            stage="complete_fixed_attraction_stress",
            commit=p["commit"],
            new_conditions=e["count"],
            native_pages_reviewed=57,
            training_updates=0,
            fresh_used=False,
            adoption=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "preflight", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    a = parser.parse_args()
    start, t = now(), time.monotonic()
    globals()[a.phase](a.root.resolve())
    print(
        dict(phase=a.phase, started_at=start, ended_at=now(), seconds=time.monotonic() - t),
        flush=True,
    )


if __name__ == "__main__":
    main()
