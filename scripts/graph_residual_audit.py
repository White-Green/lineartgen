"""IH: fixed-weight teacher-assisted decomposition, never an inference method."""

import argparse
import copy
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from attraction_graph import decode, logits_from_target, target
from attraction_stress import EL, HI, infer
from campaign_protocol import read, sha, write
from centerline_grid import render
from flax import nnx
from learned_centerline_model import load_weights
from learned_centerline_study import crop, digest, make_model
from opaque_geometry import geometry
from PIL import Image, ImageDraw
from plausible_data import raster
from rough_family_data import BASE, rough_source

IG = Path("../jax-fill-hard-negative-20261018/runs/fill-hard-negative-20261018-retry1").resolve()
DOC = Path("docs/graph-residual-audit-plan-20261019.md")
STANDARD = ("probe-36", "probe-38", "probe-37", "probe-56", "scene-1", "scene-5")
BROAD = ("width-58-0.5", "width-61-0.25", "width-63-0.5", "stress-41-rough")
MODES = {
    "raw": (),
    "GT vertex": (0,),
    "GT position at GT vertices": (1, 2),
    "GT width at GT vertices": (3,),
    "GT edges": (4, 5, 6, 7),
    "GT fill": (8,),
    "GT vertex+edges": (0, 4, 5, 6, 7),
    "GT vertex+position": (0, 1, 2),
    "GT graph": tuple(range(8)),
    "full oracle": tuple(range(9)),
}


def replace(head, truth, mode):
    result = head.copy()
    oracle = logits_from_target(truth, base_radius=1.0)
    valid = truth[..., 0] > 0.5
    for channel in MODES[mode]:
        if channel in (1, 2, 3) and mode not in ("GT graph", "full oracle"):
            result[..., channel] = np.where(valid, oracle[..., channel], result[..., channel])
        else:
            result[..., channel] = oracle[..., channel]
    return result


def draw(head):
    from projected_edge_gate import gate

    gated, rejected = gate(head, base_radius=1.0)
    image, stats = decode(gated, base_radius=1.0)
    stats["rejected_edges"] = rejected
    return crop(image), stats


def pack(source, s, w, rect):
    t, stats = target(s + 32, w, rect + 32, source.shape[0] + 64, base_radius=1.0)
    assert not stats["unrepresented_paths"]
    image, _ = draw(logits_from_target(t, base_radius=1.0))
    reference = render(s, w, rect, source.shape[0])
    # Projected endpoints may shorten; retain the difference instead of redefining an oracle.
    return dict(source=source, truth=t, oracle=image, reference=reference), stats


def prepare(root):
    assert not root.exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    for f, h in read(IG / "completion.json")["evidence"].items():
        assert sha(IG / f) == h
    root.mkdir(parents=True)
    (root / "inputs").mkdir()
    files = [
        IG / "completion.json",
        IG / "supplemental-binding.json",
        IG / "config.json",
        IG / "snapshots/step-1200/hardnegative/raw.npz",
        EL / "plan.json",
        HI / "stress-plan.json",
    ]
    records = []

    def add(name, d, output, fresh_head=False, **metadata):
        p = root / "inputs" / (name + ".npz")
        np.savez_compressed(p, **d)
        files.append(output)
        records.append(
            dict(
                name=name,
                input=str(p.relative_to(root)),
                sha256=sha(p),
                output=str(output),
                output_sha256=sha(output),
                new_head=fresh_head,
                **metadata,
            )
        )

    for name in STANDARD:
        inp, output = IG / "inputs" / (name + ".npz"), IG / "outputs" / (name + ".npz")
        files.append(inp)
        with np.load(inp) as z:
            d = dict(
                source=z["source"],
                truth=z["smalltarget0"],
                oracle=z["smalloracle"],
                reference=z["original_oracle"],
            )
        add(name, d, output, saved_key="hardnegative-d3", head_key="hardnegative-head-d3")
    extra = read(IG / "training-data.json")["extra"]
    files.append(IG / "training-data.json")
    for i in (769, 896):
        p = Path(extra[i]["path"])
        assert sha(p) == extra[i]["sha256"]
        files.append(p)
        with np.load(p) as z:
            image, _ = draw(logits_from_target(z["smalltarget0"], base_radius=1.0))
            d = dict(
                source=z["source"],
                truth=z["smalltarget0"],
                oracle=image,
                reference=render(z["segments"], z["widths"], z["rectangles"], 256),
            )
        add(
            f"train-{i}",
            d,
            IG / "fit/outputs" / f"{i:04d}.npz",
            saved_key="hardnegative-d3",
            head_key="hardnegative-head-d3",
        )
    for name in BROAD:
        inp = IG / "broad/inputs" / (name + ".npz")
        files.append(inp)
        with np.load(inp) as z:
            source = z["source"]
        if name.startswith("width"):
            _, index, f = name.split("-")
            r = next(r for r in read(EL / "plan.json")["records"] if r["index"] == int(index))
            md = copy.deepcopy(r["metadata"])
            for path in md["path_geometry"]:
                path["widths"] = (np.asarray(path["widths"]) * float(f)).tolist()
            mixture = np.stack(
                [raster(md, np.asarray(r["direction"]), o, 1) for o in (-2.0, 2.0)]
            ).mean(0)
            np.testing.assert_array_equal(source, rough_source(mixture, BASE))
            s, w, rect = geometry(md, kind=r["kind"])
        else:
            plan = read(HI / "stress-plan.json")
            r = next(r for r in plan["records"] if r["index"] == 41)
            original = HI / r["file"]
            files.append(original)
            with np.load(original) as z:
                np.testing.assert_array_equal(
                    source, rough_source(z["mixture"], plan["corruptions"]["rough"][41])
                )
            s, w, rect = geometry(r["metadata"], kind=r["kind"])
        d, st = pack(source, s, w, rect)
        add(
            name,
            d,
            IG / "broad/outputs" / (name + ".npz"),
            fresh_head=True,
            saved_key="hardnegative-d3-whole-float-t0.5",
            teacher_stats=st,
            source_regeneration_exact=True,
        )
    assert len(records) == 12
    files += [
        DOC,
        Path(__file__),
        Path("scripts/attraction_graph.py"),
        Path("scripts/projected_edge_gate.py"),
        Path("scripts/attraction_stress.py"),
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            modes=MODES,
            training_updates=0,
            new_predictions=4,
            reused_predictions=8,
            renders=120,
            native_pages=12,
            evidence={str(p.resolve()): sha(p) for p in files},
            archive_sha256=sha(root / "source.tar.gz"),
            fresh_used=False,
            adoption=False,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for p, h in plan["evidence"].items():
        assert sha(Path(p)) == h, p
    for r in plan["records"]:
        assert sha(root / r["input"]) == r["sha256"]
    assert sha(root / "source.tar.gz") == plan["archive_sha256"]
    return plan


def evaluate(root):
    plan = check(root)
    model, _ = make_model(IG)
    load_weights(model, IG / "snapshots/step-1200/hardnegative/raw.npz")
    before = digest(nnx.state(model, nnx.Param))
    (root / "outputs").mkdir()
    (root / "native").mkdir()
    rows, manifest = [], []
    for r in plan["records"]:
        name = r["name"]
        with np.load(root / r["input"]) as z:
            source, truth, oracle = z["source"], z["truth"], z["oracle"]
        with np.load(r["output"]) as z:
            previous = z[r["saved_key"]]
            if not r["new_head"]:
                head = z[r["head_key"]]
        if r["new_head"]:
            pred, head, _ = infer(model, source, 3, "whole", 0.5, 1.0)
            np.testing.assert_allclose(pred, previous, atol=1e-5, rtol=1e-5)
        raw, _ = draw(head)
        np.testing.assert_allclose(raw, previous, atol=1e-5, rtol=1e-5)
        if not r["new_head"]:
            np.testing.assert_array_equal(raw, previous)
        images, stats = {}, {}
        for mode in MODES:
            images[mode], stats[mode] = draw(replace(head, truth, mode))
        np.testing.assert_array_equal(images["full oracle"], oracle)
        np.testing.assert_array_equal(replace(head, truth, "raw"), head)
        path = root / "outputs" / (name + ".npz")
        np.savez_compressed(path, head=head, **images)
        tiles = [("source", source)] + list(images.items()) + [("direct geometry", zref(root, r))]
        n = source.shape[0]
        canvas = Image.new("RGB", (3 * n, 4 * (n + 28)), "#ddd")
        pen = ImageDraw.Draw(canvas)
        for i, (label, image) in enumerate(tiles):
            x, y = i % 3 * n, i // 3 * (n + 28)
            pen.text((x + 3, y + 4), name + " / " + label, fill="black")
            pix = np.clip(np.rint((image[..., 0] + 1) * 127.5), 0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(pix).convert("RGB"), (x, y + 28))
        p = root / "native" / (name + ".png")
        canvas.save(p)
        manifest.append(dict(file=p.name, sha256=sha(p), native_pixels=True))
        tc, hc = crop(truth), crop(head)
        valid, occupied = tc[..., 0] > 0.5, hc[..., 0] >= 0
        rows.append(
            dict(
                name=name,
                output=path.name,
                sha256=sha(path),
                graph=stats,
                raw_replay_max_abs=float(abs(raw - previous).max()),
                raw_replay_bit_exact=bool(np.array_equal(raw, previous)),
                vertex_false_positive=int((occupied & ~valid).sum()),
                vertex_false_negative=int((valid & ~occupied).sum()),
                oracle_exact=True,
                gt_vertices=int(valid.sum()),
                width_quantiles=np.quantile(
                    np.exp(np.clip(hc[..., 3][occupied], -8, np.log(8))), [0, 0.5, 0.9, 1]
                ).tolist()
                if occupied.any()
                else [],
                reference_teacher_max_abs=float(abs(oracle - zref(root, r)).max()),
            )
        )
        print(name, "done", flush=True)
    assert digest(nnx.state(model, nnx.Param)) == before
    write(root / "native/manifest.json", manifest)
    write(
        root / "evaluation.json",
        dict(
            records=rows,
            new_predictions=4,
            reused_predictions=8,
            renders=120,
            unchanged_states=True,
            training_updates=0,
        ),
    )


def zref(root, r):
    with np.load(root / r["input"]) as z:
        return z["reference"]


def finalize(root):
    plan = check(root)
    notes = read(root / "native/observations.json")
    manifest = read(root / "native/manifest.json")
    assert (
        len(manifest) == 12 and set(notes) == {r["file"] for r in manifest} and all(notes.values())
    )
    assert read(root / "evaluation.json")["unchanged_states"]
    assert not read(root / "decision.json")["adoption"]
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "completion.json"]
    write(
        root / "completion.json",
        dict(
            stage="complete_graph_residual_audit",
            commit=plan["commit"],
            new_predictions=4,
            reused_predictions=8,
            renders=120,
            native_pages_reviewed=12,
            training_updates=0,
            adoption=False,
            fresh_used=False,
            evidence={str(p.relative_to(root)): sha(p) for p in sorted(files)},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    a = parser.parse_args()
    start = time.monotonic()
    begin = datetime.datetime.now(datetime.UTC).isoformat()
    globals()[a.phase](a.root)
    print(
        json.dumps(
            dict(
                phase=a.phase,
                started_at=begin,
                finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
                seconds=time.monotonic() - start,
            )
        ),
        flush=True,
    )
