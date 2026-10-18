"""HW: bound projected connections using target arc-length geometry."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from attraction_graph import decode, logits_from_target, radius
from campaign_protocol import read, sha, write
from PIL import Image, ImageDraw
from projected_edge_gate import gate

HU = Path("../jax-learned-attraction-20261018/runs/learned-attraction-20261018-retry1").resolve()
CASES = (
    "probe-12",
    "probe-24",
    "probe-36",
    "probe-37",
    "probe-59",
    "scene-1",
    "scene-2",
    "scene-7",
)
MODES = {
    "raw": [],
    "gate": [],
    "disks": [],
    "oracle-geometry": [0, 1, 2, 3, 8],
    "oracle-geometry+gate": [0, 1, 2, 3, 8],
    "full-oracle": list(range(9)),
}


def crop(a):
    return a[32:-32, 32:-32]


def run(root):
    assert not root.exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    for f, h in read(HU / "completion.json")["evidence"].items():
        assert sha(HU / f) == h
    root.mkdir(parents=True)
    code = [
        Path(__file__),
        Path("scripts/attraction_graph.py"),
        Path("scripts/projected_edge_gate.py"),
        Path("tests/test_projected_edge_gate.py"),
        Path("docs/projected-edge-gate-plan-20261018.md"),
    ]
    inputs = [HU / "completion.json", HU / "plan.json"] + [
        HU / folder / f"{name}.npz" for folder in ("inputs", "outputs") for name in CASES
    ]
    inputs += [HU / r["file"] for r in read(HU / "plan.json")["records"]]
    write(
        root / "plan.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            modes=MODES,
            cases=CASES,
            training_updates=0,
            model_predictions=0,
            evidence={str(p.resolve()): sha(p) for p in code + inputs},
        ),
    )
    checks = []
    for record in read(HU / "plan.json")["records"]:
        with np.load(HU / record["file"]) as z:
            for level in range(6):
                h = logits_from_target(z[f"target{level}"], level)
                g, rejected = gate(h, level)
                np.testing.assert_array_equal(g[..., 4:8] >= 0, h[..., 4:8] >= 0)
                assert rejected == 0
                checks.append(dict(name=record["name"], level=level, edges_preserved=True))
    write(root / "teacher-gate-audit.json", checks)
    rows, manifest = [], []
    for name in CASES:
        with np.load(HU / "inputs" / f"{name}.npz") as z:
            t = z["target0"]
            oracle = z["oracle"]
        truth = logits_from_target(t)
        n = oracle.shape[0]
        with np.load(HU / "outputs" / f"{name}.npz") as z:
            for arm in ("control", "sparse"):
                head = z[f"{arm}-head-d3"]
                images, stats = {}, {}
                for mode, channels in MODES.items():
                    h = head.copy()
                    h[..., channels] = truth[..., channels]
                    rejected = 0
                    if mode.endswith("gate"):
                        h, rejected = gate(h)
                    elif mode == "disks":
                        h[..., 4:8] = -20
                    image, st = decode(h)
                    st["rejected_edges"] = rejected
                    images[mode] = crop(image)
                    stats[mode] = st
                np.testing.assert_array_equal(images["raw"], z[f"{arm}-d3"])
                np.testing.assert_array_equal(images["full-oracle"], oracle)
                p = root / f"{name}-{arm}.npz"
                np.savez_compressed(p, **images)
                canvas = Image.new("RGB", (n * 3, (n + 28) * 2), "#ddd")
                draw = ImageDraw.Draw(canvas)
                for k, (mode, image) in enumerate(images.items()):
                    x, y = (k % 3) * n, (k // 3) * (n + 28)
                    draw.text((x + 3, y + 4), name + " " + arm + " " + mode, fill="black")
                    a = np.clip(np.rint((image[..., 0] + 1) * 127.5), 0, 255).astype(np.uint8)
                    canvas.paste(Image.fromarray(a).convert("RGB"), (x, y + 28))
                pic = root / f"{name}-{arm}.png"
                canvas.save(pic)
                manifest.append(dict(file=pic.name, sha256=sha(pic), native_pixels=True))
                tc, hc = crop(t), crop(head)
                gt, pred = tc[..., 0] > 0.5, hc[..., 0] >= 0
                position = np.linalg.norm(
                    radius(0) * np.tanh(hc[..., 1:3]) - tc[..., 1:3], axis=-1
                )[gt]
                width = np.abs(np.exp(np.clip(hc[..., 3], -8, np.log(8))) - np.exp(tc[..., 3]))[gt]
                rows.append(
                    dict(
                        name=name,
                        arm=arm,
                        graph=stats,
                        output=p.name,
                        sha256=sha(p),
                        raw_exact=True,
                        oracle_exact=True,
                        position_error_quantiles=np.quantile(
                            position, [0, 0.5, 0.9, 0.99, 1]
                        ).tolist(),
                        width_error_quantiles=np.quantile(width, [0, 0.5, 0.9, 0.99, 1]).tolist(),
                        false_positive=int((pred & ~gt).sum()),
                        false_negative=int((~pred & gt).sum()),
                        gt_anchors=int(gt.sum()),
                        ink={m: float(((1 - a) / 2).sum()) for m, a in images.items()},
                        oracle_ink=float(((1 - oracle) / 2).sum()),
                    )
                )
    write(root / "manifest.json", manifest)
    write(root / "metrics.json", rows)


def finalize(root):
    plan = read(root / "plan.json")
    for f, h in plan["evidence"].items():
        assert sha(Path(f)) == h
    manifest = read(root / "manifest.json")
    notes = read(root / "observations.json")
    assert (
        len(manifest) == 16 and set(notes) == {r["file"] for r in manifest} and all(notes.values())
    )
    files = [
        "teacher-gate-audit.json",
        "plan.json",
        "manifest.json",
        "metrics.json",
        "observations.json",
        "decision.json",
    ]
    for r in manifest:
        assert sha(root / r["file"]) == r["sha256"]
        files.append(r["file"])
    for r in read(root / "metrics.json"):
        assert r["raw_exact"] and r["oracle_exact"] and sha(root / r["output"]) == r["sha256"]
        files.append(r["output"])
    write(
        root / "completion.json",
        dict(
            stage="complete_projected_edge_gate_audit",
            commit=plan["commit"],
            native_pages_reviewed=16,
            training_updates=0,
            model_predictions=0,
            adoption=False,
            fresh_used=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["run", "finalize"])
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
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
