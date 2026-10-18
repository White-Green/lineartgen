"""HT: audit projected-anchor geometry before training any predictor."""

import argparse
import datetime
import subprocess
import time
from pathlib import Path

import numpy as np
from attraction_graph import decode, logits_from_target, target
from campaign_protocol import read, sha, write
from PIL import Image, ImageDraw

HQ = Path(
    "../jax-centerline-grid-audit-20261018/runs/centerline-grid-audit-20261018-retry1"
).resolve()
DOC = Path("docs/attraction-graph-audit-plan-20261018.md")
CODE = [
    Path(p)
    for p in [
        "scripts/attraction_graph.py",
        "scripts/attraction_graph_audit.py",
        "tests/test_attraction_graph.py",
    ]
]


def prepare(root):
    assert (
        not root.exists()
        and not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    old = read(HQ / "completion.json")
    for f, h in old["evidence"].items():
        p = Path(f)
        assert sha(p if p.is_absolute() else HQ / p) == h
    root.mkdir(parents=True)
    (root / "inputs").symlink_to(HQ / "inputs", target_is_directory=True)
    records = read(HQ / "plan.json")["records"]
    files = [HQ / "completion.json", HQ / "plan.json", HQ / "metrics.json"]
    files += [HQ / r["file"] for r in records]
    files += [HQ / "outputs" / f"{r['name']}.npz" for r in records]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            bindings={str(p): sha(p) for p in files},
            code={str(p): sha(p) for p in CODE},
            document_sha256=sha(DOC),
            archive_sha256=sha(root / "source.tar.gz"),
            native_pages=12,
            learned_predictor=False,
            fresh_used=False,
            adoption=False,
            halo=32,
        ),
    )


def check(root):
    p = read(root / "plan.json")
    for f, h in {**p["bindings"], **p["code"]}.items():
        assert sha(Path(f)) == h
    assert sha(DOC) == p["document_sha256"] and sha(root / "source.tar.gz") == p["archive_sha256"]
    return p


def run(root):
    plan = check(root)
    out = root / "outputs"
    out.mkdir(exist_ok=False)
    rows = []
    for r in plan["records"]:
        with np.load(root / r["file"]) as z:
            data = {k: z[k] for k in z.files}
        with np.load(HQ / "outputs" / f"{r['name']}.npz") as z:
            ref, old = z["reference"], z["cell1"]
        stages, stats = [], []
        for level in range(6):
            t, st = target(
                data["segments"] + 32,
                data["widths"],
                data["rectangles"] + 32,
                r["size"] + 64,
                level,
                data["path_ids"],
            )
            assert np.isfinite(t).all()
            stages.append(t)
            stats.append(st)
        head = logits_from_target(stages[0])
        image, graph = decode(head)
        image = image[32:-32, 32:-32]
        a, b, c = [(1 - v) / 2 for v in (image, ref, old)]
        denom = max(float(b.sum()), 1)
        path = out / f"{r['name']}.npz"
        np.savez_compressed(
            path,
            source=data.get("source", ref),
            reference=ref,
            old=old,
            projected=image,
            head=head,
            **{f"target{i}": t for i, t in enumerate(stages)},
        )
        row = dict(
            name=r["name"],
            output_sha256=sha(path),
            stages=stats,
            graph=graph,
            projected_coverage_error=float(abs(a - b).sum() / denom),
            old_coverage_error=float(abs(c - b).sum() / denom),
            projected_ink_error=float(abs(a.sum() - b.sum()) / denom),
        )
        rows.append(row)
        print(r["name"], row["projected_coverage_error"], row["old_coverage_error"], flush=True)
    write(root / "metrics.json", dict(rows=rows, learned_predictor=False))
    native = root / "native"
    native.mkdir()
    pages = []
    for group in ("probe", "scene", "fixture"):
        records = [r for r in plan["records"] if r["group"] == group]
        per = 2 if group == "scene" else 4
        size = 512 if group == "scene" else 256
        for i in range(0, len(records), per):
            chosen = records[i : i + per]
            canvas = Image.new("RGB", (4 * size, 28 + len(chosen) * (size + 22)), "white")
            draw = ImageDraw.Draw(canvas)
            for col, label in enumerate(
                [
                    "source (fixture: oracle)",
                    "original opaque paths",
                    "one vertex / cell",
                    "projected anchors / oracle",
                ]
            ):
                draw.text((col * size + 3, 7), label, fill="black")
            for row, r in enumerate(chosen):
                with np.load(out / f"{r['name']}.npz") as z:
                    for col, key in enumerate(("source", "reference", "old", "projected")):
                        y = 28 + row * (size + 22)
                        draw.text((col * size + 3, y + 2), r["name"], fill="black")
                        im = np.round(np.clip((z[key][..., 0] + 1) * 127.5, 0, 255)).astype(
                            np.uint8
                        )
                        canvas.paste(Image.fromarray(im).convert("RGB"), (col * size, y + 22))
            path = native / f"{group}-{i // per}.png"
            canvas.save(path)
            pages.append(dict(file=path.name, sha256=sha(path), names=[r["name"] for r in chosen]))
    assert len(pages) == 12
    write(native / "manifest.json", pages)


def finalize(root):
    p = check(root)
    rows = read(root / "metrics.json")["rows"]
    pages = read(root / "native/manifest.json")
    notes = read(root / "native/observations.json")
    assert len(rows) == 40 and len(pages) == 12 and set(notes) == {r["file"] for r in pages}
    assert all(notes.values()) and read(root / "tests.json")["passed"]
    files = [
        "plan.json",
        "metrics.json",
        "native/manifest.json",
        "native/observations.json",
        "tests.json",
        "decision.json",
    ]
    for row in rows:
        f = f"outputs/{row['name']}.npz"
        assert sha(root / f) == row["output_sha256"]
        files.append(f)
    for row in pages:
        f = f"native/{row['file']}"
        assert sha(root / f) == row["sha256"]
        files.append(f)
    write(
        root / "completion.json",
        dict(
            stage="complete_attraction_graph_representation_audit",
            commit=p["commit"],
            training_updates=0,
            native_pages_reviewed=12,
            learned_predictor=False,
            fresh_used=False,
            adoption=False,
            evidence={f: sha(root / f) for f in files},
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["prepare", "run", "finalize"])
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    before = time.monotonic()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    globals()[a.phase](a.root)
    print(dict(phase=a.phase, started_at=started, seconds=time.monotonic() - before), flush=True)


if __name__ == "__main__":
    main()
