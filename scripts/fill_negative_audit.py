"""IF: fixed-weight final fill ablation and direct logit-gradient audit."""

import argparse
import datetime
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from attraction_graph import decode
from attraction_stress import key, prepared
from attraction_width_study import to_image
from campaign_protocol import read, sha, write
from fill_hard_negative import hard_negative_bce
from flax import nnx
from learned_centerline_model import balanced_bce, infer_heads, load_weights
from learned_centerline_study import crop, make_model
from PIL import Image, ImageDraw
from plausible_study import digest
from projected_edge_gate import gate

IE = Path("../jax-continuous-width-20261018/runs/continuous-width-20261018").resolve()
DOC = Path("docs/fill-negative-audit-plan-20261018.md")
TRAINING = (25, 512, 769, 896)
BROAD = (
    "extra-bar-center",
    "extra-bar-wide",
    "extra-wide87",
    "extra-fill160",
    "width-58-0.5",
    "width-61-0.25",
    "stress-41-rough",
    "stress-36-rough",
)
CODE = (
    "scripts/fill_negative_audit.py",
    "scripts/fill_hard_negative.py",
    "scripts/attraction_graph.py",
    "scripts/attraction_stress.py",
    "scripts/projected_edge_gate.py",
    "scripts/learned_centerline_model.py",
    "tests/test_fill_hard_negative.py",
)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def prepare(root):
    assert not root.exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    for filename in ("completion.json", "supplemental-binding.json"):
        for f, h in read(IE / filename)["evidence"].items():
            assert sha(IE / f) == h, f
    root.mkdir(parents=True)
    for filename in ("config.json",):
        shutil.copy2(IE / filename, root / filename)
    shutil.copy2(IE / "snapshots/step-3000/expanded/raw.npz", root / "initial.npz")
    records, bindings = [], {}
    training = read(IE / "training-data.json")["extra"]
    for i in TRAINING:
        r = training[i]
        p = Path(r["path"])
        assert sha(p) == r["sha256"]
        previous = IE / f"fit-probe/{i:04d}.npz"
        records.append(
            dict(
                name=f"train-{i:04d}",
                path=str(p),
                training_index=i,
                previous=str(previous),
                previous_key="expanded-d3",
            )
        )
    for name in BROAD:
        records.append(
            dict(
                name=name,
                path=str(IE / f"broad/inputs/{name}.npz"),
                previous=str(IE / f"broad/outputs/{name}.npz"),
                previous_key=key("expanded"),
            )
        )
    files = [
        IE / "completion.json",
        IE / "supplemental-binding.json",
        root / "initial.npz",
        root / "config.json",
    ]
    files += [Path(r[k]) for r in records for k in ("path", "previous")]
    for p in files:
        bindings[str(p.resolve())] = sha(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit], check=True
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            created_at=now(),
            records=records,
            bindings=bindings,
            code={f: sha(Path(f)) for f in CODE},
            document_sha256=sha(DOC),
            archive_sha256=sha(root / "source.tar.gz"),
            training_updates=0,
            fresh_used=False,
            adoption=False,
        ),
    )


def check(root):
    plan = read(root / "plan.json")
    for f, h in {**plan["bindings"], **plan["code"]}.items():
        assert sha(Path(f)) == h, f
    assert sha(DOC) == plan["document_sha256"]
    assert sha(root / "source.tar.gz") == plan["archive_sha256"]
    return plan


def gradient_audit(logit, truth):
    x, y = jnp.asarray(logit[None]), jnp.asarray(truth[None])
    masks = dict(
        false_positive=(truth == 0) & (logit >= 0),
        other_negative=(truth == 0) & (logit < 0),
        positive=truth == 1,
    )
    losses, arrays = {}, {}
    for name, fun in (("balanced", balanced_bce), ("hard1pct", hard_negative_bce)):
        value, g = jax.value_and_grad(fun)(x, y)
        g = np.asarray(g)[0]
        assert np.isfinite(value) and np.isfinite(g).all()
        losses[name] = dict(
            loss=float(value),
            gradient_l1=float(abs(g).sum()),
            regions={k: float(abs(g)[m].sum()) for k, m in masks.items()},
        )
        arrays[name] = g
    return dict(
        cells=truth.size,
        positives=int(truth.sum()),
        region_cells={k: int(m.sum()) for k, m in masks.items()},
        logit_quantiles=np.quantile(logit, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        objectives=losses,
    ), arrays


def run(root):
    plan = check(root)
    model, _ = make_model(root)
    load_weights(model, root / "initial.npz")
    before = digest(nnx.state(model, nnx.Param))
    out = root / "outputs"
    native = root / "native"
    out.mkdir()
    native.mkdir()
    records, manifest, outputs = [], [], {}
    for r in plan["records"]:
        with np.load(r["path"]) as z:
            source = z["source"]
            truth = [z[f"smalltarget{i}"] for i in range(6)] if "training_index" in r else None
        depths = (3, 6) if truth is not None else (3,)
        arrays, metrics = {}, []
        for depth in depths:
            p, mask = prepared(source, depth, "whole", 0.5)
            heads = [np.asarray(h)[0] for h in infer_heads(model, p)]
            assert all(np.isfinite(h).all() for h in heads)
            for level, h in enumerate(heads):
                arrays[f"d{depth}-head{level}"] = h
            h, rejected = gate(heads[0], base_radius=1.0)
            images, stats = {}, {}
            for variant in ("normal", "no_fill", "fill_only"):
                v = h.copy()
                if variant == "no_fill":
                    v[..., 8] = -20
                if variant == "fill_only":
                    v[..., 0] = -20
                image, st = decode(v, base_radius=1.0)
                image = np.where(mask > 0, crop(image), source)
                assert np.isfinite(image).all()
                arrays[f"d{depth}-{variant}"] = image
                images[variant], stats[variant] = image, st
            with np.load(r["previous"]) as z:
                previous = z[r["previous_key"] if depth == 3 else "expanded-d6"]
            np.testing.assert_allclose(images["normal"], previous, atol=1e-5, rtol=1e-5)
            row = dict(
                depth=depth,
                graph=stats,
                rejected_edges=rejected,
                previous_max_difference=float(abs(images["normal"] - previous).max()),
                fill_changed_pixels=int(np.count_nonzero(images["normal"] != images["no_fill"])),
            )
            if truth is not None:
                stages = []
                for level, h in enumerate(heads):
                    report, gradients = gradient_audit(h[..., 8], truth[level][..., 8])
                    report["stage_weight"] = 0.5**level / sum(0.5**k for k in range(depth))
                    stages.append(report)
                    arrays[f"d{depth}-truth{level}"] = truth[level][..., 8]
                    for k, a in gradients.items():
                        arrays[f"d{depth}-grad-{k}-{level}"] = a
                row["fill_logit_gradient"] = stages
            metrics.append(row)
            if depth == 3:
                n = source.shape[0]
                canvas = Image.new("RGB", (n * 4, n + 28), "#ddd")
                draw = ImageDraw.Draw(canvas)
                cols = [("source", source)] + list(images.items())
                for col, (label, image) in enumerate(cols):
                    draw.text((n * col + 3, 4), r["name"] + " " + label, fill="black")
                    canvas.paste(to_image(image), (n * col, 28))
                page = native / (r["name"] + ".png")
                canvas.save(page)
                manifest.append(dict(file=page.name, sha256=sha(page), native_pixels=True))
        path = out / (r["name"] + ".npz")
        np.savez_compressed(path, **arrays)
        outputs[path.name] = sha(path)
        records.append(dict(name=r["name"], records=metrics))
    assert before == digest(nnx.state(model, nnx.Param))
    write(native / "manifest.json", manifest)
    write(
        root / "result.json",
        dict(
            records=records,
            outputs=outputs,
            model_predictions=16,
            rendered_outputs=48,
            native_pages=12,
            unchanged_states=True,
            training_updates=0,
            fresh_used=False,
            finished_at=now(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    globals()[args.phase](args.root.resolve())
