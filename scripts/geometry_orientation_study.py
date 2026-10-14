"""Test orientation dependence of fixed geometry predictors on known development inputs."""

import argparse
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from PIL import Image, ImageDraw
from plausible_study import digest
from subpixel_field_model import infer, load_field
from wide_style_data import allstroke_candidates

EA = Path("../jax-d4-geometry-20261014/runs/d4-geometry-20261014").resolve()
PATHS = {m: EA / f"snapshots/step-12000/{m}/raw" for m in ("control", "d4")}
INDICES = (36, 37, 56, 58, 59, 61, 62, 63)
DOC = Path("docs/d4-orientation-recheck-plan-20261014.md")
SEED = 2026101427


def transform(x, turns, mirror, *, inverse=False):
    """Exact D4 action on NHWC arrays; inverse restores original coordinates."""
    assert x.ndim == 4
    if inverse:
        x = jnp.rot90(x, -turns, axes=(1, 2))
        return jnp.flip(x, axis=2) if mirror else x
    x = jnp.flip(x, axis=2) if mirror else x
    return jnp.rot90(x, turns, axes=(1, 2))


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    assert read(EA / "completion.json")["stage"] == "complete_d4_geometry"
    root.mkdir(parents=True, exist_ok=False)
    records = [
        r
        for r in read(EA / "data.json")["evaluation"]
        if r["split"] == "new_probe" and r["index"] in INDICES
    ]
    assert tuple(r["index"] for r in records) == INDICES
    bound = {}
    for r in records:
        for p in (
            EA / r["file"],
            EA / f"evaluation/predictions/new_probe-{r['index']:03d}-whole05-0.npz",
        ):
            bound[str(p)] = sha(p)
    for path in PATHS.values():
        bound.update({str(p): sha(p) for p in path.iterdir() if p.is_file()})
    for p in (EA / "completion.json", EA / "diagnostics/review-observations.json", DOC):
        bound[str(p.resolve())] = sha(p)
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    for name in tracked:
        if name.endswith((".py", ".toml")) or name == "uv.lock":
            p = Path(name).resolve()
            bound[str(p)] = sha(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), commit],
        check=True,
    )
    write(
        root / "plan.json",
        dict(
            commit=commit,
            records=records,
            models={k: str(v) for k, v in PATHS.items()},
            bindings=bound,
            source_archive_sha256=sha(root / "source.tar.gz"),
            seed=SEED,
            depth=3,
            strength=0.5,
            transforms=8,
            outputs=128,
            native_pages=16,
            identity_atol=2e-6,
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
    rows, manifest, bindings, identities = [], [], {}, []
    start = time.perf_counter()
    for r in plan["records"]:
        index = r["index"]
        with np.load(EA / r["file"]) as z:
            data = {k: z[k] for k in z.files}
        source = data["source"]
        clean = allstroke_candidates(data, r)[13]
        prepared = prepare_inputs(
            clean[None],
            source[None],
            np.full_like(source[None], 0.5),
            jax.random.fold_in(jax.random.PRNGKey(SEED), index),
            3,
        )
        arrays = {}
        for mirror in (False, True):
            for turns in range(4):
                p = jax.tree.map(
                    lambda x, turns=turns, mirror=mirror: transform(x, turns, mirror), prepared
                )
                for method, model in models.items():
                    pred = transform(infer(model, p), turns, mirror, inverse=True)
                    pred = np.asarray(pred)[0]
                    assert np.isfinite(pred).all() and pred.min() >= -1 and pred.max() <= 1
                    key = f"{method}_{int(mirror)}_{turns}"
                    arrays[key] = pred
                    base = arrays[f"{method}_0_0"]
                    delta = pred - base
                    rows.append(
                        dict(
                            index=index,
                            method=method,
                            mirror=mirror,
                            turns=turns,
                            equivariance_rms=float(np.sqrt(np.mean(delta**2))),
                            equivariance_max_abs=float(abs(delta).max()),
                        )
                    )
        with np.load(EA / f"evaluation/predictions/new_probe-{index:03d}-whole05-0.npz") as old:
            for method in models:
                pred = arrays[f"{method}_0_0"]
                np.testing.assert_allclose(pred, old[method], atol=plan["identity_atol"], rtol=0)
                identities.append(
                    dict(
                        index=index,
                        method=method,
                        max_abs=float(abs(pred - old[method]).max()),
                        exact=bool(np.array_equal(pred, old[method])),
                    )
                )
        path = out / f"probe-{index:02d}.npz"
        np.savez_compressed(path, **arrays)
        bindings[str(path)] = sha(path)
        for mirror in (False, True):
            canvas = Image.new("RGB", (6 * 256, 36 + 2 * 280), "white")
            draw = ImageDraw.Draw(canvas)
            for col, label in enumerate(
                ("original source", "one permitted geometry")
                + tuple(
                    f"inverse mapped {'mirror+' if mirror else ''}{90 * k} degrees"
                    for k in range(4)
                )
            ):
                draw.text((col * 256 + 4, 8), label, fill="black")
            for row, method in enumerate(models):
                yy = 36 + row * 280
                draw.text((4, yy), f"probe{index} {r['kind']} / {method}", fill="black")
                values = [source, clean] + [arrays[f"{method}_{int(mirror)}_{k}"] for k in range(4)]
                for col, value in enumerate(values):
                    pixels = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
                    canvas.paste(Image.fromarray(pixels).convert("RGB"), (col * 256, yy + 20))
            page = out / f"native-probe-{index:02d}-{'mirror' if mirror else 'direct'}.png"
            canvas.save(page)
            manifest.append(
                dict(
                    file=page.name,
                    sha256=sha(page),
                    prediction=str(path),
                    prediction_sha256=sha(path),
                )
            )
        print(dict(case=index), flush=True)
    assert len(rows) == 128 and len(manifest) == 16
    assert states == {name: digest(nnx.state(model, nnx.Param)) for name, model in models.items()}
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            outputs=128,
            identities=identities,
            states_unchanged=True,
            bindings=bindings,
            evaluation_seconds=time.perf_counter() - start,
        ),
    )
    write(out / "manifest.json", dict(records=manifest))


def finalize(root):
    check(root)
    out = root / "evaluation"
    metrics, review = read(out / "metrics.json"), read(out / "visual-review.json")
    assert metrics["outputs"] == 128 and metrics["states_unchanged"]
    assert review["reviewed"] and len(review["observations"]) == 16
    for p, h in metrics["bindings"].items():
        assert sha(Path(p)) == h
    for r in read(out / "manifest.json")["records"]:
        assert sha(out / r["file"]) == r["sha256"] == review["files"][r["file"]]
        assert r["file"] in review["observations"]
        assert sha(Path(r["prediction"])) == r["prediction_sha256"]
    write(
        root / "completion.json",
        dict(
            stage="complete_d4_orientation_recheck",
            outputs=128,
            native_pages_reviewed=16,
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
