"""Evaluate allowed stroke families, source fidelity, protection, and native outputs."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import forward, prepare_inputs
from cue_preserving_data import MODES, probes_for, strength_map
from PIL import Image, ImageDraw
from plausible_data import (
    CENTER,
    CHOICES,
    COUNT,
    ENDPOINTS,
    OPACITY,
    SIZE,
    profile_samples,
    raster,
    shape_metrics,
)
from plausible_objective import geometry_parts
from plausible_study import EVAL_SEED, METHODS, STEPS, check
from scipy import ndimage

from lineartgen.evaluation import _connected, _labels_near
from lineartgen.models.serialization import load_inference

ALL_METHODS = ("initial",) + METHODS
GROUPS = ((0, 6, 12, 18), (24, 30, 36, 42), (48, 54, 56, 58))
KEYS = (
    "family_profile_error",
    "missing_profile_fraction",
    "geometry_violation",
    "off_family_ink_fraction",
    "connection_cut_rate",
    "connection_bridge_rate",
    "blank_max_opacity",
    "black_interior_mean",
)
RULES = {
    "family_profile_error": 0.25,
    "missing_profile_fraction": 0.0,
    "geometry_violation": 0.2,
    "off_family_ink_fraction": 0.02,
    "connection_cut_rate": 0.0,
    "connection_bridge_rate": 0.0,
    "blank_max_opacity": 0.05,
}


def family_topology(prediction, data):
    observed = profile_samples(prediction, data["probes"])
    offsets = np.linspace(-1, 1, 129)[None] * data["probes"][:, 4:5]
    mass = np.trapezoid(data["profiles"], offsets[None], axis=-1)
    errors = np.trapezoid(abs(data["profiles"] - observed[None]), offsets[None], axis=-1)
    variant = int(np.argmin((errors / np.maximum(mass, 0.25)).mean(-1)))
    queries = data["connections"].copy()
    offset, _ = data.get("choices", CHOICES)[variant]
    queries[:, :2] += offset * data["direction"]
    queries[:, 2:4] += offset * data["direction"]
    cuts, bridges, cut_total, bridge_total, valid = 0, 0, 0, 0, 0
    for opacity in (0.1, 0.25):
        lp = ndimage.label((1 - prediction[..., 0]) / 2 > opacity, np.ones((3, 3)))[0]
        lt = ndimage.label((1 - data["variants"][variant, ..., 0]) / 2 > opacity, np.ones((3, 3)))[
            0
        ]
        for query in queries:
            present = bool(_labels_near(lt, query[:2]) and _labels_near(lt, query[2:4]))
            expected = bool(query[4])
            if not present or _connected(lt, query) != expected:
                continue
            valid += 1
            connected = _connected(lp, query)
            if expected:
                cuts += not connected
                cut_total += 1
            else:
                bridges += connected
                bridge_total += 1
    return dict(
        connection_cut_rate=cuts / cut_total if cut_total else None,
        connection_bridge_rate=bridges / bridge_total if bridge_total else None,
        connection_queries_valid=valid,
        connection_queries_total=2 * len(queries),
    )


@jax.jit
def geometry_metric(prediction, source, strength):
    return geometry_parts(prediction[None], source[None], strength[None], 0)["geometry"]


def metrics(prediction, data, strength, kind, mode):
    active = strength[..., 0] > 0
    assert np.isfinite(prediction).all()
    error = np.abs(prediction - data["source"])[~active]
    result = shape_metrics(prediction, data, strength)
    result["protected_max_error"] = float(error.max()) if error.size else None
    assert result["protected_max_error"] in (None, 0.0)
    result["geometry_violation"] = float(
        geometry_metric(jnp.asarray(prediction), jnp.asarray(data["source"]), jnp.asarray(strength))
    )
    if mode != "partial05" and len(data["probes"]):
        result |= family_topology(prediction, data)
    alpha = (1 - prediction[..., 0]) / 2
    if kind == "blank":
        result["blank_max_opacity"] = float(alpha.max())
    if kind == "black":
        core = (
            ndimage.distance_transform_edt(data["variants"][CENTER, ..., 0] < -0.98) > 8
        ) & active
        result["black_interior_mean"] = float(alpha[core].mean())
    # A designated teacher error is diagnostic only, never a criterion below.
    result["designated_teacher_mse"] = float(
        ((prediction - data["variants"][CENTER]) ** 2)[active].mean()
    )
    assert all(v is None or np.isfinite(v) for v in result.values())
    return result


def specification(root):
    points = np.column_stack((np.linspace(40, 216, 129), np.full(129, 128)))
    meta = dict(
        kinds=["straight"], path_geometry=[dict(points=(points - 32).tolist(), widths=[1.6] * 129)]
    )
    direction = np.array([0.0, 1.0])
    variants = np.stack([raster(meta, direction, o, w) for o, w in CHOICES])
    mixture = variants[list(ENDPOINTS)].mean(0)
    source = 1 - 2 * OPACITY * ndimage.gaussian_filter(
        (1 - mixture) / 2, (1.2, 1.2, 0), mode="constant"
    )
    probes, connections = probes_for(meta, margin=32)
    probes[:, 4] += 4
    data = dict(
        variants=variants,
        source=source,
        probes=probes,
        connections=connections,
        direction=direction,
        profiles=np.stack([profile_samples(v, probes) for v in variants]),
    )
    candidates = {
        "allowed_left": variants[ENDPOINTS[0]],
        "allowed_center": variants[CENTER],
        "allowed_right": variants[ENDPOINTS[1]],
        "allowed_new": raster(meta, direction, 0.75, 0.9),
        "gray_average": mixture,
        "blank": np.ones_like(source),
        "unrelated_clean": raster(meta, direction, 22, 1),
        "cut": variants[CENTER].copy(),
    }
    candidates["cut"][118:138, 120:136] = 1
    t = np.full_like(source, 0.5)
    rows = {name: metrics(v, data, t, "straight", "whole05") for name, v in candidates.items()}
    valid = [r["family_profile_error"] for n, r in rows.items() if n.startswith("allowed")]
    assert max(valid) < RULES["family_profile_error"], rows
    assert rows["gray_average"]["family_profile_error"] > max(valid) + 0.2, rows
    assert rows["blank"]["geometry_violation"] > RULES["geometry_violation"]
    assert rows["unrelated_clean"]["geometry_violation"] > RULES["geometry_violation"]
    assert rows["cut"]["connection_cut_rate"] > 0
    for n, r in rows.items():
        if n.startswith("allowed"):
            assert r["geometry_violation"] <= RULES["geometry_violation"]
            assert r["connection_cut_rate"] == 0
    canvas = Image.new("RGB", (4 * SIZE, 2 * (SIZE + 30)), "white")
    pen = ImageDraw.Draw(canvas)
    for i, (name, v) in enumerate(candidates.items()):
        x = i % 4 * SIZE
        y = i // 4 * (SIZE + 30)
        pen.text((x + 5, y + 5), name, fill="black")
        canvas.paste(
            Image.fromarray(np.rint((v[..., 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)).convert(
                "RGB"
            ),
            (x, y + 30),
        )
    canvas.save(root / "metric-specification.png")
    write(
        root / "metric-specification.json",
        dict(
            passed=True,
            rows=rows,
            rule="Simple straight-stroke specification only, not a universal lineart metric",
            files={"metric-specification.png": sha(root / "metric-specification.png")},
        ),
    )


def evaluate(root):
    check(root)
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    paths = {m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS} | dict(
        initial=root / "initial"
    )
    models = {m: load_inference(paths[m])[0] for m in ALL_METHODS}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    write(
        out / "checkpoint-binding.json",
        {
            str(p / n): sha(p / n)
            for p in paths.values()
            for n in ("weights.npz", "metadata.json", "config.json")
        },
    )
    rows, refs, bindings = [], [], {}
    for record in read(root / "data.json")["records"]:
        split, index, kind = record["split"], record["index"], record["kind"]
        with np.load(root / record["file"]) as z:
            data = {k: z[k] for k in z.files}
        for mode_index, mode in enumerate(MODES):
            t = strength_map(mode)
            references = dict(
                source=data["source"],
                gray_average=data["mixture"],
                allowed_center=data["variants"][CENTER],
                allowed_left=data["variants"][ENDPOINTS[0]],
                allowed_right=data["variants"][ENDPOINTS[1]],
            )
            for name, value in references.items():
                pred = np.where(t > 0, value, data["source"])
                refs.append(
                    dict(
                        split=split,
                        index=index,
                        kind=kind,
                        mode=mode,
                        method=name,
                        **metrics(pred, data, t, kind, mode),
                    )
                )
            for draw in range(2):
                key = jax.random.PRNGKey(EVAL_SEED)
                for v in (int(split == "probe"), index, mode_index, draw):
                    key = jax.random.fold_in(key, v)
                p = prepare_inputs(
                    data["variants"][CENTER][None], data["source"][None], t[None], key, depth=3
                )
                saved = {}
                for name, model in models.items():
                    pred = np.asarray(forward(model, p)[0][0])[0]
                    rows.append(
                        dict(
                            split=split,
                            index=index,
                            kind=kind,
                            mode=mode,
                            draw=draw,
                            method=name,
                            **metrics(pred, data, t, kind, mode),
                        )
                    )
                    saved[name] = pred
                file = out / f"predictions/{split}-{index:03d}-{mode}-{draw}.npz"
                np.savez_compressed(file, **saved)
                bindings[str(file.relative_to(out))] = sha(file)
        if (index + 1) % 16 == 0:
            progress = dict(split=split, cases=index + 1, model_rows=len(rows))
            write(out / "progress.json", progress)
            print(progress, flush=True)
    assert len(rows) == 3072 and len(refs) == 1920
    assert {(r["split"], r["index"], r["mode"], r["draw"], r["method"]) for r in rows} == {
        (s, i, m, d, w)
        for s in ("train", "probe")
        for i in range(COUNT)
        for m in MODES
        for d in range(2)
        for w in ALL_METHODS
    }
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            references=refs,
            coverage_exact=True,
            protected_exact=True,
            source_sha256=sha(root / "source.json"),
            binding_sha256=sha(out / "checkpoint-binding.json"),
        ),
    )
    write(out / "prediction-binding.json", bindings)
    summarize(root)
    render(root)
    write(
        root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=STEPS)
    )


def summarize(root):
    out = root / "evaluation"
    report = read(out / "metrics.json")
    groups = defaultdict(list)
    checks = []
    for r in report["rows"] + report["references"]:
        for kind in (r["kind"], "ALL"):
            groups[(r["split"], r["method"], r["mode"], kind)].append(r)
        for key, limit in RULES.items():
            if r.get(key) is not None:
                checks.append(
                    {k: r[k] for k in ("split", "index", "method", "mode", "draw") if k in r}
                    | dict(metric=key, value=r[key], threshold=limit, passed=r[key] <= limit)
                )
        if "black_interior_mean" in r:
            checks.append(
                {k: r[k] for k in ("split", "index", "method", "mode", "draw") if k in r}
                | dict(
                    metric="black_interior_mean",
                    value=r["black_interior_mean"],
                    threshold=0.95,
                    passed=r["black_interior_mean"] >= 0.95,
                )
            )
    summaries = []
    for g, rs in sorted(groups.items()):
        means = {
            k: float(np.mean([r[k] for r in rs if r.get(k) is not None]))
            for k in KEYS + ("designated_teacher_mse",)
            if any(r.get(k) is not None for r in rs)
        }
        summaries.append(
            dict(
                zip(("split", "method", "mode", "kind"), g, strict=True), rows=len(rs), means=means
            )
        )
    write(
        out / "summary.json",
        dict(
            groups=summaries,
            checks=checks,
            adoption=False,
            warning="Finite allowable family is a toy diagnostic. "
            "Teacher MSE and critic scores are not criteria.",
        ),
    )


def render(root):
    out = root / "evaluation"
    dest = out / "native-review"
    dest.mkdir(exist_ok=True)
    records = []
    for split in ("train", "probe"):
        for mode in ("whole05", "partial05"):
            for group, indices in enumerate(GROUPS):
                canvas = Image.new("RGB", (6 * SIZE, 36 + len(indices) * (SIZE + 24)), "white")
                pen = ImageDraw.Draw(canvas)
                for col, label in enumerate(
                    ("source", "allowed A", "allowed B", "pixel", "tolerant", "prior")
                ):
                    pen.text((col * SIZE + 8, 8), label, fill="black")
                bindings = {}
                for row, index in enumerate(indices):
                    ip = root / f"data/{split}/{index:03d}.npz"
                    pp = out / f"predictions/{split}-{index:03d}-{mode}-0.npz"
                    with np.load(ip) as data, np.load(pp) as pred:
                        t = strength_map(mode)
                        a, b = [
                            np.where(t > 0, data["variants"][i], data["source"]) for i in ENDPOINTS
                        ]
                        values = (
                            data["source"],
                            a,
                            b,
                            pred["pixel"],
                            pred["tolerant"],
                            pred["prior"],
                        )
                        for col, v in enumerate(values):
                            gray = np.rint(np.clip((v[..., 0] + 1) / 2, 0, 1) * 255).astype(
                                np.uint8
                            )
                            canvas.paste(
                                Image.fromarray(gray).convert("RGB"),
                                (col * SIZE, 60 + row * (SIZE + 24)),
                            )
                    pen.text((8, 38 + row * (SIZE + 24)), f"{split} {index} / {mode}", fill="black")
                    bindings[str(ip)] = sha(ip)
                    bindings[str(pp)] = sha(pp)
                file = dest / f"{split}-{mode}-{group}.png"
                canvas.save(file)
                records.append(dict(file=file.name, sha256=sha(file), bindings=bindings))
    write(
        dest / "manifest.json",
        dict(records=records, native_size=SIZE, contrast_rescaling=False, thresholding=False),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("specification", "evaluate", "summarize", "render"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
