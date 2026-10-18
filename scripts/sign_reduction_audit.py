"""HK: inspect loss sensitivity at frozen HJ outputs; no model or field training."""

import argparse
import copy
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from cue_capacity import prepare_inputs
from flax import nnx
from learned_coarse_axis_model import infer, load_field
from opaque_geometry import geometry
from PIL import Image, ImageDraw
from plausible_study import digest
from procedural_shape_data import render as rectangle
from sign_audit_objective import field_parts
from subpixel_field_data import candidate_fields, geometry_field
from subpixel_field_model import coverage, field_forward
from support_merge_data import merge_allowed, merge_geometry

PARENT = Path("../jax-coarse-duration-20261018/runs/coarse-duration-20261018").resolve()
GK = Path("../jax-bar-fill-transfer-20261017/runs/bar-fill-transfer-20261017").resolve()
GO = Path(
    "../jax-wide-context-identifiability-20261017/runs/wide-context-identifiability-20261017"
).resolve()
FU = Path("../jax-local-ambiguity-20261016/runs/local-ambiguity-20261016").resolve()
MODEL = PARENT / "snapshots/step-18000/consistent/raw"
CASES = (
    "probe37-whole",
    "probe37-partial",
    "oldtrain36-whole",
    "scene2-whole",
    "scene7-whole",
    "scene7-partial",
    "bar80-whole",
    "bar87-whole",
    "bar62-whole",
    "fill-whole",
)
DOC = Path("docs/sign-reduction-audit-plan-20261018.md")


def region_gradient(v, masks, chosen, *, epsilon=0.01, tiles=1):
    """Derivative of a region L4 mean, retaining common candidate denominators."""
    # v is H,W for one selected candidate; masks is K,H,W.
    k, h, w = masks.shape
    shape = (tiles, h // tiles, tiles, w // tiles)
    counts = masks.reshape(k, *shape).sum((2, 4)).max(0)
    valid = counts > 0
    denominator = np.maximum(counts, 1)
    mask = masks[chosen].astype(np.float64)
    moment = (v**4 * mask).reshape(shape).sum((1, 3)) / denominator
    factor = 1 / (denominator * (moment + epsilon**4) ** 0.75)
    factor *= valid / max(valid.sum(), 1)
    expanded = np.repeat(np.repeat(factor, h // tiles, 0), w // tiles, 1)
    gradient = v**3 * mask * expanded
    return gradient, moment, counts


def inputs(case):
    name, mode = case.split("-")
    metadata = None
    merged = False
    if name.startswith("scene"):
        index = int(name[5:])
        record = next(
            r for r in read(PARENT / "scene512/plan.json")["records"] if r["index"] == index
        )
        ip = PARENT / f"scene512/input-{index}.npz"
        with np.load(ip) as z:
            source, clean = z["quantized"], z["clean"]
        pp = PARENT / f"scene512/predictions-{index}-{mode}-0.npz"
        predkey, predindex = "consistent_uint8", None
        metadata, direction, depth = record["metadata"], [1.0, 0.0], 6
    elif name in ("probe37", "oldtrain36"):
        split, index = ("new_probe", 37) if name == "probe37" else ("old_train", 36)
        record = next(
            r
            for r in read(PARENT / "data.json")["evaluation"]
            if r["split"] == split and r["index"] == index
        )
        ip = PARENT / record["file"]
        with np.load(ip) as z:
            source, clean = z["source"], z["variants"][13]
        pp = PARENT / f"evaluation/predictions/{split}-{index:03d}-{mode}05-0.npz"
        predkey, predindex = "consistent", None
        metadata = copy.deepcopy(record["metadata"])
        merged = merge_allowed(source, merge_geometry(record))
        if merged:
            a, b = metadata["path_geometry"]
            points = (np.asarray(a["points"]) + np.asarray(b["points"])) / 2
            metadata["path_geometry"] = [
                dict(kind="bezier", points=points.tolist(), widths=[2.0] * len(points))
            ]
            metadata["paths"] = 1
        direction, depth = record["direction"], 3
    else:
        depth, predkey = 3, "consistent"
        if name == "bar80":
            ip = GK / "inputs.npz"
            with np.load(ip) as z:
                source, clean = z["source"][2, 0], z["target"][2, 0]
            pp, predindex = PARENT / "diagnostics/geometry/predictions-3-2.npz", 0
            recipe = dict(width=80.0, length=160.0, center=[128.0, 128.0], filled=False)
        elif name == "bar62":
            ip = FU / "inputs.npz"
            with np.load(ip) as z:
                source, clean = z["source"][0], z["clean"][0]
            pp, predindex = PARENT / "diagnostics/ambiguity/depth-3.npz", 0
            # FU62 is shifted to x63 rather than x128.
            recipe = dict(width=62.0, length=160.0, center=[63.0, 128.0], filled=False)
        else:
            ip = GO / "inputs.npz"
            predindex = 0 if name == "bar87" else 1
            with np.load(ip) as z:
                source, clean = z["source"][predindex], z["clean"][predindex]
            pp = PARENT / "diagnostics/collision/predictions-3.npz"
            recipe = (
                dict(width=87.5, length=200.0, center=[127.75, 128.0], filled=False)
                if name == "bar87"
                else dict(width=160.0, length=200.0, center=[164.0, 128.0], filled=True)
            )
        arrays = rectangle(recipe | dict(angle=0.0, sigma=1.2, opacity=0.4))
        np.testing.assert_array_equal(arrays["clean"], clean)
        base, direction = jnp.asarray(arrays["base"])[None], arrays["direction"]
    with np.load(pp) as z:
        expected = z[predkey] if predindex is None else z[predkey][predindex]
    size = source.shape[0]
    if metadata is not None:
        args = tuple(jnp.asarray(v) for v in geometry(metadata, size=size))
        base = geometry_field(*args, size=size)[None].astype(jnp.float16)
    yy, xx = np.indices((size, size))
    strength = (
        np.full_like(source, 0.5)
        if mode == "whole"
        else np.where((xx > size * 0.6 - yy * 0.25)[..., None], 0.5, 0).astype(np.float32)
    )
    prepared = prepare_inputs(
        clean[None], source[None], strength[None], jax.random.PRNGKey(2026101801), depth
    )
    targets = candidate_fields(base, jnp.asarray(direction, dtype=jnp.float32)[None], depth)
    return (
        prepared,
        targets,
        source,
        strength,
        expected,
        dict(
            input_file=str(ip),
            prediction_file=str(pp),
            merge_selected=bool(merged),
            depth=depth,
            teacher="source-supported merged path" if merged else "opaque original geometry",
        ),
    )


@nnx.jit
def fields_for(model, prepared):
    return field_forward(model, prepared)


@jax.jit
def parts_and_gradients(fields, targets, strengths):
    parts = field_parts(fields, targets, strengths)

    def objective(ff):
        p = field_parts(ff, targets, strengths)
        return jnp.stack((p["distance"], 0.1 * p["geometry_gradient"], 0.25 * p["sign_margin"]))

    gradients = jax.jacrev(objective)(fields)
    return parts, gradients


def describe(fields, targets, strengths, choice, grads):
    rows, arrays = [], {}
    norm = sum(0.5**i for i in range(len(fields)))
    for level, (ff, tt, strength, gg) in enumerate(
        zip(fields, targets, strengths, grads, strict=True)
    ):
        f = np.asarray(ff)[0, ..., 0].astype(np.float64)
        t = np.clip(np.asarray(tt)[0, ..., 0], -8, 8).astype(np.float64)
        active = np.repeat(np.repeat(np.asarray(strength)[0, ..., 0] > 0, 4, 0), 4, 1)
        near = (abs(t) < 4).any(0)
        margin = np.minimum(abs(t[choice]), 0.25)
        all_global = np.zeros_like(f)
        all_low = np.zeros_like(f)
        all_local = np.zeros_like(f)
        for region_name, region in (("near", near & active), ("far", ~near & active)):
            for side, masks, violation, sign in (
                ("inside", (t < 0) * region, np.maximum(margin + f, 0), 1.0),
                ("outside", (t > 0) * region, np.maximum(margin - f, 0), -1.0),
            ):
                g, moment, counts = region_gradient(violation, masks, choice)
                low, _, _ = region_gradient(violation, masks, choice, epsilon=0.0001)
                local, _, _ = region_gradient(violation, masks, choice, tiles=4)
                weight = sign * 0.125 * 0.5**level / norm
                all_global += weight * g
                all_low += weight * low
                all_local += weight * local
                selected = masks[choice] & (violation > 0)
                fourth = float(moment[0, 0])
                rows.append(
                    dict(
                        level=level,
                        region=region_name,
                        side=side,
                        count=int(counts[0, 0]),
                        violations=int(selected.sum()),
                        fourth_moment=fourth,
                        l4_unregularized=fourth**0.25,
                        epsilon01_to0001_gradient_ratio=((fourth + 0.01**4) / (fourth + 0.0001**4))
                        ** 0.75,
                        violation_max=float(violation[selected].max(initial=0)),
                    )
                )
        analytic_error = float(abs(all_global - np.asarray(gg)[2, 0, ..., 0]).max())
        np.testing.assert_allclose(all_global, np.asarray(gg)[2, 0, ..., 0], atol=2e-7, rtol=2e-4)
        if level == 0:
            alpha = np.clip(0.5 - 4 * f, 0, 1)
            # Outside every allowed shift by >1px excludes legitimate shifted edges.
            exterior = (t.min(0) > 1) & active & (alpha > 0.01)
            faint = exterior & (alpha < 0.99)
            inside = (t[choice] < -0.25) & active & (alpha < 0.99)
            total = np.asarray(gg)[:, 0, ..., 0].sum(0)
            for name, mask in (
                ("exterior", exterior),
                ("faint_exterior", faint),
                ("weak_interior", inside),
            ):
                direction = 1 if name == "weak_interior" else -1

                def energy(a, mask=mask):
                    return float(abs(a[mask]).sum())

                rows.append(
                    dict(
                        level=0,
                        selection=name,
                        subpixels=int(mask.sum()),
                        current_sign_gradient_l1=energy(all_global),
                        epsilon0001_gradient_l1=energy(all_low),
                        local4x4_gradient_l1=energy(all_local),
                        distance_gradient_l1=energy(np.asarray(gg)[0, 0, ..., 0]),
                        geometry_gradient_l1=energy(np.asarray(gg)[1, 0, ..., 0]),
                        total_correct_direction_fraction=float((total[mask] * direction > 0).mean())
                        if mask.any()
                        else None,
                    )
                )
            arrays = dict(
                field=f.astype(np.float32),
                selected_target=t[choice].astype(np.float32),
                sign_gradient=all_global.astype(np.float32),
                low_epsilon_gradient=all_low.astype(np.float32),
                local_gradient=all_local.astype(np.float32),
                component_gradients=np.asarray(gg)[:, 0, ..., 0],
                faint_exterior=faint,
                weak_interior=inside,
                exterior=exterior,
            )
        rows.append(dict(level=level, analytic_gradient_max_error=analytic_error))
    return rows, arrays


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    paths = [
        DOC,
        PARENT / "review-completion.json",
        PARENT / "supplemental-binding.json",
        PARENT / "data.json",
        PARENT / "scene512/plan.json",
    ]
    paths += [p for p in MODEL.iterdir() if p.is_file()]
    paths += [
        Path(p)
        for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if p.endswith((".py", ".toml")) or p == "uv.lock"
    ]
    for name in (
        "evaluation/predictions",
        "scene512",
        "diagnostics/ambiguity",
        "diagnostics/geometry",
        "diagnostics/collision",
    ):
        paths += [p for p in (PARENT / name).glob("*.npz")]
    for folder in ("data/new_probe", "data/expanded"):
        paths += [PARENT / folder / ("037.npz" if folder.endswith("new_probe") else "288.npz")]
    paths += [GK / "inputs.npz", GO / "inputs.npz", FU / "inputs.npz"]
    write(
        root / "plan.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            cases=CASES,
            cnn_training=False,
            field_training=False,
            bindings={str(p.resolve()): sha(p) for p in paths},
            interpretation=(
                "Diagnostic only; output-field gradients do not prove CNN update behavior."
            ),
            cross_inference_tolerance=dict(atol=1e-5, rtol=1e-5),
        ),
    )


def evaluate(root):
    plan = read(root / "plan.json")
    for p, h in plan["bindings"].items():
        assert sha(Path(p)) == h, p
    assert jax.devices()[0].platform != "cpu"
    model, _ = load_field(MODEL)
    initial_state = digest(nnx.state(model, nnx.Param))
    rows, manifest, bindings = [], [], {}
    for case in CASES:
        p, t, source, strength, expected, info = inputs(case)
        fields = fields_for(model, p)
        pred = np.asarray(infer(model, p))[0]
        split = np.where(strength > 0, np.asarray(coverage(fields[0]))[0], source)
        for actual in (pred, split):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(actual[strength == 0], source[strength == 0])
        info |= dict(
            case=case,
            infer_max_error=float(abs(pred - expected).max()),
            split_max_error=float(abs(split - expected).max()),
            infer_exact=bool(np.array_equal(pred, expected)),
        )
        parts, gradients = parts_and_gradients(fields, t, p.noise_levels)
        choice = int(np.asarray(parts["choice"])[0, 0])
        details, arrays = describe(fields, t, p.noise_levels, choice, gradients)
        info |= dict(
            choice=choice,
            parts={k: np.asarray(v).tolist() for k, v in parts.items()},
            regions=details,
        )
        rowfile = root / f"{case}.npz"
        oracle = np.where(strength > 0, np.asarray(coverage(t[0][:, choice]))[0], source)
        np.savez_compressed(
            rowfile, source=source, prediction=pred, oracle=oracle, strength=strength, **arrays
        )
        bindings[str(rowfile.resolve())] = sha(rowfile)
        size = source.shape[0]
        page = Image.new("RGB", (size * 3, size + 28), "white")
        pen = ImageDraw.Draw(page)
        for col, (label, value) in enumerate(
            (("source", source), ("selected opaque teacher", oracle), ("HJ frozen18k", pred))
        ):
            px = np.uint8(np.rint(np.clip((value[..., 0] + 1) * 127.5, 0, 255)))
            page.paste(Image.fromarray(px).convert("RGB"), (col * size, 28))
            pen.text((col * size + 4, 7), label, fill="black")
        file = root / f"{case}-native.png"
        page.save(file)
        manifest.append(dict(file=file.name, sha256=sha(file), native_scale=True))
        rows.append(info)
        write(root / "progress.json", dict(cases_complete=len(rows), case=case))
        print(
            dict(
                case=case,
                choice=choice,
                merge=info["merge_selected"],
                infer_error=info["infer_max_error"],
            ),
            flush=True,
        )
    assert digest(nnx.state(model, nnx.Param)) == initial_state
    write(
        root / "results.json",
        dict(rows=rows, states_unchanged=True, training=False, bindings=bindings),
    )
    write(root / "manifest.json", dict(records=manifest, native_scale=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.root.resolve())
