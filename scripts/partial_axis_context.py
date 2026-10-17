"""Fixed-weight feasibility audit of global whole-only axis permission."""

import argparse
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ambiguous_shape_data import render
from axis_gradient_projection import native_page
from axis_replay_mix import GY
from campaign_protocol import read, sha, write
from consistent_axis_objective import candidate_mask
from cue_capacity import prepare_inputs
from cue_preserving_data import strength_map
from flax import nnx
from plausible_study import digest
from pooled_axis_model import load_field
from pooled_axis_study import GW
from scipy import ndimage
from subpixel_field_data import candidate_fields
from subpixel_field_model import coverage, field_forward

PATHS = dict(
    initial=GW / "initials/choice",
    winner=GW / "snapshots/step-6000/choice/raw",
    consistent=GY / "snapshots/step-6000/consistent/raw",
)
REGION = (200, 190, 216, 226)
DOC = Path("docs/partial-axis-context-plan-20261017.md")


def prepare(root):
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    recipe = dict(
        width=48.0,
        length=48.0,
        angle=0.0,
        center=[208.0, 208.0],
        sigma=1.2,
        opacity=0.4,
        filled=False,
    )
    arrays = render(recipe)
    np.savez_compressed(
        root / "inputs.npz",
        **arrays,
        whole=strength_map("whole05"),
        partial=strength_map("partial05"),
    )
    write(root / "recipe.json", recipe)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(root / "source.tar.gz"), source], check=True
    )
    files = [p for p in root.iterdir() if p.is_file()] + [DOC.resolve()]
    files += [p for d in PATHS.values() for p in d.iterdir() if p.is_file()]
    files += [
        Path(p).resolve()
        for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if p.endswith((".py", ".toml")) or p == "uv.lock"
    ]
    write(
        root / "plan.json",
        dict(source=source, region=REGION, bindings={str(p.resolve()): sha(p) for p in files}),
    )
    print(dict(prepared=True), flush=True)


def check(root):
    p = read(root / "plan.json")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == p["source"]
    for f, h in p["bindings"].items():
        assert sha(Path(f)) == h


@nnx.jit
def prediction(model, prepared):
    field = field_forward(model, prepared)[0]
    raw = coverage(field)
    return field, raw, jnp.where(prepared.noise_levels[0] > 0, raw, prepared.guides[0])


def evaluate(root):
    check(root)
    with np.load(root / "inputs.npz") as z:
        a = {k: z[k] for k in z.files}
    t = np.stack([a["whole"], a["partial"]])
    source = np.stack([a["source"]] * 2)
    clean = np.stack([a["clean"]] * 2)
    p = prepare_inputs(clean, source, t, jax.random.PRNGKey(2026101712), 1)
    normal = jnp.asarray(np.stack([a["direction"]] * 2))
    mask = np.asarray(candidate_mask(normal, jnp.asarray(t), "consistent"))
    assert (
        not mask[0, :9].any() and mask[0, 9:].all() and mask[1, :9].all() and not mask[1, 9:].any()
    )
    vertical = np.asarray(candidate_fields(jnp.asarray(a["base"][None]), normal[:1], 1)[0][0])
    horizontal = np.asarray(
        candidate_fields(
            jnp.asarray(a["axis_base"][None]), jnp.asarray(a["axis_direction"][None]), 1
        )[0][0]
    )
    x0, y0, x1, y1 = REGION
    region = (slice(y0 * 4, y1 * 4), slice(x0 * 4, x1 * 4))
    contradictions = []
    for i, h in enumerate(horizontal):
        for j, v in enumerate(vertical):
            hh, vv = h[region], v[region]
            contradictions.append(
                dict(
                    horizontal=i,
                    vertical=j,
                    opposed_sign_subpixels=int(
                        np.count_nonzero(
                            ((hh > 0.25) & (vv < -0.25)) | ((vv > 0.25) & (hh < -0.25))
                        )
                    ),
                )
            )
    assert all(r["opposed_sign_subpixels"] > 0 for r in contradictions)
    support = np.any(np.concatenate([vertical, horizontal], axis=0)[..., 0] < 4, axis=0)
    active = np.repeat(np.repeat(t[1, ..., 0] > 0, 4, 0), 4, 1)
    assert np.all(active[support])
    whole_teacher = a["axis_clean"]
    partial_teacher = np.where(t[1] > 0, a["clean"], a["source"])
    records = []

    def draw(name, rows, labels):
        f = root / name
        native_page(f, rows, labels)
        records.append(dict(file=name, sha256=sha(f)))

    draw(
        "native-teachers.png",
        [[a["source"], whole_teacher, partial_teacher, 1 - 2 * (t[1] > 0).astype(np.float32)]],
        [["same source", "whole: horizontal", "partial: vertical", "partial active mask"]],
    )
    results = {}
    crops = []
    labels = []
    bindings = {}
    for name, path in PATHS.items():
        m, _ = load_field(path)
        state = digest(nnx.state(m, nnx.Param))
        layers = (
            [m.stem]
            + [layer for block in m.blocks for layer in (block.first, block.second)]
            + [m.context_block.first, m.context_block.second]
        )
        radius = sum(layer.dilation for layer in layers)
        assert radius == 76
        differing = (t[0, ..., 0] != t[1, ..., 0]).astype(np.uint8)
        reachable = ndimage.maximum_filter(differing, size=2 * radius + 1, mode="constant", cval=0)
        assert not reachable[y0:y1, x0:x1].any()
        field, raw, pred = [np.asarray(v) for v in prediction(m, p)]
        field_equal = np.array_equal(field[0][region], field[1][region])
        raw_equal = np.array_equal(raw[0, y0:y1, x0:x1], raw[1, y0:y1, x0:x1])
        np.testing.assert_array_equal(pred[t == 0], source[t == 0])
        assert state == digest(nnx.state(m, nnx.Param))
        f = root / f"predictions-{name}.npz"
        np.savez_compressed(f, field=field, raw=raw, prediction=pred)
        bindings[str(f)] = sha(f)
        results[name] = dict(
            noise_level_radius=radius,
            mask_reach_exact=True,
            field_region_exact=field_equal,
            raw_region_exact=raw_equal,
            maximum_field_difference=float(np.max(np.abs(field[0][region] - field[1][region]))),
            protected_exact=True,
            state_unchanged=True,
        )
        draw(
            f"native-{name}.png",
            [[source[0], whole_teacher, pred[0]], [source[1], partial_teacher, pred[1]]],
            [
                [name + " whole source", "horizontal required", "whole output"],
                [name + " partial source", "vertical required", "partial output"],
            ],
        )
        crops.append(
            [
                whole_teacher[y0:y1, x0:x1],
                partial_teacher[y0:y1, x0:x1],
                raw[0, y0:y1, x0:x1],
                raw[1, y0:y1, x0:x1],
            ]
        )
        labels.append([name + " H", "V", "whole", "partial"])
    # Pad for legible labels; preserve every crop pixel without resizing.
    padded = []
    for row in crops:
        values = []
        for crop in row:
            value = np.ones((y1 - y0, 96, 1), np.float32)
            value[:, 40 : 40 + x1 - x0] = crop
            values.append(value)
        padded.append(values)
    draw("native-conflict-crops.png", padded, labels)
    write(root / "manifest.json", dict(records=records, native_scale=True))
    write(
        root / "metrics.json",
        dict(
            models=results,
            outputs=6,
            region=REGION,
            contradictions=contradictions,
            all81_pairs_conflicting=True,
            all_candidates_support_fully_active=True,
            bindings=bindings,
            interpretation=(
                "Constructed depth1 incompatibility of consistent whole-only axis labels; "
                "not a claim that this explains all observed failures."
            ),
        ),
    )
    print(results, flush=True)


def finalize(root):
    check(root)
    m = read(root / "metrics.json")
    assert m["outputs"] == 6
    assert all(
        r["field_region_exact"]
        and r["raw_region_exact"]
        and r["state_unchanged"]
        and r["protected_exact"]
        for r in m["models"].values()
    )
    for p, h in m["bindings"].items():
        assert sha(Path(p)) == h
    records = read(root / "manifest.json")["records"]
    notes = read(root / "observations.json")
    assert len(records) == len(notes) == 5
    for r in records:
        assert notes[r["file"]] and sha(root / r["file"]) == r["sha256"]
    files = [p for p in root.iterdir() if p.is_file() and p.name != "completion.json"]
    write(
        root / "completion.json",
        dict(
            stage="complete_partial_axis_context_audit",
            training_updates=0,
            outputs=6,
            native_pages_reviewed=5,
            adoption=False,
            fresh_confirmation=False,
            evidence={str(p.relative_to(root)): sha(p) for p in files},
        ),
    )
    print(dict(complete=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "evaluate", "finalize"))
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
