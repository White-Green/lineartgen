"""Read-only defect ranking by the old critic, on new generated geometric patches.

These are diagnostic perturbations, not a held-out classifier benchmark or a
proposal to change the generator. Positive clean-minus-defect margin is desirable.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from campaign_protocol import read, sha, write
from flax import nnx
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from lineartgen.config import SketchAugmentationConfig, load_config
from lineartgen.evaluation import KINDS, line_case
from lineartgen.models.serialization import load_inference
from lineartgen.sketch import augment_sketch
from lineartgen.training import lineart_state
from lineartgen.training.lineart_study import create_arm, study_methods


def defects(target):
    ink = np.clip((1 - target[..., 0]) / 2, 0, 1)
    halo = np.maximum(ink, 0.5 * ndimage.gaussian_filter(ink, 1.5))
    variants = dict(clean=ink, halo=halo, weak=ink * 0.6)
    points = np.argwhere(ink > 0.5)
    if not len(points):
        raise ValueError("Expected a visible line")
    center = points[np.argmin(np.square(points - 32).sum(1))]
    yy, xx = np.indices(ink.shape)
    cut = ink.copy()
    cut[np.square(yy - center[0]) + np.square(xx - center[1]) <= 9**2] = 0
    variants["cut"] = cut
    labels, count = ndimage.label(ink > 0.5, np.ones((3, 3)))
    if count == 2:
        left, right = np.argwhere(labels == 1), np.argwhere(labels == 2)
        distances, indices = cKDTree(right).query(left)
        index = int(np.argmin(distances))
        if 1 < distances[index] <= 40:
            a, b = left[index], right[indices[index]]
            # A 3 px antialiased connector across a genuinely empty corridor.
            drawing = Image.new("L", (256, 256), 0)
            ImageDraw.Draw(drawing).line(
                [(float(a[1] * 4), float(a[0] * 4)), (float(b[1] * 4), float(b[0] * 4))],
                fill=255,
                width=12,
            )
            connector = np.asarray(drawing.resize((64, 64), Image.Resampling.LANCZOS)) / 255
            variants["bridge"] = np.maximum(ink, connector)
    return {name: (1 - 2 * value[..., None]).astype(np.float32) for name, value in variants.items()}


def run(parent, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = dict(
        geometry_kinds=list(KINDS),
        variants=list(range(24, 32)),
        strengths=[0.1, 0.5],
        sources=["clean", "strokes_strong"],
        seed=2026100710,
        step=10000,
        defects=dict(
            halo_sigma=1.5,
            halo_opacity_multiplier=0.5,
            weak_multiplier=0.6,
            cut_radius=9,
            bridge_width=3,
        ),
        purpose="Rank clean above artificial defects, with identical source/noise condition",
        limitations="New geometric draws; synthetic perturbations and conditions, not natural "
        "sketches. No critic or generator updates. No threshold/defect tuning after scoring.",
        source_sha256=sha(Path(__file__)),
    )
    write(output / "plan.json", plan)
    protocol = read(parent / "protocol.json")
    initial, _ = load_inference(protocol["initial"])
    arms = {
        name: create_arm(
            load_config(parent / "configs" / f"{name}.json"),
            initial,
            adversarial=name == "patch_prior",
            patch_size=64,
        )
        for name in study_methods(protocol)
    }
    untrained = nnx.clone(arms["patch_prior"]["critic"])
    checkpoint = parent / "milestone-checkpoints"
    manifest_path = checkpoint / "step-10000-sha256.json"
    archive = read(manifest_path)
    assert archive["step"] == 10000
    for path, digest in archive["files"].items():
        assert sha(checkpoint / "10000" / path) == digest, path
    checkpoint_hashes = {
        str(p.relative_to(checkpoint)): sha(p)
        for p in sorted((checkpoint / "10000").rglob("*"))
        if p.is_file()
    }
    with ocp.CheckpointManager(
        checkpoint.resolve(), options=ocp.CheckpointManagerOptions(read_only=True)
    ) as saved:
        assert lineart_state.restore(saved, arms, 10000) == 10000
    trained = arms["patch_prior"]["critic"]
    critics = {"untrained": untrained, "trained": trained}
    predict = nnx.jit(lambda model, images: model(images))
    sketch_config = SketchAugmentationConfig(
        mode="strokes",
        probability=1.0,
        opacity_min=0.2,
        opacity_max=0.45,
        blur_sigma_min=1.5,
        blur_sigma_max=2.5,
        dilation_radius=2,
        displacement=3.0,
    )
    augment = jax.jit(lambda x, key: augment_sketch(x, key, sketch_config))
    inputs, metadata, panels = [], [], []
    for kind in plan["geometry_kinds"]:
        for variant in plan["variants"]:
            gray, _, _ = line_case(kind, variant)
            target = np.clip((gray[224:288, 224:288] * 2 - 1) * 1.1, -1, 1)
            candidates = defects(target)
            if kind not in ("parallel", "gap"):
                candidates.pop("bridge", None)
            target_index = plan["geometry_kinds"].index(kind) * 8 + variant - 24
            rough = np.asarray(
                augment(target[None], jax.random.PRNGKey(plan["seed"] + target_index))
            )[0]
            if variant == 24:
                panels.append((kind, candidates))
            for source_name in plan["sources"]:
                source = target if source_name == "clean" else rough
                for strength in plan["strengths"]:
                    rng = np.random.default_rng([plan["seed"], target_index, int(strength * 100)])
                    noisy = source * (1 - strength) + rng.normal(size=source.shape) * strength
                    for defect, candidate in candidates.items():
                        inputs.append(
                            np.concatenate(
                                (
                                    np.clip(noisy, -3, 3) / 3,
                                    np.full_like(candidate, strength),
                                    candidate,
                                ),
                                axis=-1,
                            ).astype(np.float32)
                        )
                        metadata.append(
                            dict(
                                kind=kind,
                                variant=variant,
                                source=source_name,
                                strength=strength,
                                defect=defect,
                            )
                        )
    scores = {}
    for name, critic in critics.items():
        scores[name] = np.concatenate(
            [
                np.asarray(predict(critic, np.asarray(inputs[i : i + 64])))
                for i in range(0, len(inputs), 64)
            ]
        )
    rows = [
        row | {name: float(values[i]) for name, values in scores.items()}
        for i, row in enumerate(metadata)
    ]
    summary = []
    for critic in critics:
        clean = {
            (r["kind"], r["variant"], r["source"], r["strength"]): r[critic]
            for r in rows
            if r["defect"] == "clean"
        }
        groups = defaultdict(list)
        for row in rows:
            if row["defect"] == "clean":
                continue
            key = tuple(row[k] for k in ("kind", "variant", "source", "strength"))
            groups[row["source"], row["defect"]].append(clean[key] - row[critic])
        for (source, defect), margins in sorted(groups.items()):
            summary.append(
                dict(
                    critic=critic,
                    source=source,
                    defect=defect,
                    pairs=len(margins),
                    clean_preferred_fraction=float((np.asarray(margins) > 0).mean()),
                    mean_margin=float(np.mean(margins)),
                    median_margin=float(np.median(margins)),
                )
            )
    write(
        output / "results.json",
        dict(
            plan_sha256=sha(output / "plan.json"),
            checkpoint_files=checkpoint_hashes,
            parent_checkpoint_manifest_sha256=sha(manifest_path),
            rows=rows,
            summary=summary,
            generator_and_critic_modified=False,
        ),
    )
    canvas = Image.new("L", (5 * 128, len(panels) * 152), 255)
    for y, (kind, candidates) in enumerate(panels):
        for x, defect in enumerate(["clean", "halo", "weak", "cut", "bridge"]):
            if defect not in candidates:
                continue
            pixels = np.rint((candidates[defect][..., 0] + 1) * 127.5).astype(np.uint8)
            canvas.paste(
                Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST),
                (x * 128, y * 152 + 24),
            )
            ImageDraw.Draw(canvas).text((x * 128 + 3, y * 152 + 4), f"{kind}: {defect}", fill=0)
    canvas.save(output / "defects.png")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())
