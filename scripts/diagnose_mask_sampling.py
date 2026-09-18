"""Replay training crops to measure N2's mask area and source-boundary exposure."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from PIL import Image, ImageDraw
from summarize_training_methods import sha

from lineartgen.config import load_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, discover, split_experiment
from lineartgen.diffusion import noise_map
from lineartgen.masks import rectangle_candidates, rectangle_crossing_scores
from lineartgen.sketch import augment_sketch
from lineartgen.training.comparison import comparison_configs


def audit_training(root):
    record = json.loads((root / "sampling-diagnostic.json").read_text())
    rows = [json.loads(s) for s in (root / "training/timings.jsonl").read_text().splitlines()]
    for draw in record["records"]:
        row = rows[draw["step"] - 1]
        np.testing.assert_array_equal(row["strengths"]["boundary8"], draw["strengths"])
        np.testing.assert_array_equal(row["mask_areas"]["boundary8"], draw["areas"])
    path = root / "sampling-training-replay.json"
    path.write_text(
        json.dumps(
            dict(
                updates=record["updates"],
                exact=True,
                diagnostic_sha256=sha(root / "sampling-diagnostic.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--audit-training", action="store_true")
    args = parser.parse_args()
    if args.audit_training:
        audit_training(args.root)
        return
    config = load_config(args.root / "config.toml")
    variants = comparison_configs(config, study="mask_sampling")
    training, _, _ = split_experiment(discover(config.data), config.data, config.train.seed)
    output = args.root / "sampling-diagnostic.json"
    if output.exists():
        raise FileExistsError(output)

    @jax.jit
    def inspect(clean, key):
        source = augment_sketch(clean, jax.random.fold_in(key, 0x534B4554), config.sketch)
        map_key = jax.random.split(key)[0]
        masks = [noise_map(source, map_key, c.diffusion) for c in variants.values()]
        draws = jax.random.uniform(map_key, (source.shape[0], 10, 1, 1, 1))[:, :, 0, 0, 0]
        eligible = (draws[:, 1] > 2 / 3) & (draws[:, 8] >= config.diffusion.identity_probability)
        bounds = [rectangle_candidates(m > 0, map_key, 1) for m in masks]
        scores = [rectangle_crossing_scores(source, b)[:, 0] for b in bounds]
        return (source, masks, eligible, scores, bounds)

    stream = batches(
        training,
        config.data,
        batch_size=config.train.effective_batch_size,
        count=args.updates,
        seed=config.train.seed,
    )
    records, before, after, changed, previews = [], [], [], [], []
    try:
        for step, batch in enumerate(stream):
            clean, _ = mix_geometry(
                batch.images,
                config.geometry,
                seed=config.train.seed,
                step=step,
                image_scale=config.data.image_scale,
            )
            source, masks, eligible, scores, bounds = jax.device_get(
                inspect(clean, jax.random.fold_in(jax.random.PRNGKey(config.train.seed), step))
            )
            a, b = masks
            strengths = b.max((1, 2, 3))
            areas = (b > 0).sum((1, 2, 3))
            np.testing.assert_array_equal(a.max((1, 2, 3)), strengths)
            np.testing.assert_array_equal((a > 0).sum((1, 2, 3)), areas)
            np.testing.assert_array_equal(a[~eligible], b[~eligible])
            assert np.all(scores[1][eligible] >= scores[0][eligible] - 1e-4)
            before.extend(scores[0][eligible].tolist())
            after.extend(scores[1][eligible].tolist())
            different = (a != b).any((1, 2, 3))
            changed.extend(different[eligible].tolist())
            records.append(
                dict(
                    step=step + 1,
                    areas=areas.tolist(),
                    strengths=strengths.tolist(),
                    eligible=eligible.tolist(),
                    changed=different.tolist(),
                    scores=[s.tolist() for s in scores],
                )
            )
            for i in np.flatnonzero(eligible & different):
                if len(previews) >= 4:
                    break
                pixels = np.rint(np.clip((source[i, ..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
                row = Image.new("RGB", (1024, 536), "white")
                draw = ImageDraw.Draw(row)
                for column, label in enumerate(("uniform", "boundary8")):
                    row.paste(Image.fromarray(pixels).convert("RGB"), (512 * column, 24))
                    y0, x0, y1, x1 = map(int, bounds[column][i, 0])
                    draw.rectangle(
                        (x0 + 512 * column, y0 + 24, x1 - 1 + 512 * column, y1 + 23),
                        outline=(220, 0, 0),
                        width=2,
                    )
                    draw.text(
                        (512 * column + 8, 5), f"{label}, update {step + 1}, item {i}", fill="black"
                    )
                previews.append(row)
    finally:
        stream.close()
    assert before and len(records) == args.updates
    canvas = Image.new("RGB", (1024, 536 * len(previews)), "white")
    for i, row in enumerate(previews):
        canvas.paste(row, (0, i * 536))
    preview = args.root / "sampling-preview.png"
    canvas.save(preview)
    result = dict(
        updates=args.updates,
        examples=args.updates * config.train.effective_batch_size,
        rectangle_examples=len(before),
        changed=int(sum(changed)),
        mean_crossing_ink=dict(control=float(np.mean(before)), boundary8=float(np.mean(after))),
        nonzero_fraction=dict(
            control=float(np.mean(np.array(before) > 0)),
            boundary8=float(np.mean(np.array(after) > 0)),
        ),
        area_strength_nonrectangle_exact=True,
        records=records,
        config_sha256=sha(args.root / "config.toml"),
        preview_sha256=sha(preview),
        note="First training crops, before fitting; source-ink exposure is not a quality metric.",
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
