"""Ambiguous shifted-line controls for local opacity transport; no model training."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import sha, write

from lineartgen.config import load_config
from lineartgen.training.local_transport import local_opacity_transport
from lineartgen.training.losses import image_loss


def run(root, config_path, distance):
    assert not root.exists()
    root.mkdir(parents=True)
    config = load_config(config_path)
    weights = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
    write(
        root / "plan.json",
        dict(
            widths=[0.6, 1.4, 3.4],
            shifts=[1, 2],
            tile=16,
            distance=distance,
            candidates="41 interpolations, their opaque union and blank",
            coefficients=weights,
            config=str(config_path.resolve()),
            config_sha256=sha(config_path),
            diagnostic_only=True,
            model_training=False,
        ),
    )
    rows, arrays, summaries = [], {}, []
    for width in (0.6, 1.4, 3.4):
        x = np.arange(128)
        coverage = np.clip(
            np.minimum(x + 0.5, 56.25 + width / 2) - np.maximum(x - 0.5, 56.25 - width / 2), 0, 1
        )
        alpha = np.zeros((128, 128, 1), np.float32)
        alpha[16:112, :, 0] = coverage
        for shift in (1, 2):
            left, right = (1 - 2 * np.roll(alpha, d, axis=1) for d in (-shift, shift))
            targets = jnp.asarray(np.stack((left, right)))
            candidates = [(1 - u) * left + u * right for u in np.linspace(0, 1, 41)]
            candidates += [np.minimum(left, right), np.ones_like(left)]
            names = [f"mix-{u:.3f}" for u in np.linspace(0, 1, 41)] + ["union", "blank"]
            noise = jnp.full_like(targets, 0.5)
            configs = (config.loss, config.loss.model_copy(update={"opacity_deficit_weight": 0.0}))

            @jax.jit
            @jax.vmap
            def score(candidate, targets=targets, noise=noise, configs=configs):
                prediction = jnp.stack((candidate, candidate))
                bases = [
                    image_loss(
                        prediction,
                        targets,
                        noise,
                        c,
                        config.diffusion.min_noise,
                        preclip=prediction,
                    )[0]
                    for c in configs
                ]
                transport = local_opacity_transport(
                    prediction, targets, noise > 0, distance=distance
                )
                return jnp.stack((*bases, transport))

            values = np.asarray(score(jnp.asarray(np.stack(candidates))))
            for name, value in zip(names, values, strict=True):
                rows.append(
                    dict(
                        width=width,
                        shift=shift,
                        candidate=name,
                        base_with_deficit=float(value[0]),
                        base_without_deficit=float(value[1]),
                        transport=float(value[2]),
                    )
                )
            for keep_deficit in (True, False):
                for weight in weights:
                    total = values[:, 0 if keep_deficit else 1] + weight * values[:, 2]
                    best = np.flatnonzero(total <= total.min() + 1e-6)
                    summaries.append(
                        dict(
                            width=width,
                            shift=shift,
                            keep_deficit=keep_deficit,
                            coefficient=weight,
                            best=[names[i] for i in best],
                            minimum=float(total.min()),
                            sharp=float(total[0]),
                            average=float(total[20]),
                            union=float(total[-2]),
                            blank=float(total[-1]),
                        )
                    )
            key = f"w{width}-s{shift}"
            arrays[key] = np.stack((left, right, candidates[20], candidates[-2]))
    np.savez_compressed(root / "controls.npz", **arrays)
    write(
        root / "metrics.json",
        dict(
            rows=rows,
            summaries=summaries,
            script_sha256=sha(Path(__file__)),
            controls_sha256=sha(root / "controls.npz"),
            source_sha256=sha(Path("src/lineartgen/training/local_transport.py")),
            plan_sha256=sha(root / "plan.json"),
            model_training=False,
            adoption=False,
        ),
    )
    for row in summaries:
        if row["coefficient"] in (0.0, 1.0, 3.0, 10.0):
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--distance", choices=("squared", "smooth_absolute"), default="squared")
    args = parser.parse_args()
    run(args.root, args.config, args.distance)
