"""Diagnostic AT endpoint errors on known mean trajectories, no AV learning."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from learned_bridge_toy import endpoint, targets

from lineartgen.diffusion import pyramid
from lineartgen.models.serialization import load_inference


def run(root):
    plan = read(root / "plan.json")
    parent = Path(plan["control_parent"])
    model, _ = load_inference(parent / "snapshots/step-6000/bridge/raw")
    a, b, s = targets()
    rows = []
    for partial in (False, True):
        for strength in (0.1, 0.5):
            source = jnp.broadcast_to(jnp.asarray(s), (2, *s.shape))
            t = jnp.full_like(source, strength)
            if partial:
                t = t.at[:, :, :32].set(0)
            teacher = jnp.where(t > 0, jnp.asarray(np.stack((a, b))), source)
            half = jnp.where(t > 0, (jnp.asarray(a) - jnp.asarray(b)) / 2, 0)
            mean = jnp.where(t > 0, (jnp.asarray(a) + jnp.asarray(b)) / 2, source)
            levels, guides = pyramid(t, 3), pyramid(source, 3)
            for remaining in (1.0, 0.5, 0.1, 0.01, 0.001, 0.0001):
                tau = 1 - remaining
                x = remaining * source + tau * teacher
                images = pyramid(x, 3)
                progress = tuple(jnp.full_like(v, remaining) for v in images)
                prediction = np.asarray(endpoint(model, images, levels, guides, progress)[0])
                center = remaining * source + tau * mean
                precision = jnp.where(t > 0, 1 / jnp.maximum((0.25 * t * remaining) ** 2, 1e-30), 0)
                logit = tau * jnp.sum(
                    (x - center) * half * precision, axis=(1, 2, 3), keepdims=True
                )
                oracle = np.asarray(mean + half * jnp.tanh(logit))
                for i, name in enumerate(("A", "B")):
                    active = np.asarray(t[i]) > 0
                    error = float(
                        np.sqrt(np.mean((prediction[i][active] - oracle[i][active]) ** 2))
                    )
                    rows.append(
                        dict(
                            partial=partial,
                            strength=strength,
                            teacher=name,
                            remaining=remaining,
                            endpoint_error_rms=error,
                            implied_velocity_error_rms=error / remaining,
                        )
                    )
    assert len(rows) == 48 and all(np.isfinite(r["implied_velocity_error_rms"]) for r in rows)
    write(
        root / "parent-terminal-error.json",
        dict(
            rows=rows,
            known_teacher_trajectories=True,
            neural_updates=0,
            parent_source_sha256=sha(parent / "source.json"),
            script_sha256=sha(Path(__file__)),
            interpretation=(
                "Physical endpoint errors vs analytic conditional mean on known paths; "
                "not AV quality"
            ),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root.resolve())
