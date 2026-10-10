"""Audit auxiliary interval gradients on the eight fixed real prepared tensors."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from campaign_protocol import read, sha, write
from edge_energy_fit import PROBE, TRAIN, batch

from lineartgen.config import load_config
from lineartgen.diffusion import recursive_forward_with_preclip
from lineartgen.models.serialization import load_inference
from lineartgen.training.edge_energy import edge_energy_loss
from lineartgen.training.losses import image_loss


def audit(root):
    plan = read(root / "plan.json")
    parent = Path(plan["control_parent"])
    old = load_config(parent / "configs/band.json")
    new = load_config(root / "configs/energy.json")
    a, b = old.model_dump(), new.model_dump()
    changes = [f"{s}.{k}" for s in a for k in a[s] if a[s][k] != b[s][k]]
    assert changes == ["loss.edge_energy_weight"]
    model, _ = load_inference(plan["initial"])
    folder = root / "energy-mechanism"
    folder.mkdir()
    arrays, rows = {}, []
    maximum_error = 0.0
    for split, ids in [("train", TRAIN), ("probe", PROBE)]:
        prepared = batch(root, ids, 3)
        y, t = prepared["target"], prepared["levels"][0]
        _, physical = recursive_forward_with_preclip(
            model, prepared["noisy"], prepared["levels"], guides=prepared["guides"]
        )
        z = physical[0]
        active = t > new.diffusion.min_noise
        value, g = jax.value_and_grad(edge_energy_loss)(z, y, active)
        assert np.isfinite(value) and np.isfinite(g).all()
        rate = 1e-3 / jnp.maximum(jnp.max(jnp.abs(g)), 1e-12)
        after = edge_energy_loss(z - rate * g, y, active)
        assert float(after) < float(value)
        # This spatial loss can trade neighboring errors; not every pixel points at its teacher.
        np.testing.assert_array_equal(jax.grad(edge_energy_loss, argnums=1)(z, y, active), 0)
        partial = active.at[:, :, :256].set(False)
        pg = jax.grad(edge_energy_loss)(z, y, partial)
        np.testing.assert_array_equal(np.asarray(pg)[:, :, :256], 0)

        def total_gradient(c, z, target=y, strength=t):
            return jax.grad(
                lambda p: image_loss(
                    jnp.clip(p, -1, 1), target, strength, c.loss, c.diffusion.min_noise, preclip=p
                )[0]
            )(z)

        g0, g1 = total_gradient(old, z), total_gradient(new, z)
        error = np.asarray(g1 - g0 - new.loss.edge_energy_weight * g)
        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
        np.testing.assert_allclose(g1 - g0, new.loss.edge_energy_weight * g, atol=1e-8, rtol=1e-5)
        for j, case in enumerate(ids):
            arrays[f"{case}-preclip"] = np.asarray(z[j])
            arrays[f"{case}-gradient"] = np.asarray(g[j])
            rows.append(
                dict(
                    case=case,
                    split=split,
                    gradient_nonzero=int(np.count_nonzero(np.asarray(g[j]))),
                    gradient_norm=float(np.linalg.norm(np.asarray(g[j]))),
                    base_gradient_norm=float(np.linalg.norm(np.asarray(g0[j]))),
                    weighted_aux_to_base_norm=float(
                        np.linalg.norm(np.asarray(new.loss.edge_energy_weight * g[j]))
                        / max(np.linalg.norm(np.asarray(g0[j])), 1e-12)
                    ),
                )
            )
    np.savez_compressed(folder / "arrays.npz", **arrays)
    write(
        folder / "audit.json",
        dict(
            finite_losses=True,
            direct_descent_correct=True,
            protected_gradients_zero=True,
            target_gradient_zero=True,
            maximum_additive_gradient_error=maximum_error,
            rows=rows,
            config_only_differences=changes,
            inputs_sha256=sha(root / "inputs.npz"),
            files={"arrays.npz": sha(folder / "arrays.npz")},
            source_bindings={
                str(p.resolve()): sha(p)
                for p in [
                    Path(__file__),
                    Path("src/lineartgen/training/edge_energy.py"),
                    Path("src/lineartgen/training/losses.py"),
                ]
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root.resolve())
