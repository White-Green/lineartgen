"""Analytical posterior diagnostic on known teacher-interpolated toy paths."""

import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import endpoints
from campaign_protocol import read, sha, write
from diagnose_learnability import freeze, verify
from learned_bridge_toy import path_inputs
from PIL import Image, ImageDraw
from scipy.special import expit, xlogy
from velocity_bridge_toy import EVAL_SEED, targets

from lineartgen.diffusion import pyramid
from lineartgen.models.serialization import load_inference

TIMES = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95)
DISPLAY_TIMES = (0.0, 0.002, 0.01, 0.1, 0.95)
METHODS = ("bv_noisy", "bv_source", "bx_noisy", "bx_source")


def posterior(images, levels, guides, a_levels, b_levels, tau):
    """Equal-prior likelihood ratio with independent per-scale Gaussian noise."""
    if not 0 <= tau < 1:
        raise ValueError("Posterior requires 0 <= tau < 1")
    odds = np.zeros(images[0].shape[0], np.float64)
    for x, t, s, a, b in zip(images, levels, guides, a_levels, b_levels, strict=True):
        x, t, s, a, b = (np.asarray(v, np.float64) for v in (x, t, s, a, b))
        mu_a, mu_b = (1 - tau) * s + tau * a, (1 - tau) * s + tau * b
        variance = ((1 - tau) * 0.25 * t) ** 2
        term = (x - (mu_a + mu_b) / 2) * (mu_a - mu_b) / np.where(t > 0, variance, 1)
        odds += np.where(t > 0, term, 0).sum((1, 2, 3))
    p = expit(odds)
    weight = p[:, None, None, None]
    mean = weight * np.asarray(a_levels[0], np.float64) + (1 - weight) * np.asarray(
        b_levels[0], np.float64
    )
    return p, mean, odds


def prepare(root):
    assert not root.exists()
    parents = {
        tag: Path(f"../jax-{name}-20261011/runs/{name}-20261011").resolve()
        for tag, name in (("bv", "anchored-endpoint"), ("bx", "progress-modulation"))
    }
    for p in parents.values():
        verify(p)
    assert (
        read(parents["bv"] / "final-audit.json")["stage"] == "complete_anchored_endpoint_diagnostic"
    )
    (root / "configs").mkdir(parents=True)
    shutil.copytree(parents["bv"] / "initial", root / "initial")
    for name in ("inputs.npz", "probe.json"):
        shutil.copyfile(parents["bv"] / name, root / name)
    for tag, p in parents.items():
        for anchor in ("noisy", "source"):
            shutil.copyfile(p / f"configs/{anchor}.json", root / f"configs/{tag}_{anchor}.json")
    write(root / "train.json", dict(neural_updates=0, checkpoint=6000, times=list(TIMES)))
    bound = [
        p / name
        for p in parents.values()
        for name in ("source.json", "plan.json", "test-results.json")
    ]
    write(
        root / "plan.json",
        dict(
            initial=str(root / "initial"),
            initial_files={p.name: sha(p) for p in (root / "initial").iterdir()},
            data={"toy": dict(path=str(root / "inputs.npz"), sha256=sha(root / "inputs.npz"))},
            parents={k: str(v) for k, v in parents.items()},
            bound_files={str(p): sha(p) for p in bound},
            plan_document_sha256=sha(Path("docs/endpoint-posterior-plan-20261011.md")),
            methods=list(METHODS),
            adoption=False,
            teacher_conditioned_diagnostic=True,
        ),
    )


def check(root):
    verify(root)
    p = read(root / "plan.json")
    for parent in p["parents"].values():
        verify(Path(parent))
    for path, digest in p["bound_files"].items():
        assert sha(Path(path)) == digest
    assert sha(Path("docs/endpoint-posterior-plan-20261011.md")) == p["plan_document_sha256"]
    return p


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu" and read(root / "test-results.json")["passed"]
    parents = {k: Path(v) for k, v in plan["parents"].items()}
    assert (
        read(parents["bx"] / "final-audit.json")["stage"]
        == "complete_progress_modulation_diagnostic"
    )
    paths = {m: parents[m[:2]] / f"snapshots/step-6000/{m[3:]}/raw" for m in METHODS}
    models = {m: load_inference(p)[0] for m, p in paths.items()}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True)
    bound = [p for folder in paths.values() for p in folder.iterdir()]
    bound += [p / "final-audit.json" for p in parents.values()]
    write(out / "checkpoint-binding.json", dict(files={str(p): sha(p) for p in bound}))
    a, b, s = targets()
    with np.load(root / "inputs.npz") as data:
        for k, v in (("a", a), ("b", b), ("source", s)):
            np.testing.assert_array_equal(data[k], v)
    rows, bindings, displays = [], {}, {}
    for depth in (1, 3):
        for partial in (False, True):
            for strength in (0.1, 0.5):
                source = jnp.broadcast_to(jnp.asarray(s), (64, *s.shape))
                t = jnp.full_like(source, strength)
                if partial:
                    t = t.at[:, :, :32].set(0)
                labels = jnp.arange(64) % 2
                clean = jnp.where(labels[:, None, None, None] == 0, jnp.asarray(a), jnp.asarray(b))
                aa = pyramid(jnp.where(t > 0, jnp.asarray(a), source), depth)
                bb = pyramid(jnp.where(t > 0, jnp.asarray(b), source), depth)
                key = jax.random.fold_in(
                    jax.random.PRNGKey(EVAL_SEED),
                    depth * 4 + int(partial) * 2 + int(strength == 0.5),
                )
                denom = float(
                    np.sum(
                        ((np.asarray(aa[0][0], np.float64) - np.asarray(bb[0][0], np.float64)) / 2)
                        ** 2
                    )
                )
                ambiguous = np.asarray(aa[0]) != np.asarray(bb[0])
                active = np.asarray(t) > 0
                common = active & ~ambiguous
                group = f"d{depth}-p{int(partial)}-t{strength}"
                for tau in TIMES:
                    values = path_inputs(source, clean, t, key, jnp.full((64, 1, 1, 1), tau), depth)
                    args = {k: values[k] for k in ("images", "levels", "guides", "progress")}
                    probability, reference, odds = posterior(
                        values["images"], values["levels"], values["guides"], aa, bb, tau
                    )
                    assert np.isfinite(reference).all() and np.isfinite(odds).all()
                    if tau == 0:
                        np.testing.assert_array_equal(probability, 0.5)
                    arrays = dict(
                        input=np.asarray(values["images"][0]),
                        reference=reference,
                        probability=probability,
                        log_odds=odds,
                        labels=np.asarray(labels),
                    )
                    for method, model in models.items():
                        prediction = np.asarray(
                            endpoints(model, **args, source_anchor=method.endswith("source"))[1][0]
                        )
                        assert np.isfinite(prediction).all()
                        np.testing.assert_array_equal(
                            prediction[~active], np.asarray(source)[~active]
                        )
                        err = (prediction.astype(np.float64) - reference) ** 2
                        rows.append(
                            dict(
                                method=method,
                                depth=depth,
                                partial=partial,
                                strength=strength,
                                tau=tau,
                                mean_relative_posterior_rms=float(
                                    np.sqrt(err.sum((1, 2, 3)) / denom).mean()
                                ),
                                ambiguous_mse=float(err[ambiguous].mean()),
                                common_mse=float(err[common].mean()),
                                posterior_mean=float(probability.mean()),
                                posterior_entropy=float(
                                    (
                                        -xlogy(probability, probability)
                                        - xlogy(1 - probability, 1 - probability)
                                    ).mean()
                                ),
                                posterior_resolved=float(
                                    np.mean((probability < 0.1) | (probability > 0.9))
                                ),
                                irreducible_ambiguous_mse=float(
                                    np.mean(4 * probability * (1 - probability))
                                ),
                                finite=True,
                                protected_exact=True,
                            )
                        )
                        arrays[method] = prediction
                    path = out / f"predictions/{group}-tau{tau}.npz"
                    np.savez_compressed(path, **arrays)
                    bindings[str(path)] = sha(path)
                    if depth == 3 and tau in DISPLAY_TIMES:
                        displays[group, tau] = {
                            k: v[:2]
                            for k, v in arrays.items()
                            if k in ("input", "reference", *METHODS)
                        }
                print(f"evaluated {group}; rows={len(rows)}", flush=True)
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            bindings=bindings,
            finite=True,
            protected_exact=True,
            neural_updates=0,
            teacher_conditioned=True,
            adoption=False,
        ),
    )
    records = []
    for partial in (False, True):
        for strength in (0.1, 0.5):
            group = f"d3-p{int(partial)}-t{strength}"
            for example in (0, 1):
                canvas = Image.new("L", (104 * len(DISPLAY_TIMES), 104 * 6), 255)
                draw = ImageDraw.Draw(canvas)
                for col, tau in enumerate(DISPLAY_TIMES):
                    for row, method in enumerate(("input", "reference", *METHODS)):
                        pixels = np.rint(
                            np.clip(
                                (displays[group, tau][method][example, ..., 0] + 1) * 127.5, 0, 255
                            )
                        ).astype(np.uint8)
                        draw.text((col * 104 + 2, row * 104 + 2), f"{method} tau{tau}", fill=0)
                        canvas.paste(Image.fromarray(pixels), (col * 104 + 20, row * 104 + 28))
                path = out / f"{group}-teacher-{example}-native.png"
                canvas.save(path)
                records.append(dict(file=path.name, sha256=sha(path)))
    write(
        out / "render-manifest.json",
        dict(records=records, teacher_conditioned=True, first_two_fixed=True),
    )
    write(
        root / "workflow-status.json",
        dict(stage="numerical_complete_pending_review", neural_updates=0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "evaluate": evaluate}[args.action](args.root.resolve())
