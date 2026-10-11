"""Fourier modulation against both frozen BV anchors on identical Gaussian draws."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from anchored_endpoint import velocity as anchored_velocity
from campaign_protocol import read, sha, write
from PIL import Image, ImageDraw
from progress_modulation_fit import METHODS, STEPS, check
from velocity_bridge_toy import EVAL_SEED, measure, sample, targets

from lineartgen.models.serialization import load_inference


def evaluate(root):
    plan = check(root)
    assert jax.devices()[0].platform != "cpu"
    assert read(root / "snapshot-audit.json")["raw_ema_exact"]
    parent = Path(plan["parent"])
    paths = {m + "_control": root / "control" / m for m in METHODS} | {
        m: root / f"snapshots/step-{STEPS}/{m}/raw" for m in METHODS
    }
    models = {name: load_inference(p)[0] for name, p in paths.items()}
    out = root / "evaluation"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    write(
        out / "checkpoint-binding.json",
        dict(files={str(f): sha(f) for p in paths.values() for f in p.iterdir()}),
    )
    old = read(parent / "evaluation/metrics.json")
    old_rows = {
        (r["method"], r["depth"], r["partial"], r["strength"], r["steps"]): r
        for r in old["rows"]
        if r["method"] in METHODS
    }
    a, b, s = targets()
    rows, displays, bindings, replay = [], {}, {}, []
    for depth in (1, 2, 3):
        for partial in (False, True):
            for strength in (0.1, 0.5):
                source = jnp.broadcast_to(jnp.asarray(s), (64, *s.shape))
                t = jnp.full_like(source, strength)
                if partial:
                    t = t.at[:, :, :32].set(0)
                key = jax.random.fold_in(
                    jax.random.PRNGKey(EVAL_SEED),
                    depth * 4 + int(partial) * 2 + int(strength == 0.5),
                )
                group = f"d{depth}-p{int(partial)}-t{strength}"
                displays[group] = {
                    "A": np.where(np.asarray(t[0]) > 0, a, s),
                    "B": np.where(np.asarray(t[0]) > 0, b, s),
                    "rough": s,
                }
                for method, model in models.items():
                    anchor = method.removesuffix("_control")

                    def predict(m, x, t, s, u, a=anchor == "source"):
                        return anchored_velocity(m, x, t, s, u, a)

                    for steps in (1, 8, 32):
                        physical = np.asarray(
                            sample(
                                lambda x, t, s, u, m=model, f=predict: f(m, x, t, s, u),
                                source,
                                t,
                                key,
                                depth=depth,
                                steps=steps,
                            )
                        )
                        prediction = np.clip(physical, -1, 1)
                        metrics = measure(prediction, a, b, np.asarray(source), np.asarray(t))
                        assert (
                            metrics["finite"]
                            and np.isfinite(physical).all()
                            and metrics["protected_max_error"] == 0
                        )
                        row = dict(
                            method=method,
                            depth=depth,
                            partial=partial,
                            strength=strength,
                            steps=steps,
                            **metrics,
                            unclipped_minimum=float(physical.min()),
                            unclipped_maximum=float(physical.max()),
                        )
                        if method.endswith("_control"):
                            p = parent / f"evaluation/predictions/{group}-{anchor}-{steps}.npz"
                            assert sha(p) == old["bindings"][str(p)]
                            with np.load(p) as previous:
                                np.testing.assert_array_equal(physical, previous["physical"])
                                np.testing.assert_array_equal(prediction, previous["prediction"])
                            assert {**row, "method": anchor} == old_rows[
                                anchor, depth, partial, strength, steps
                            ]
                            replay.append(
                                dict(
                                    anchor=anchor,
                                    group=group,
                                    steps=steps,
                                    physical_and_clipped_exact=True,
                                )
                            )
                        rows.append(row)
                        p = out / f"predictions/{group}-{method}-{steps}.npz"
                        np.savez_compressed(p, physical=physical, prediction=prediction)
                        bindings[str(p)] = sha(p)
                        if steps == 32:
                            for i in range(4):
                                displays[group][f"{method}[{i}]"] = prediction[i]
                write(out / "progress.json", dict(groups=len(displays), total=12, rows=len(rows)))
                print(f"evaluated {len(displays)}/12; rows={len(rows)}", flush=True)
    assert len(rows) == 144 and len(replay) == 72
    checks = []
    for r in rows:
        if r["steps"] != 32:
            continue
        for name, passed in dict(
            near_modes=r["mean_relative_endpoint_rms"] <= 0.15,
            mostly_resolved=r["fraction_within_020"] >= 0.8,
            both_modes=0.1 <= r["nearest_A"] / 64 <= 0.9,
        ).items():
            checks.append(
                {
                    **{k: r[k] for k in ("method", "depth", "partial", "strength")},
                    "guard": name,
                    "passed": passed,
                }
            )
    assert len(checks) == 144
    write(
        out / "metrics.json",
        dict(
            rows=rows,
            checks=checks,
            bindings=bindings,
            finite=True,
            protected_exact=True,
            samples_per_group=64,
            adoption=False,
        ),
    )
    write(root / "replay-audit.json", dict(outputs_exact=72, metrics_exact=True, rows=replay))
    records = []
    for anchor in METHODS:
        for depth in (1, 2, 3):
            methods = (anchor + "_control", anchor)
            columns = ("A", "B", "rough", *[f"{name}[{i}]" for name in methods for i in range(4)])
            canvas = Image.new("L", (104 * len(columns), 116 * 4), 255)
            draw = ImageDraw.Draw(canvas)
            for row, (partial, strength) in enumerate(
                (p, t) for p in (False, True) for t in (0.1, 0.5)
            ):
                group = f"d{depth}-p{int(partial)}-t{strength}"
                draw.text((2, row * 116 + 2), group, fill=0)
                for col, label in enumerate(columns):
                    pixels = np.rint(
                        np.clip((displays[group][label][..., 0] + 1) * 127.5, 0, 255)
                    ).astype(np.uint8)
                    draw.text((col * 104 + 2, row * 116 + 18), label, fill=0)
                    canvas.paste(Image.fromarray(pixels), (col * 104 + 20, row * 116 + 40))
            path = out / f"{anchor}-depth-{depth}-native.png"
            canvas.save(path)
            records.append(dict(file=path.name, sha256=sha(path)))
    write(out / "render-manifest.json", dict(records=records, first_four_fixed=True))
    write(
        root / "workflow-status.json", dict(stage="numerical_complete_pending_review", step=STEPS)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    evaluate(p.parse_args().root.resolve())
