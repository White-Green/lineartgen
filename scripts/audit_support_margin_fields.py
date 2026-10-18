"""HP initial-field diagnosis on ten previously reviewed inputs, before training."""

import argparse
from pathlib import Path

import jax
import numpy as np
import sign_reduction_audit as prior
from campaign_protocol import read, sha, write
from learned_coarse_axis_model import load_field
from pooled_axis_study import HO, check
from subpixel_field_model import coverage
from support_margin_objective import support_margin


def main(root):
    check(root)
    assert jax.devices()[0].platform != "cpu"
    prior.PARENT = HO
    model, _ = load_field(root / "initials/consistent")
    out = root / "diagnostics/support-fields"
    out.mkdir(parents=True, exist_ok=False)
    rows, bindings = [], {}
    for case in prior.CASES:
        p, targets, source, strength, expected, meta = prior.inputs(case)
        fields = prior.fields_for(model, p)
        image = np.where(strength > 0, np.asarray(coverage(fields[0]))[0], source)
        np.testing.assert_allclose(image, expected, atol=1e-5, rtol=1e-5)
        value, grad = jax.jit(jax.value_and_grad(support_margin))(
            fields[0], targets[0], p.noise_levels[0]
        )
        f, g = np.asarray(fields[0])[0, ..., 0], np.asarray(grad)[0, ..., 0]
        lower = np.asarray(targets[0])[0, ..., 0].min(0)
        active = np.repeat(np.repeat(strength[..., 0] > 0, 4, 0), 4, 1)
        alpha = np.clip(0.5 - 4 * f, 0, 1)
        faint = active & (lower > 1) & (alpha > 0) & (alpha < 0.25)
        wrong = active & (lower > 1) & (f < 0.25)
        assert np.isfinite(value) and np.isfinite(g).all()
        assert np.all(g[wrong] < 0) and np.all(g[~wrong] == 0)
        assert np.all(g[faint] < 0)
        file = out / f"{case}.npz"
        np.savez_compressed(
            file, field=f, gradient=g, teacher_min=lower, active=active, faint_exterior=faint
        )
        bindings[str(file)] = sha(file)
        for key in ("input_file", "prediction_file"):
            bindings[meta[key]] = sha(Path(meta[key]))
        rows.append(
            dict(
                case=case,
                support_margin=float(value),
                faint_exterior_count=int(faint.sum()),
                faint_whiteward_count=int((g[faint] < 0).sum()),
                wrong_count=int(wrong.sum()),
                maximum_prediction_difference=float(np.max(abs(image - expected))),
                metadata=meta,
            )
        )
        print(rows[-1], flush=True)
    assert sum(r["faint_exterior_count"] for r in rows) > 0
    write(
        out / "results.json",
        dict(
            passed=True,
            rows=rows,
            bindings=bindings,
            new_native_views=0,
            reference_review=str(HO / "diagnostics/all-native-observations.json"),
            reference_review_sha256=sha(HO / "diagnostics/all-native-observations.json"),
            limitation="Field derivative only; neither a CNN update nor a quality result.",
        ),
    )
    assert read(root / "plan.json")["steps"] == 1200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    main(parser.parse_args().root.resolve())
