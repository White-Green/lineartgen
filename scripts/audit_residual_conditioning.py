"""Measure required residual-head scale on fixed fitting pairs, without fitting a model."""

import argparse
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write


def audit(root):
    out = root / "residual-conditioning"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    strengths = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9]
    floors = [0.0, 0.05, 0.1, 0.25]
    write(
        out / "plan.json",
        dict(
            strengths=strengths,
            hypothetical_floors=floors,
            input_sha256=sha(root / "train.npz"),
            script_sha256=sha(Path(__file__)),
            formula="F*=(clean-sqrt(1-t)*source)/sqrt(max(t,floor))-sqrt(t/max(t,floor))*epsilon",
            interpretation=(
                "Algebraic required head values, not learned predictions, quality, "
                "or measured parameter gradients. No model/default changes."
            ),
            protected_rule="t=0 must remain exact identity; hypothetical floor only applies at t>0",
        ),
    )
    data, records = np.load(root / "train.npz"), read(root / "train.json")
    rows = []
    for family in ("clean", "tone", "blur"):
        indices = [
            i for i, r in enumerate(records) if r["family"] == family and r["kind"] != "blank"
        ]
        target, source = data["clean"][indices], data["source"][indices]
        core = target < 0
        for strength in strengths:
            mean_input = np.sqrt(1 - strength) * source
            for floor in floors:
                scale = np.sqrt(max(strength, floor))
                required_mean = (target - mean_input) / scale
                # Check algebra on all pixels, including teacher background.
                np.testing.assert_allclose(mean_input + scale * required_mean, target, atol=3e-7)
                values = np.abs(required_mean[core])
                rows.append(
                    dict(
                        family=family,
                        strength=strength,
                        floor=floor,
                        core_pixels=int(core.sum()),
                        mean_required_abs=float(values.mean()),
                        p95_required_abs=float(np.quantile(values, 0.95)),
                        max_required_abs=float(values.max()),
                        required_head_noise_std=float(np.sqrt(strength) / scale),
                    )
                )
    write(
        out / "results.json",
        dict(rows=rows, recomposition_verified=True, plan_sha256=sha(out / "plan.json")),
    )
    for r in rows:
        if r["family"] == "blur" and r["floor"] in (0, 0.05):
            print(r)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root.resolve())
