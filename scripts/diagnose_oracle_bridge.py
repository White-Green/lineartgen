"""Check a candidate conditional path against the unchanged runtime algebra.

The callback is an oracle: it returns the known clean target. These are NOT
learned model predictions and cannot establish a quality gain.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from lineartgen_runtime.sampling import denoise, pyramid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def path_state(source, target, strength, noise, u):
    return (
        np.sqrt(1 - strength * u) * target
        + np.sqrt(u) * np.sqrt(1 - strength) * (source - target)
        + np.sqrt(strength * u) * noise
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root.resolve()
if (root / "results.json").exists():
    raise FileExistsError(root / "results.json")
plan = json.loads((root / "plan.json").read_text())
source_root = Path(plan["source"])
runtime_path = Path("packages/lineartgen-runtime/src/lineartgen_runtime/sampling.py")
assert plan["adoption"] is False and plan["confirmation_opened"] is False
rows, path_checks = [], []
for split in plan["splits"]:
    data = np.load(source_root / f"{split}.npz")
    metadata = json.loads((source_root / f"{split}.json").read_text())
    assert len(data["source"]) == len(metadata) == plan["pairs_per_split"]
    for pair, (rough, clean, meta) in enumerate(
        zip(data["source"], data["clean"], metadata, strict=True)
    ):
        rough, clean = rough[None], clean[None]
        for strength_value in plan["strengths"]:
            for mask in plan["masks"]:
                strength = np.full_like(rough, strength_value)
                if mask == "partial":
                    strength[:, :103] = 0
                    strength[:, :, :171] = 0
                target = np.where(strength > 0, clean, rough)
                sources = pyramid(rough, plan["levels"])
                targets = pyramid(target, plan["levels"])
                strengths = pyramid(strength, plan["levels"])
                seed = plan["seed"] + pair + (0 if split == "train" else 1000)
                rng = np.random.default_rng(seed)
                noises = [rng.standard_normal(x.shape, dtype=np.float32) for x in sources]
                for level, (r, c, s, e) in enumerate(
                    zip(sources, targets, strengths, noises, strict=True)
                ):
                    support = (c < 0.9) & (s > 0)
                    for u in plan["progress"]:
                        state = path_state(r, c, s, e, np.float32(u))
                        legacy = np.sqrt(1 - s * u) * r + np.sqrt(s * u) * e
                        delta = (np.sqrt(u) * np.sqrt(1 - s) - np.sqrt(1 - s * u)) * (r - c)
                        error = float(np.max(np.abs((state - legacy) - delta)))
                        assert error < plan["tolerances"]["float32_path"]
                        if meta["family"] == "clean":
                            assert np.max(np.abs(delta)) == 0
                        rows.append(
                            dict(
                                split=split,
                                pair=pair,
                                family=meta["family"],
                                kind=meta["kind"],
                                strength=strength_value,
                                mask=mask,
                                level=level,
                                progress=u,
                                identity_error=error,
                                maximum_state_shift=float(np.max(np.abs(delta))),
                                mean_ink_shift_on_target=(
                                    float(np.mean(np.abs(delta[support])) / 2)
                                    if np.any(support)
                                    else None
                                ),
                            )
                        )
                for steps in plan["steps"]:
                    visited = []

                    def oracle(
                        image,
                        noise_level,
                        insert,
                        guide,
                        *,
                        visited=visited,
                        steps=steps,
                        sources=sources,
                        targets=targets,
                        strengths=strengths,
                        noises=noises,
                    ):
                        call = len(visited)
                        step = call // plan["levels"]
                        level = plan["levels"] - 1 - call % plan["levels"]
                        u = np.float32((steps - step) / steps)
                        expected = path_state(
                            sources[level], targets[level], strengths[level], noises[level], u
                        )
                        error = float(np.max(np.abs(image - expected)))
                        assert error < plan["tolerances"]["float32_path"], error
                        np.testing.assert_array_equal(guide, sources[level])
                        np.testing.assert_array_equal(noise_level, strengths[level] * u)
                        if call % plan["levels"]:
                            np.testing.assert_array_equal(insert, targets[level + 1])
                        else:
                            assert not np.any(insert)
                        visited.append(error)
                        return targets[level]

                    output = denoise(
                        oracle,
                        rough,
                        strength,
                        seed=seed,
                        steps=steps,
                        levels=plan["levels"],
                        source_conditioning=True,
                        preserve_zero_noise=True,
                    )
                    np.testing.assert_array_equal(output, target)
                    np.testing.assert_array_equal(output[strength == 0], rough[strength == 0])
                    assert len(visited) == steps * plan["levels"]
                    path_checks.append(
                        dict(
                            split=split,
                            pair=pair,
                            strength=strength_value,
                            mask=mask,
                            steps=steps,
                            calls=len(visited),
                            maximum_path_error=max(visited),
                            final_exact=True,
                            protected_exact=True,
                        )
                    )
        write(root / "progress.json", dict(split=split, pair=pair, pairs_per_split=len(metadata)))

# Independent finite-difference check in float64, away from the singular endpoint.
rng = np.random.default_rng(plan["seed"])
r, c, e = (rng.normal(size=2048) for _ in range(3))
s = rng.uniform(0.001, 0.99, size=r.shape)
velocity_checks = []
for u in (0.2, 0.5, 0.8):
    x = path_state(r, c, s, e, u)
    velocity = x / (2 * u) - c / (2 * u * np.sqrt(1 - s * u))
    h = 1e-5
    numeric = (path_state(r, c, s, e, u + h) - path_state(r, c, s, e, u - h)) / (2 * h)
    error = float(np.max(np.abs(numeric - velocity)))
    assert error < plan["tolerances"]["float64_velocity"]
    velocity_checks.append(dict(progress=u, maximum_error=error))

summary = []
for split in plan["splits"]:
    for family in ("clean", "tone", "blur"):
        for strength in (0.1, 0.5):
            for u in plan["progress"]:
                selected = [
                    x
                    for x in rows
                    if x["split"] == split
                    and x["family"] == family
                    and x["strength"] == strength
                    and x["progress"] == u
                    and x["mask"] == "whole"
                    and x["level"] == 0
                    and x["mean_ink_shift_on_target"] is not None
                ]
                summary.append(
                    dict(
                        split=split,
                        family=family,
                        strength=strength,
                        progress=u,
                        mean_ink_shift_on_target=float(
                            np.mean([x["mean_ink_shift_on_target"] for x in selected])
                        ),
                        conditions=len(selected),
                    )
                )
result = dict(
    plan_sha256=sha(root / "plan.json"),
    script_sha256=sha(__file__),
    runtime_sha256=sha(runtime_path),
    source_data={
        f"{split}.{ext}": sha(source_root / f"{split}.{ext}")
        for split in plan["splits"]
        for ext in ("json", "npz")
    },
    state_differences=rows,
    runtime_checks=path_checks,
    velocity_checks=velocity_checks,
    summary=summary,
    adoption=False,
    learned_predictions=0,
    confirmation_opened=False,
)
write(root / "results.json", result)
print(
    json.dumps(
        dict(
            runtime_cases=len(path_checks),
            state_differences=len(rows),
            maximum_path_error=max(x["maximum_path_error"] for x in path_checks),
            velocity_checks=velocity_checks,
        ),
        indent=2,
    ),
    flush=True,
)
