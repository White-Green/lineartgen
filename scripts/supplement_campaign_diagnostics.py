"""Complete preregistered depth/iteration diagnostics without changing selection rules."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from summarize_sketch_rates import summaries

from lineartgen.config import load_config


def inherited_lock(root):
    descriptors = []
    for path in Path("/proc/self/fd").iterdir():
        try:
            if path.readlink() == root / "controller.lock":
                descriptors.append(int(path.name))
        except FileNotFoundError:
            pass
    assert len(descriptors) == 1, "Run development analysis through the locked campaign controller"
    return tuple(descriptors)


def validate(path, arm, cases, step, mode, steps, recursion):
    result = read(path)
    manifest = read(cases / "cases.json")
    config = load_config(arm / "config.json")
    widths = 1 + len(config.model.nested_channels)
    expected = set(itertools.product(range(len(manifest["cases"])), range(widths), steps))
    actual = [(r["case"], r["subnet"], r["denoise_steps"]) for r in result["results"]]
    assert len(actual) == len(expected) and set(actual) == expected
    assert result["manifest_sha256"] == sha(cases / "cases.json")
    assert result["cases"] == len(manifest["cases"]) and set(result["denoise_steps"]) == set(steps)
    assert result["checkpoint_step"] == step and result["ema"] == (mode == "ema")
    assert result["noise_generator"] == "numpy" and result["recursion"] == recursion
    for row in result["results"]:
        case = manifest["cases"][row["case"]]
        levels = (
            3
            if recursion == "runtime"
            else config.diffusion.levels_for_scale(
                512, config.model.minimum_size, case["scale_level"]
            )
        )
        assert row["method"] == "model" and row["recursion_levels"] == levels
        assert row["protected_max_error"] in (None, 0.0)
        assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    return result


def run_diagnostics(root):
    """Runs only while the controller waits for development analysis, after all training."""
    root = root.resolve()
    cases = root / "development"
    cohorts = read(root / "cohorts.json")
    step = read(root / "decisions/budget.json")["step"]
    jobs = []
    for arm in sorted((root / "arms").glob("seed42_*pool_d*")):
        for mode in ("raw", "ema"):
            jobs.append((arm, 10000, mode, (1,), "dataset", "architecture-depths"))
    for name in cohorts["42"].values():
        arm = root / "arms" / name
        jobs.append((arm, step, "ema", (1,), "dataset", "final-depths"))
        jobs.append((arm, step, "ema", (2, 5), "runtime", "iteration-count"))
    records, artifacts = [], {}
    for arm, checkpoint, mode, iterations, recursion, label in jobs:
        output = arm / "diagnostics" / f"{label}-{checkpoint}-{mode}"
        path = output / "metrics.json"
        if not path.exists():
            if (root / "stop").exists():
                raise InterruptedError("Campaign stop file exists")
            for source, digest in read(root / "training-source.json")["file_sha256"].items():
                assert sha(source) == digest, source
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
            # Parent statistics use CPU. The child inherits the controller's GPU access.
            env["JAX_PLATFORMS"] = env.get("LINEARTGEN_DIAGNOSTIC_PLATFORMS", "rocm,cpu")
            log = root / "logs" / f"{arm.name}-{label}-{checkpoint}-{mode}.log"
            write(
                root / "diagnostic-status.json",
                dict(
                    arm=arm.name,
                    label=label,
                    step=checkpoint,
                    mode=mode,
                    source_sha256=sha(__file__),
                    evaluator_sha256=sha("scripts/evaluate_training_methods.py"),
                ),
            )
            print(f"Development diagnostic: {arm.name} {label} {checkpoint} {mode}", flush=True)
            # The original controller still holds its lock throughout this call.
            with log.open("a") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/evaluate_training_methods.py",
                        "--config",
                        str(arm / "config.json"),
                        "--variants",
                        str(arm / "variants.json"),
                        "--cases",
                        str(cases),
                        "--run-group",
                        str(arm / "training"),
                        "--step",
                        str(checkpoint),
                        "--steps",
                        *map(str, iterations),
                        "--recursion",
                        recursion,
                        "--noise-generator",
                        "numpy",
                        "--output",
                        str(output),
                        *(["--raw"] if mode == "raw" else []),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    pass_fds=inherited_lock(root),
                )
        result = validate(path, arm, cases, checkpoint, mode, iterations, recursion)
        artifacts[str(path)] = sha(path)
        for denoise_steps in iterations:
            for depth in sorted({r["recursion_levels"] for r in result["results"]}):
                selected = [
                    r
                    for r in result["results"]
                    if r["denoise_steps"] == denoise_steps and r["recursion_levels"] == depth
                ]
                records.append(
                    dict(
                        arm=arm.name,
                        checkpoint_step=checkpoint,
                        mode=mode,
                        label=label,
                        recursion=recursion,
                        recursion_levels=depth,
                        denoise_steps=denoise_steps,
                        summary=summaries(dict(results=selected)),
                    )
                )
    result = dict(
        records=records,
        artifacts=artifacts,
        source_sha256=sha(__file__),
        selection_policy="Descriptive development diagnostics only. Architecture adoption, "
        "training budget and EMA/denoise1 size selection retain their frozen criteria.",
        limitations="Dataset recursion is paired to scale_level, so different depths also "
        "contain different examples. It is not a causal comparison of recursion depths.",
    )
    write(root / "diagnostic-summary.json", result)
    write(root / "diagnostic-status.json", dict(stage="complete", reports=len(jobs)))
    return result
