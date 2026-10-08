"""Fixed repeated-inference local evaluation; resume only matching complete blocks."""

import argparse
import itertools
from pathlib import Path

import numpy as np
from campaign_protocol import read, sha, write
from diagnose_learnability import SEED, fixed_inputs, verify
from flax import nnx

from lineartgen.diffusion import sample_prepared
from lineartgen.lineart_metrics import halo_metrics
from lineartgen.models.serialization import load_inference


def check_rows(rows, names, partials, steps):
    expected = set(
        itertools.product(
            names, ("raw", "ema"), ("train", "probe"), steps, partials, (0.1, 0.5), range(57)
        )
    )
    keys = [
        (
            r["method"],
            r["mode"],
            r["split"],
            r["denoise_steps"],
            r["partial"],
            r["strength"],
            r["pair"],
        )
        for r in rows
    ]
    assert len(keys) == len(expected) and set(keys) == expected
    assert all(r["protected_max_error"] in (None, 0.0) for r in rows)
    assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, float))


def evaluate(root, step, *, reference=False):
    verify(root)
    plan = read(root / "plan.json")
    assert step in plan["milestones"]
    if reference:
        names = [plan["reference"]["method"]]
        out = root / "evaluation/reference-1200"
    else:
        names = plan["arms"]
        out = root / f"evaluation/step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    forward = nnx.jit(
        lambda m, x, t, g, n: sample_prepared(m, x, t, steps=n, guides=g), static_argnums=4
    )
    rows, artifacts = [], {}
    for name in names:
        for mode in ("raw", "ema"):
            checkpoint = (
                Path(plan["reference"]["checkpoints"][mode]["path"])
                if reference
                else root / f"snapshots/step-{step}/{name}/{mode}"
            )
            files = {
                n: sha(checkpoint / n) for n in ("weights.npz", "config.json", "metadata.json")
            }
            if reference:
                assert files == plan["reference"]["checkpoints"][mode]["files"]
            model, _ = load_inference(checkpoint)
            for split in ("train", "probe"):
                data, records = np.load(root / f"{split}.npz"), read(root / f"{split}.json")
                for steps in plan["evaluation_steps"]:
                    for partial in (False, True) if reference or step == 1200 else (False,):
                        key = f"{name}-{mode}-{split}-n{steps}-{'partial' if partial else 'whole'}"
                        json_path, arrays_path = out / f"{key}.json", out / f"{key}.npz"
                        binding = dict(
                            checkpoint=str(checkpoint),
                            checkpoint_files=files,
                            source_sha256=sha(root / "source.json"),
                            script_sha256=sha(__file__),
                            plan_sha256=sha(root / "plan.json"),
                        )
                        if json_path.exists():
                            saved = read(json_path)
                            assert saved["binding"] == binding and saved["arrays_sha256"] == sha(
                                arrays_path
                            )
                            assert len(saved["results"]) == 2 * len(records)
                            assert np.load(arrays_path)["images"].shape == (
                                2 * len(records),
                                512,
                                512,
                                1,
                            )
                        else:
                            images, results = [], []
                            for strength in (0.1, 0.5):
                                for i, record in enumerate(records):
                                    clean, source = (
                                        data["clean"][i : i + 1],
                                        data["source"][i : i + 1],
                                    )
                                    x, t, g, target, level = fixed_inputs(
                                        clean,
                                        source,
                                        strength,
                                        SEED + 10000 + i + (1000 if split == "probe" else 0),
                                        partial=partial,
                                    )
                                    image = np.asarray(forward(model, x, t, g, steps))[0]
                                    assert np.isfinite(image).all()
                                    metrics = halo_metrics(image, target[0], level[0])
                                    assert metrics["protected_max_error"] in (None, 0.0)
                                    metrics["active_mse"] = float(
                                        ((image - target[0]) ** 2)[level[0] > 0].mean()
                                    )
                                    results.append(
                                        dict(
                                            method=name,
                                            mode=mode,
                                            split=split,
                                            denoise_steps=steps,
                                            partial=partial,
                                            strength=strength,
                                            **record,
                                            **metrics,
                                        )
                                    )
                                    images.append(image)
                            temporary = arrays_path.with_suffix(".npz.partial")
                            with temporary.open("wb") as stream:
                                np.savez_compressed(stream, images=np.stack(images))
                            temporary.replace(arrays_path)
                            saved = dict(
                                binding=binding, results=results, arrays_sha256=sha(arrays_path)
                            )
                            write(json_path, saved)
                        rows.extend(saved["results"])
                        artifacts[str(json_path)] = sha(json_path)
                        artifacts[str(arrays_path)] = saved["arrays_sha256"]
                        print(
                            dict(block=key, reference=reference, step=step, complete=True),
                            flush=True,
                        )
    check_rows(
        rows,
        names,
        (False, True) if reference or step == 1200 else (False,),
        plan["evaluation_steps"],
    )
    write(
        out / "metrics.json",
        dict(
            step=1200 if reference else step,
            reference=reference,
            results=rows,
            predictions=len(rows),
            artifacts=artifacts,
            source_sha256=sha(root / "source.json"),
        ),
    )
    summaries = []
    for name in names:
        for mode in ("raw", "ema"):
            for split in ("train", "probe"):
                for steps in plan["evaluation_steps"]:
                    for family in ("clean", "tone", "blur"):
                        for partial in (False, True) if reference or step == 1200 else (False,):
                            for strength in (None, 0.1, 0.5):
                                chosen = [
                                    r
                                    for r in rows
                                    if r["method"] == name
                                    and r["mode"] == mode
                                    and r["split"] == split
                                    and r["denoise_steps"] == steps
                                    and r["family"] == family
                                    and r["partial"] == partial
                                    and (strength is None or r["strength"] == strength)
                                    and r["kind"] != "blank"
                                ]
                                keys = (
                                    "ink_f1_alpha50",
                                    "ink_deficit_fraction",
                                    "halo_opacity_excess",
                                    "active_mse",
                                )
                                means = {
                                    k: float(np.mean([r[k] for r in chosen if r[k] is not None]))
                                    for k in keys
                                }
                                failed = [
                                    dict(pair=r["pair"], strength=r["strength"])
                                    for r in chosen
                                    if r["ink_f1_alpha50"] < 0.95
                                    or r["ink_deficit_fraction"] > 0.05
                                    or (r["halo_opacity_excess"] or 0) > 0.02
                                ]
                                summaries.append(
                                    dict(
                                        method=name,
                                        mode=mode,
                                        split=split,
                                        denoise_steps=steps,
                                        family=family,
                                        partial=partial,
                                        strength=strength,
                                        means=means,
                                        conditions=len(chosen),
                                        failed_fit=failed,
                                    )
                                )
    write(out / "summary.json", summaries)
    if reference:
        parent = Path(plan["parent"])
        checks = []
        for mode in ("raw", "ema"):
            for split in ("train", "probe"):
                actual = np.load(out / f"warm_detail16-{mode}-{split}-n1-whole.npz")["images"]
                old_path = parent / f"evaluation/step-1200/warm_detail16-{mode}-{split}.npz"
                np.testing.assert_array_equal(actual, np.load(old_path)["images"])
                checks.append(
                    dict(mode=mode, split=split, exact=True, historical_sha256=sha(old_path))
                )
        write(out / "historical-one-step-audit.json", dict(checks=checks, all_exact=True))


def compare(root):
    verify(root)
    original = root / "evaluation/reference-1200"
    final = root / "evaluation/step-1200"
    assert read(original / "historical-one-step-audit.json")["all_exact"]
    assert read(root / "baseline-replay-audit.json")["full_state_exact"]
    assert read(root / "training/restore-audit.json")["step"] == 1200
    assert read(root / "training/restore-audit.json")["all_state_exact"]
    for folder, names in (
        (original, ["warm_detail16"]),
        (final, ["progress_rough", "progress_bridge"]),
    ):
        report = read(folder / "metrics.json")
        check_rows(report["results"], names, (False, True), (1, 2, 5))
        for path, digest in report["artifacts"].items():
            assert sha(path) == digest, path
    summary = read(final / "summary.json") + read(original / "summary.json")

    def row(name, split, family, steps):
        return next(
            r
            for r in summary
            if r["method"] == name
            and r["mode"] == "raw"
            and r["split"] == split
            and r["family"] == family
            and r["denoise_steps"] == steps
            and not r["partial"]
            and r["strength"] is None
        )

    deltas, failures = [], []
    for split in ("train", "probe"):
        for family in ("clean", "tone", "blur"):
            for old_name, old_steps, steps in (
                ("progress_rough", 5, 5),
                ("warm_detail16", 1, 5),
                ("progress_rough", 1, 1),
            ):
                before, after = (
                    row(old_name, split, family, old_steps),
                    row("progress_bridge", split, family, steps),
                )
                delta = {k: after["means"][k] - before["means"][k] for k in before["means"]}
                item = dict(
                    split=split,
                    family=family,
                    reference=old_name,
                    reference_steps=old_steps,
                    steps=steps,
                    delta=delta,
                    absolute_fit_failures=len(after["failed_fit"]),
                )
                deltas.append(item)
                reasons = []
                if steps == 5 and family == "blur" and delta["ink_f1_alpha50"] < 0.03:
                    reasons.append("blur F1 gain below .03")
                if steps == 5 and delta["ink_deficit_fraction"] > 0.01:
                    reasons.append("ink deficit increases over .01")
                if family == "clean" and delta["ink_f1_alpha50"] < -0.01:
                    reasons.append("clean F1 drops over .01")
                if delta["halo_opacity_excess"] > 0.01:
                    reasons.append("halo increases over .01")
                if reasons:
                    failures.append(dict(item, reasons=reasons))
    write(
        root / "matched-results.json",
        dict(
            numerical_probe_pass=not failures,
            failures=failures,
            comparisons=deltas,
            primary="raw",
            primary_steps=5,
            adoption=False,
            visual_review_required=True,
            source_sha256=sha(root / "source.json"),
            reference_metrics_sha256=sha(original / "metrics.json"),
            candidate_metrics_sha256=sha(final / "metrics.json"),
            reference_summary_sha256=sha(original / "summary.json"),
            candidate_summary_sha256=sha(final / "summary.json"),
        ),
    )
    print(dict(numerical_probe_pass=not failures, failed_comparisons=len(failures)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["evaluate", "compare"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--step", type=int, default=1200)
    p.add_argument("--reference", action="store_true")
    a = p.parse_args()
    if a.action == "evaluate":
        evaluate(a.root.resolve(), a.step, reference=a.reference)
    else:
        compare(a.root.resolve())
