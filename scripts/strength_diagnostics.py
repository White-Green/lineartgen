"""Freeze high-strength development cases and audit N1 sampling and evaluation."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from evaluate_training_methods import summarize
from run_sketch_rates import validate
from summarize_sketch_experiment import strength_slices
from summarize_sketch_rates import contrast, summaries
from summarize_training_methods import sha


def prepare(reference, output):
    if output.exists():
        raise FileExistsError(output)
    old = json.loads((reference / "cases.json").read_text())
    output.mkdir(parents=True)
    records = []
    for index, case in enumerate(old["cases"]):
        if case["strength"] != 0.5 or case["suite"] == "legacy":
            continue
        with np.load(reference / case["file"], allow_pickle=False) as arrays:
            values = {k: arrays[k] for k in arrays.files}
        values["strength"] = np.where(values["strength"] > 0, np.float32(0.9), np.float32(0))
        name = f"case-{len(records):03d}.npz"
        np.savez_compressed(output / name, **values)
        records.append(dict(case, strength=0.9, file=name, original_case=index))
    manifest = dict(
        old,
        cases=records,
        reference_manifest_sha256=sha(reference / "cases.json"),
        limitations="Same development materials, masks and seeds at strength .9; "
        "not an independent test.",
    )
    (output / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output.parent / "high-strength-input-sha256.json").write_text(
        json.dumps({p.name: sha(p) for p in sorted(output.iterdir())}, indent=2) + "\n"
    )
    return records


def audit_sampling(root, through):
    from lineartgen.config import load_config
    from lineartgen.training.comparison import comparison_configs

    config = load_config(root / "config.toml")
    rows = [json.loads(x) for x in (root / "training/timings.jsonl").read_text().splitlines()][
        :through
    ]
    assert [r["step"] for r in rows] == list(range(1, through + 1))
    key = jax.random.PRNGKey(config.train.seed)
    draws = np.asarray(
        jax.jit(
            jax.vmap(
                lambda i: jax.random.uniform(
                    jax.random.split(jax.random.fold_in(key, i))[0], (16, 10, 1, 1, 1)
                )
            )
        )(np.arange(1, through + 1, dtype=np.uint32))
    )
    u, identity = (
        draws[:, :, 0, 0, 0, 0],
        draws[:, :, 8, 0, 0, 0] < config.diffusion.identity_probability,
    )
    records = []
    for method, settings in comparison_configs(config, study="noise_strength").items():
        c = settings.diffusion
        # Independent NumPy inverse CDF, permitting only FP32 rounding differences.
        expected = c.min_noise + u * (1 - c.min_noise)
        if c.strength_sampling == "low_mixture":
            expected = np.where(
                u < c.low_strength_probability,
                c.min_noise + u / c.low_strength_probability * (c.low_strength_max - c.min_noise),
                c.low_strength_max
                + (u - c.low_strength_probability)
                / (1 - c.low_strength_probability)
                * (1 - c.low_strength_max),
            )
        expected = np.where(identity, 0, expected)
        actual = np.asarray([r["strengths"][method] for r in rows])
        np.testing.assert_array_equal(actual == 0, identity)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
        active = actual[actual > 0]
        records.append(
            dict(
                method=method,
                examples=int(actual.size),
                identity=int(identity.sum()),
                editable=int(active.size),
                low_fraction=float(np.mean(active <= 0.5)),
                high_fraction=float(np.mean(active > 0.9)),
                mean_strength=float(active.mean()),
                inverse_cdf_max_error=float(np.abs(actual - expected).max()),
            )
        )
    return dict(
        through=through,
        records=records,
        shared_identity_exact=True,
        note="Recorded strengths are per-image maxima before pyramid pooling; masks share keys.",
    )


def audit_high(root):
    inputs = root / "high-strength"
    hashes = json.loads((root / "high-strength-input-sha256.json").read_text())
    assert len(hashes) == 617
    for name, digest in hashes.items():
        assert sha(inputs / name) == digest, name
    manifest = json.loads((inputs / "cases.json").read_text())
    primary = json.loads((root / "evaluation/cases.json").read_text())
    assert manifest["reference_manifest_sha256"] == sha(root / "evaluation/cases.json")
    for case in manifest["cases"]:
        old = primary["cases"][case["original_case"]]
        assert case["seed"] == old["seed"] and case["strength"] == 0.9
        assert {
            k: v for k, v in case.items() if k not in ("original_case", "strength", "file")
        } == {k: v for k, v in old.items() if k not in ("strength", "file")}
        with (
            np.load(inputs / case["file"], allow_pickle=False) as actual,
            np.load(root / "evaluation" / old["file"], allow_pickle=False) as ref,
        ):
            assert set(actual.files) == set(ref.files)
            for k in actual.files:
                expected = (
                    np.where(ref[k] > 0, np.float32(0.9), np.float32(0))
                    if k == "strength"
                    else ref[k]
                )
                np.testing.assert_array_equal(actual[k], expected)
    reports = {}
    files = [root / "high-strength-input-sha256.json", inputs / "cases.json"]
    total = 0
    for step, mode in [(0, "raw"), (10000, "raw"), (10000, "ema")]:
        p = inputs / f"step-{step}-{mode}" / "metrics.json"
        r = json.loads(p.read_text())
        validate(
            r, ("initial",) if step == 0 else ("control", "low80"), step, "runtime", case_count=616
        )
        assert r["manifest_sha256"] == sha(inputs / "cases.json") and r["ema"] == (mode == "ema")
        assert r["summary"] == summarize(r["results"])
        for row in r["results"]:
            case = manifest["cases"][row["case"]]
            for field in (
                "strength",
                "partial",
                "suite",
                "kind",
                "family",
                "severity",
                "scale_level",
            ):
                assert row.get(field) == case.get(field)
        total += len(r["results"])
        files.append(p)
        reports[f"{step}-{mode}"] = r
    assert total == 9240
    return dict(
        case_count=616,
        outputs=total,
        materials_unchanged=True,
        comparison={
            mode: dict(
                summary=summaries(reports["10000-" + mode]),
                paired_minus_control=contrast(
                    reports["10000-" + mode], reports["10000-" + mode], manifest, "control"
                ),
                paired_minus_initial=contrast(
                    reports["10000-" + mode], reports["0-raw"], manifest, "initial"
                ),
                strength_slices=strength_slices(reports["10000-" + mode]),
            )
            for mode in ("raw", "ema")
        },
        artifacts={str(p): sha(p) for p in files},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-from", type=Path)
    parser.add_argument("--through", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_from:
        rows = prepare(args.prepare_from, args.root / "high-strength")
        assert len(rows) == 616
        print(f"Frozen {len(rows)} high-strength cases")
        return
    assert args.output is not None
    result = dict(sampling=audit_sampling(args.root, args.through))
    if args.through == 10000:
        result["high_strength"] = audit_high(args.root)
        manifest = json.loads((args.root / "evaluation/cases.json").read_text())
        result["primary_strength_slices"] = {}
        for mode in ("raw", "ema"):
            report = json.loads(
                (args.root / f"evaluation/step-10000-{mode}/metrics.json").read_text()
            )
            result["primary_strength_slices"][mode] = {}
            for strength in (0.1, 0.3, 0.5, 0.7):
                selected = dict(
                    report, results=[r for r in report["results"] if r["strength"] == strength]
                )
                result["primary_strength_slices"][mode][str(strength)] = dict(
                    summary=summaries(selected),
                    paired_minus_control=contrast(selected, selected, manifest, "control"),
                )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
