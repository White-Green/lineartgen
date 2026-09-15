"""Freeze and verify reuse of the two previously completed boundary-loss arms."""

import argparse
import ast
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
from summarize_training_methods import sha

from lineartgen.config import load_config


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_timings(actual, previous, through):
    a = [
        read_line
        for line in actual.read_text().splitlines()
        if (read_line := json.loads(line))["step"] <= through
    ]
    b = [json.loads(line) for line in previous.read_text().splitlines()][:through]
    assert [r["step"] for r in a] == list(range(1, through + 1))
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        for key in ("step", "scale_level", "recursion_levels"):
            assert left[key] == right[key], (left["step"], key)
        assert left["generated"]["half"] == right["generated"]["boundary"]
        assert right["generated"]["boundary"] == right["generated"]["control"]
    return dict(updates=through, scale_recursion_generated_keys_exact=True)


class WithoutHalfChoice(ast.NodeTransformer):
    def visit_List(self, node):
        self.generic_visit(node)
        node.elts = [
            x for x in node.elts if not (isinstance(x, ast.Constant) and x.value == "boundary_half")
        ]
        return node

    visit_Tuple = visit_List


def prepare(root, reference, benchmark):
    assert not (root / "training").exists(), "Freeze reference before half training"
    root.mkdir(parents=True, exist_ok=True)
    config, prior = load_config(root / "config.toml"), load_config(reference / "config.toml")
    assert config.loss.boundary_weight == prior.loss.boundary_weight / 2 == 0.2603685
    assert config.model_copy(update={"loss": prior.loss}) == prior
    assert (
        config.loss.model_copy(update={"boundary_weight": prior.loss.boundary_weight}) == prior.loss
    )
    calibration = read(reference / "calibration.json")
    write(
        root / "calibration.json",
        dict(
            coefficient=config.loss.boundary_weight,
            coefficient_frozen_before_training=True,
            selection="Exactly half the previously calibrated coefficient; no new calibration",
            reference_coefficient=calibration["coefficient"],
            reference_calibration=str(reference / "calibration.json"),
            reference_calibration_sha256=sha(reference / "calibration.json"),
            reference_weights_sha256=calibration["reference_weights_sha256"],
            selection_context="Follow-up selected after previous development results; "
            "not an unseen test",
        ),
    )
    artifacts = read(benchmark)["artifacts"]
    for path, digest in artifacts.items():
        assert sha(Path(path)) == digest, path
    for path in (
        benchmark,
        reference / "training/timings.jsonl",
        reference / "training/dataset.json",
        reference / "training/boundary/metrics.jsonl",
        reference / "final-evidence-source.json",
    ):
        artifacts[str(path)] = sha(path)
    source = read(reference / "training-source.json")
    unchanged = {
        path: digest
        for path, digest in source["file_sha256"].items()
        if (path.startswith("src/") and path != "src/lineartgen/training/comparison.py")
        or path in ("uv.lock", "pyproject.toml")
    }
    for path, digest in unchanged.items():
        assert sha(Path(path)) == digest, path
    # The execution/evaluation loops must differ only by an additional CLI choice.
    archive = Path(source["archive"])
    assert sha(archive) == source["archive_sha256"]
    ast_exact = []
    with tarfile.open(archive) as saved:
        for name in ("scripts/compare_nested_training.py", "scripts/evaluate_training_methods.py"):
            old = ast.parse(saved.extractfile(name).read())
            new = WithoutHalfChoice().visit(ast.parse(Path(name).read_text()))
            assert ast.dump(old) == ast.dump(new), name
            ast_exact.append(name)
    replay = root / "replay-full"
    assert load_config(replay / "half/config.json") == prior
    assert read(replay / "dataset.json") == read(reference / "training/dataset.json")
    assert read(replay / "initialization.json") == read(reference / "training/initialization.json")
    timing = check_timings(replay / "timings.jsonl", reference / "training/timings.jsonl", 30)

    def logs(path):
        return {
            row["step"]: {k: v for k, v in row.items() if k != "update_seconds"}
            for line in path.read_text().splitlines()
            if (row := json.loads(line))["phase"] == "train" and row["step"] <= 30
        }

    actual = logs(replay / "half/metrics.jsonl")
    previous = logs(reference / "training/boundary/metrics.jsonl")
    assert set(actual) == set(previous) == {10, 20, 30}
    paired = root / "replay-paired"
    paired_logs = logs(paired / "boundary/metrics.jsonl")

    def compare_logs(left, right):
        errors = []
        for step in (10, 20, 30):
            assert set(left[step]) == set(right[step])
            for key, value in left[step].items():
                other = right[step][key]
                if key in (
                    "step",
                    "phase",
                    "learning_rate",
                    "scale_level",
                    "recursion_levels",
                    "generated_examples",
                ):
                    assert value == other, (step, key)
                    continue
                rtol = 1e-4 if key == "gradient_norm" else 1e-5
                np.testing.assert_allclose(
                    value, other, atol=1e-8, rtol=rtol, err_msg=f"{step}/{key}"
                )
                errors.append(
                    dict(
                        step=step,
                        metric=key,
                        absolute=abs(value - other),
                        relative=abs(value - other) / max(abs(other), 1e-30),
                    )
                )
        return dict(
            bit_exact=left == right,
            max_absolute=max(e["absolute"] for e in errors),
            max_relative=max(e["relative"] for e in errors),
            errors=errors,
        )

    comparisons = {
        "single_vs_previous": compare_logs(actual, previous),
        "paired_vs_previous": compare_logs(paired_logs, previous),
        "single_vs_paired": compare_logs(actual, paired_logs),
    }
    assert read(paired / "dataset.json") == read(replay / "dataset.json")
    assert read(paired / "initialization.json") == read(replay / "initialization.json")
    for line, old in zip(
        (paired / "timings.jsonl").read_text().splitlines(),
        (reference / "training/timings.jsonl").read_text().splitlines()[:30],
        strict=True,
    ):
        a, b = json.loads(line), json.loads(old)
        for key in ("step", "scale_level", "recursion_levels", "generated"):
            assert a[key] == b[key]
    for path in (
        paired / "boundary/metrics.jsonl",
        paired / "timings.jsonl",
        paired / "dataset.json",
        paired / "initialization.json",
    ):
        artifacts[str(path)] = sha(path)
    for path in (
        replay / "half/metrics.jsonl",
        replay / "timings.jsonl",
        replay / "dataset.json",
        replay / "initialization.json",
    ):
        artifacts[str(path)] = sha(path)
    initial = Path(read(reference / "training/initialization.json")["source"])
    write(
        root / "reference-model-sha256.json",
        {str(p): sha(p) for p in initial.iterdir() if p.is_file()},
    )
    cases = root / "evaluation"
    cases.mkdir(exist_ok=True)
    images = read(reference / "evaluation-input-sha256.json")
    assert len(images) == 1563
    for name, digest in images.items():
        assert sha(reference / "evaluation" / name) == digest
        if not (cases / name).exists():
            shutil.copyfile(reference / "evaluation" / name, cases / name)
        assert sha(cases / name) == digest
    write(root / "evaluation-input-sha256.json", images)
    record = dict(
        reference_root=str(reference),
        immutable_artifacts=artifacts,
        unchanged_numerical_source=unchanged,
        execution_ast_equal_except_cli=ast_exact,
        replay=dict(
            **timing,
            logged_steps=[10, 20, 30],
            metrics_comparisons=comparisons,
            tolerances=dict(loss_rtol=1e-5, gradient_norm_rtol=1e-4, atol=1e-8),
            dataset_and_initialization_exact=True,
        ),
        limitation="Same short replay protocol with bounded GPU numerical differences; "
        "neither bitwise full-training reproducibility nor a second seed",
    )
    write(root / "reference-reuse.json", record)
    verify_reuse(root)
    print(
        json.dumps(
            {
                name: {k: v for k, v in value.items() if k != "errors"}
                for name, value in comparisons.items()
            }
        )
    )


def verify_reuse(root, through=None):
    record = read(root / "reference-reuse.json")
    for path, digest in {
        **record["immutable_artifacts"],
        **record["unchanged_numerical_source"],
    }.items():
        assert sha(Path(path)) == digest, path
    reference = Path(record["reference_root"])
    config, prior = load_config(root / "config.toml"), load_config(reference / "config.toml")
    assert config.loss.boundary_weight == prior.loss.boundary_weight / 2
    assert config.model_copy(update={"loss": prior.loss}) == prior
    assert (
        config.loss.model_copy(update={"boundary_weight": prior.loss.boundary_weight}) == prior.loss
    )
    if through is not None:
        assert read(root / "training/dataset.json") == read(reference / "training/dataset.json")
        assert read(root / "training/initialization.json") == read(
            reference / "training/initialization.json"
        )
        check_timings(
            root / "training/timings.jsonl", reference / "training/timings.jsonl", through
        )
    return reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-artifacts", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.reference_root.resolve(), args.reference_artifacts.resolve())
