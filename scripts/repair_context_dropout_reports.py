"""Correct report labels only; leave the frozen inference/training modules intact."""

import argparse
import importlib
import inspect
import time
from datetime import UTC, datetime
from pathlib import Path

from campaign_protocol import read, sha, write
from context_dropout_study import check

LABELS = {
    '"merge"': '"control"',
    '"mixed"': '"dropout"',
    '"merge_uint8"': '"control_uint8"',
    '"mixed_uint8"': '"dropout_uint8"',
    '"mixed_float"': '"dropout_float"',
    "EI group6 raw18000": "EQ mixed raw18000",
    "fresh width + merge": "control warm6000",
    "fresh merge + mixed opacity": "context dropout warm6000",
}


def run(root, phase):
    check(root)
    suffix = {"evaluation": "", "stress": "_rough", "scene512": "_scenes"}[phase]
    module = importlib.import_module("evaluate_context_dropout" + suffix)
    source = inspect.getsource(module.render)
    for old, new in LABELS.items():
        source = source.replace(old, new)
    # Only the rendering function is replaced; inference and its assertions remain frozen.
    exec(compile(source, __file__ + ":" + phase, "exec"), module.__dict__)
    if phase == "evaluation":
        assert read(root / "driver-evaluation.json")["returncode"] == 1
        before = read(root / "evaluation/prediction-binding.json")
        for f, h in before.items():
            assert sha(root / "evaluation" / f) == h
        assert len(read(root / "evaluation/metrics.json")["rows"]) == 4608
        module.render(root)
        for f, h in before.items():
            assert sha(root / "evaluation" / f) == h
        write(
            root / "diagnostics/report-repair.json",
            dict(
                original_evaluation_exit=1,
                numerical_predictions_complete=True,
                predictions_unchanged=True,
                prediction_files=len(before),
                adapter_sha256=sha(Path(__file__)),
            ),
        )
    elif phase == "stress":
        module.evaluate(root)
    else:
        module.ROOT = root
        module.OUT = root / "scene512"
        module.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("evaluation", "stress", "scene512"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tick = time.monotonic()
    started = datetime.now(UTC).isoformat()
    name = "evaluation-render" if args.phase == "evaluation" else args.phase
    code = 1
    try:
        run(root, args.phase)
        code = 0
    finally:
        write(
            root / f"driver-{name}.json",
            dict(
                phase=name,
                started_at=started,
                finished_at=datetime.now(UTC).isoformat(),
                seconds=time.monotonic() - tick,
                returncode=code,
                adapter_sha256=sha(Path(__file__)),
            ),
        )
