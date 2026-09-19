"""Resumable execution of all remaining planned experiments; final test is sealed."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from campaign_protocol import (
    needs_continuation,
    needs_receptive_field,
    read,
    report,
    screen,
    sha,
    write,
)

from lineartgen.config import ExperimentConfig, ModelConfig, load_config, save_config


class Campaign:
    def __init__(self, root, lock):
        self.root = root.resolve()
        self.lock = lock
        self.base = load_config(root / "config.json")
        self.protocol = read(root / "protocol.json")
        self.cases = self.root / "development"
        self.controller_sha256 = sha(__file__)

    def status(self, stage, **extra):
        row = dict(
            stage=stage,
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            pid=os.getpid(),
            controller_sha256=self.controller_sha256,
            **extra,
        )
        write(self.root / "status.json", row)
        with (self.root / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def check_source(self):
        for path, digest in read(self.root / "training-source.json")["file_sha256"].items():
            assert sha(path) == digest, f"Frozen numerical/protocol source changed: {path}"
        if (self.root / "stop").exists():
            self.status("stop_requested")
            raise InterruptedError("Campaign stop file exists")

    def run(self, script, args, label, *, cpu=False):
        self.check_source()
        self.status(label, script=script, script_sha256=sha(f"scripts/{script}.py"))
        env = dict(os.environ)
        if cpu:
            env.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1")
        with (self.root / "logs" / f"{label}.log").open("a") as log:
            subprocess.run(
                [sys.executable, "-u", f"scripts/{script}.py", *map(str, args)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                pass_fds=(self.lock.fileno(),),
            )
        self.check_source()

    def arm(self, name, model, seed):
        path = self.root / "arms" / name
        config = self.base.model_copy(
            update={
                "model": model,
                "train": self.base.train.model_copy(
                    update={"seed": seed["data"], "model_seed": seed["model"]}
                ),
            }
        )
        # Revalidate copied immutable models, including all new fields.
        config = ExperimentConfig.model_validate(config.model_dump())
        if path.exists():
            assert load_config(path / "config.json") == config
            assert read(path / "variants.json") == {"model": config.model_dump(mode="json")}
        else:
            path.mkdir(parents=True)
            save_config(config, path / "config.json")
            write(path / "variants.json", {"model": config.model_dump(mode="json")})
        return name

    def evaluate(self, name, step, *, final=False):
        arm = self.root / "arms" / name
        cases = self.root / "final" if final else self.cases
        scope = "final-evaluation" if final else "evaluation"
        for mode in ("raw", "ema"):
            output = arm / scope / f"step-{step}-{mode}"
            if not (output / "metrics.json").exists():
                self.run(
                    "evaluate_training_methods",
                    [
                        "--config",
                        arm / "config.json",
                        "--variants",
                        arm / "variants.json",
                        "--cases",
                        cases,
                        "--run-group",
                        arm / "training",
                        "--step",
                        step,
                        "--steps",
                        1,
                        "--recursion",
                        "runtime",
                        "--noise-generator",
                        "numpy",
                        "--output",
                        output,
                        *(["--raw"] if mode == "raw" else []),
                    ],
                    f"{name}-{scope}-{step}-{mode}",
                )
            report(arm, step, mode, cases, scope=scope)

    def train_to(self, name, step):
        arm = self.root / "arms" / name
        progress = arm / "training/progress.json"
        completed = read(progress)["step"] if progress.exists() else 0
        if completed < step:
            self.recover_logs(arm, completed)
            self.run(
                "compare_nested_training",
                [
                    "--config",
                    arm / "config.json",
                    "--variants",
                    arm / "variants.json",
                    "--run-dir",
                    arm / "training",
                    "--stop-after",
                    step - completed,
                    "--stop-file",
                    self.root / "stop",
                    *(["--resume"] if completed else []),
                ],
                f"{name}-train-{step}",
            )
            assert read(progress) == dict(step=step, runs=["model"])
        self.evaluate(name, step)

    def recover_logs(self, arm, completed):
        """Roll back only uncommitted log tails; preserve their original bytes."""
        training = arm / "training"
        if not training.exists():
            return
        if completed == 0:
            assert not list(training.glob("model/checkpoints/[0-9]*")), (
                "Checkpoint without group position"
            )
            backup = arm / f"initialization-recovery-{time.time_ns()}"
            training.rename(backup)
            self.status("initialization_recovered", arm=arm.name, backup=str(backup))
            return
        for path in (training / "timings.jsonl", training / "model/metrics.jsonl"):
            if not path.exists():
                continue
            lines = path.read_text().splitlines(keepends=True)
            accepted = []
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if row["step"] > completed:
                    break
                accepted.append(line)
            if len(accepted) != len(lines):
                backup = path.with_name(path.name + f".uncommitted-{time.time_ns()}")
                path.rename(backup)
                path.write_text("".join(accepted))
                self.status(
                    "log_tail_recovered",
                    arm=arm.name,
                    completed=completed,
                    backup=str(backup),
                    removed_rows=len(lines) - len(accepted),
                )

    def initial_core(self, shared, core):
        paths = [
            self.root / "arms" / name / "training/initial/model/weights.npz"
            for name in (shared, core)
        ]
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            for key in b.files:
                np.testing.assert_array_equal(a[key], b[key])
        write(
            self.root / "initialization" / f"{core}.json",
            dict(shared=shared, core=core, exact=True, files={str(p): sha(p) for p in paths}),
        )

    def execute(self, *, architecture_only=False):
        self.check_source()
        seeds = self.protocol["seeds"]
        first = seeds[0]
        original = self.base.model.model_copy(
            update={"skip_mode": "post_pool", "bottleneck_dilation": 1}
        )
        improved = original.model_copy(update={"skip_mode": "pre_pool"})
        control = self.arm("seed42_post_pool_d1", original, first)
        pre = self.arm("seed42_pre_pool_d1", improved, first)
        for step in self.protocol["architecture_steps"]:
            for name in (control, pre):
                self.train_to(name, step)
        a2 = screen(self.root, pre, control, self.protocol)
        write(self.root / "decisions/a2.json", a2)
        selected = pre if a2["adopt"] else control
        selected_model = improved if a2["adopt"] else original
        trigger = needs_receptive_field(self.root, selected, self.protocol)
        write(self.root / "decisions/a3-trigger.json", trigger)
        if trigger["required"]:
            candidate_model = selected_model.model_copy(update={"bottleneck_dilation": 2})
            candidate = self.arm(f"seed42_{selected_model.skip_mode}_d2", candidate_model, first)
            for step in self.protocol["architecture_steps"]:
                self.train_to(candidate, step)
            a3 = screen(self.root, candidate, selected, self.protocol)
            write(self.root / "decisions/a3.json", a3)
            if a3["adopt"]:
                selected, selected_model = candidate, candidate_model
        write(
            self.root / "decisions/architecture.json",
            dict(arm=selected, model=selected_model.model_dump(mode="json")),
        )
        if architecture_only:
            self.status("architecture_complete")
            return
        cohorts = {}
        for seed in seeds:
            prefix = f"seed{seed['model']}"
            shared = (
                selected if seed == first else self.arm(f"{prefix}_shared", selected_model, seed)
            )
            core = self.arm(
                f"{prefix}_core", selected_model.model_copy(update={"nested_channels": ()}), seed
            )
            large = self.arm(
                f"{prefix}_baseline",
                ModelConfig(
                    channels=(4, 10, 20, 40),
                    convolutions_per_block=3,
                    output_mode="residual",
                    skip_mode=selected_model.skip_mode,
                    bottleneck_dilation=selected_model.bottleneck_dilation,
                ),
                seed,
            )
            cohorts[str(seed["model"])] = dict(shared=shared, core=core, baseline=large)
        write(self.root / "cohorts.json", cohorts)
        first_arms = list(cohorts[str(first["model"])].values())
        for step in (2000, 10000, 20000, 30000):
            for name in first_arms:
                self.train_to(name, step)
        self.initial_core(*first_arms[:2])
        final_step = 30000
        for before, after, target in ((20000, 30000, 60000), (30000, 60000, 100000)):
            if final_step != after:
                break
            decision = needs_continuation(self.root, first_arms, before, after, self.protocol)
            write(self.root / f"decisions/continue-{after}.json", decision)
            if not decision["extend"]:
                break
            for name in first_arms:
                self.train_to(name, target)
            final_step = target
        write(self.root / "decisions/budget.json", dict(step=final_step, source_seed=first))
        for seed in seeds[1:]:
            arms = list(cohorts[str(seed["model"])].values())
            for step in self.protocol["long_steps"]:
                if step > final_step:
                    break
                for name in arms:
                    self.train_to(name, step)
            self.initial_core(*arms[:2])
        # These stages operate on an already fixed model/step cohort and never tune on final data.
        self.run(
            "analyze_completion_campaign",
            ["--root", self.root, "--stage", "development"],
            "development-decision",
            cpu=True,
        )
        self.run("prepare_campaign_final", ["--root", self.root], "prepare-final", cpu=True)
        for group in cohorts.values():
            for name in group.values():
                self.evaluate(name, final_step, final=True)
        self.run(
            "analyze_completion_campaign",
            ["--root", self.root, "--stage", "final"],
            "final-analysis",
            cpu=True,
        )
        self.run("verify_completion_campaign", ["--root", self.root], "model-verification")
        self.run(
            "verify_completion_campaign",
            ["--root", self.root, "--export"],
            "runtime-verification",
            cpu=True,
        )
        self.run("render_campaign_blind", ["--root", self.root], "blind-panels")
        self.status(
            "awaiting_blind_review",
            step=final_step,
            note="Numerical work complete; visual review, report and integrity audit remain.",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--architecture-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "logs").mkdir(exist_ok=True)
        for source, destination in (
            (Path("configs/completion-campaign.toml"), root / "base.toml"),
            (Path("configs/completion-protocol.json"), root / "protocol.json"),
        ):
            if destination.exists():
                assert sha(source) == sha(destination)
            else:
                shutil.copyfile(source, destination)
        config = load_config(root / "base.toml")
        if (root / "config.json").exists():
            assert load_config(root / "config.json") == config
        else:
            save_config(config, root / "config.json")
        campaign = Campaign(root, lock)
        try:
            campaign.execute(architecture_only=args.architecture_only)
        except InterruptedError:
            return
        except Exception as exc:
            campaign.status("failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
