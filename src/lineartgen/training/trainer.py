"""The experiment lifecycle: input -> accumulated gradients -> update -> evaluate/save.

An optimizer step consumes batch_size * accumulate_steps examples. Checkpoints
are written only at these boundaries, so resume also restores the exact data
and noise streams. --stop-after limits a run without changing its LR schedule.
"""

import json
import math
import time
from pathlib import Path

import jax
import numpy as np
import optax
from flax import nnx
from PIL import Image
from rich.console import Console
from tensorboardX import SummaryWriter

from lineartgen.config import load_config, save_config
from lineartgen.data.pipeline import batches, dataset_description, discover, split_sources
from lineartgen.diffusion import sample
from lineartgen.models import UNet

from . import checkpoints
from .steps import create_optimizer, make_steps


class Trainer:
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = Path(run_dir).resolve()
        self.console = Console()

    def fit(self, *, resume=False, stop_after=None):
        config, train = self.config, self.config.train
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not resume:
            raise FileExistsError(f"Run directory is not empty: {self.run_dir}; use --resume")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.run_dir / "config.json"
        if resume:
            if load_config(config_path) != config:
                raise ValueError(
                    "Resume requires the saved configuration; start a new run for changes"
                )
        else:
            save_config(config, config_path)
        self.console.print("Indexing source images…")
        sources = discover(config.data)
        training, validation = split_sources(sources, config.data.validation_count, train.seed)
        description = dataset_description(sources, training, validation)
        split_path = self.run_dir / "dataset.json"
        if resume and json.loads(split_path.read_text()) != description:
            raise ValueError("Dataset or split changed since this run was created")
        split_path.write_text(json.dumps(description, indent=2) + "\n")
        model = UNet(config.model, rngs=nnx.Rngs(train.seed))
        optimizer, schedule = create_optimizer(model, config)
        ema = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
        gradient_step, update_step, evaluate_step = make_steps(config)
        sample_step = nnx.jit(
            lambda model, clean, key: sample(
                model,
                clean,
                key,
                strength=0.5,
                steps=config.diffusion.sample_steps,
                levels=config.diffusion.recursion_levels,
            )
        )
        root_key = jax.random.PRNGKey(train.seed)
        started, completed = time.monotonic(), 0
        with checkpoints.manager(self.run_dir / "checkpoints", train.keep_checkpoints) as saved:
            if resume:
                completed, ema = checkpoints.restore(saved, model, optimizer, ema)
            end = train.steps if stop_after is None else min(train.steps, completed + stop_after)
            stream = batches(
                training,
                config.data,
                batch_size=train.batch_size,
                count=(end - completed) * train.accumulate_steps,
                seed=train.seed,
                start=completed * train.accumulate_steps,
            )
            self.console.print(f"Training on {jax.devices()}; optimizer steps {completed} → {end}")
            with SummaryWriter(
                str(self.run_dir / "events"), purge_step=completed if resume else None
            ) as writer:
                try:
                    for step in range(completed, end):
                        gradients, metrics = None, None
                        for micro in range(train.accumulate_steps):
                            clean = jax.device_put(next(stream))
                            key = jax.random.fold_in(
                                root_key, step * train.accumulate_steps + micro
                            )
                            parts, grads = gradient_step(model, clean, key)
                            if gradients is None:
                                gradients, metrics = grads, parts
                            else:
                                gradients = jax.tree.map(lambda a, b: a + b, gradients, grads)
                                metrics = jax.tree.map(lambda a, b: a + b, metrics, parts)
                        gradients = jax.tree.map(lambda x: x / train.accumulate_steps, gradients)
                        metrics = jax.tree.map(lambda x: x / train.accumulate_steps, metrics)
                        norm = float(optax.tree.norm(gradients))
                        if not math.isfinite(norm) or not math.isfinite(float(metrics["loss"])):
                            raise FloatingPointError(
                                f"Nonfinite loss/gradients before update {step + 1}"
                            )
                        _, ema = update_step(model, optimizer, gradients, ema)
                        completed = step + 1
                        if completed % train.log_every == 0 or completed == end:
                            logged = {name: float(value) for name, value in metrics.items()}
                            logged.update(gradient_norm=norm, learning_rate=float(schedule(step)))
                            self._log(writer, completed, "train", logged)
                        if completed % train.evaluate_every == 0 or completed == end:
                            eval_model = nnx.clone(model)
                            nnx.update(eval_model, ema)
                            self._evaluate(
                                writer,
                                completed,
                                eval_model,
                                validation,
                                evaluate_step,
                                sample_step,
                            )
                        if completed % train.checkpoint_every == 0 or completed == end:
                            checkpoints.save(saved, model, optimizer, ema, completed)
                except KeyboardInterrupt:
                    # Any partial gradient accumulation is discarded; completed is unchanged.
                    if completed and saved.latest_step() != completed:
                        checkpoints.save(saved, model, optimizer, ema, completed)
                    self.console.print(f"Interrupted; saved optimizer step {completed}")
                finally:
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()
        self.console.print(f"Finished at step {completed} ({time.monotonic() - started:.1f}s)")
        return completed

    def _log(self, writer, step, phase, metrics):
        record = {"step": step, "phase": phase, **metrics}
        with (self.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        for name, value in metrics.items():
            writer.add_scalar(f"{phase}/{name}", value, step)
        self.console.print(f"{phase} step={step} loss={metrics['loss']:.6f}")

    def _evaluate(self, writer, step, model, validation, evaluate_step, sample_step):
        train, config = self.config.train, self.config
        stream = batches(
            validation,
            config.data,
            batch_size=train.batch_size,
            count=train.validation_batches,
            seed=train.seed + 1,
            training=False,
        )
        totals = {}
        first = None
        for index, clean in enumerate(stream):
            clean = jax.device_put(clean)
            if first is None:
                first = clean[:1]
            parts = evaluate_step(
                model, clean, jax.random.fold_in(jax.random.PRNGKey(train.seed + 2), index)
            )
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value) / train.validation_batches
        self._log(writer, step, "validation", totals)
        generated = sample_step(model, first, jax.random.PRNGKey(train.seed + 3))
        panel = np.concatenate((np.asarray(first[0]), np.asarray(generated[0])), axis=1)
        panel = np.rint(np.clip((panel[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
        samples = self.run_dir / "samples"
        samples.mkdir(exist_ok=True)
        Image.fromarray(panel).save(samples / f"step-{step:08d}.png")
        writer.add_image("validation/target_and_sample", panel[None], step)
