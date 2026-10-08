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
from flax import nnx
from PIL import Image
from rich.console import Console
from tensorboardX import SummaryWriter

from lineartgen.config import load_config, save_config
from lineartgen.data.geometry import mix_geometry
from lineartgen.data.pipeline import batches, dataset_description, discover, split_experiment
from lineartgen.diffusion import sample, training_inputs
from lineartgen.models import create_model

from . import checkpoints
from .steps import accumulate_gradients, create_optimizer, make_steps


class Trainer:
    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = Path(run_dir).resolve()
        self.console = Console()

    def fit(self, *, resume=False, stop_after=None):
        config, train = self.config, self.config.train
        if config.matching.method == "exact" and config.matching.solver == "scipy":
            try:
                jax.local_devices(backend="cpu")
            except RuntimeError as error:
                raise RuntimeError(
                    "SciPy assignment needs a CPU backend for JAX callbacks. "
                    "Use JAX_PLATFORMS=rocm,cpu (or cuda,cpu), or matching.solver='optax'."
                ) from error
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
        training, validation, final_test = split_experiment(sources, config.data, train.seed)
        description = dataset_description(sources, training, validation, final_test)
        split_path = self.run_dir / "dataset.json"
        if resume and json.loads(split_path.read_text()) != description:
            raise ValueError("Dataset or split changed since this run was created")
        split_path.write_text(json.dumps(description, indent=2) + "\n")
        model = create_model(config.model, rngs=nnx.Rngs(train.initialization_seed))
        optimizer, schedule = create_optimizer(model, config)
        ema = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
        gradient_step, update_step, evaluate_step = make_steps(config)
        # Assign a full candidate pool before slicing microbatches. Accumulation
        # must not silently shrink the assignment problem to the microbatch.
        prepare_step = jax.jit(
            lambda clean, key, levels: training_inputs(
                clean,
                key,
                config.diffusion,
                config.matching,
                levels=levels,
                sketch=config.sketch,
                with_source=config.model.source_conditioning,
            ),
            static_argnums=(2,),
        )
        sample_step = nnx.jit(
            lambda model, clean, key, levels: sample(
                model,
                clean,
                key,
                strength=0.5,
                steps=config.diffusion.sample_steps,
                levels=levels,
            ),
            static_argnums=(3,),
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
                batch_size=train.effective_batch_size,
                count=end - completed,
                seed=train.seed,
                start=completed,
            )
            self.console.print(f"Training on {jax.devices()}; optimizer steps {completed} → {end}")
            with SummaryWriter(
                str(self.run_dir / "events"), purge_step=completed if resume else None
            ) as writer:
                try:
                    for step in range(completed, end):
                        batch = next(stream)
                        images, generated = mix_geometry(
                            batch.images,
                            config.geometry,
                            seed=train.seed,
                            step=step,
                            image_scale=config.data.image_scale,
                        )
                        clean = jax.device_put(images)
                        levels = config.diffusion.levels_for_scale(
                            config.data.crop_size,
                            config.model.minimum_size,
                            batch.scale_level,
                        )
                        prepared = prepare_step(clean, jax.random.fold_in(root_key, step), levels)
                        metrics, gradients, norm = accumulate_gradients(
                            gradient_step, model, prepared, train
                        )
                        norm = float(norm)
                        if not math.isfinite(norm) or not math.isfinite(float(metrics["loss"])):
                            raise FloatingPointError(
                                f"Nonfinite loss/gradients before update {step + 1}"
                            )
                        _, ema = update_step(model, optimizer, gradients, ema)
                        completed = step + 1
                        if completed % train.log_every == 0 or completed == end:
                            logged = {name: float(value) for name, value in metrics.items()}
                            logged.update(
                                gradient_norm=norm,
                                learning_rate=float(schedule(step)),
                                scale_level=batch.scale_level,
                                recursion_levels=levels,
                                generated_examples=len(generated),
                            )
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
                                prepare_step,
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

    def _evaluate(self, writer, step, model, validation, evaluate_step, sample_step, prepare_step):
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
        first_levels = None
        for index, batch in enumerate(stream):
            clean = jax.device_put(batch.images)
            levels = config.diffusion.levels_for_scale(
                config.data.crop_size,
                config.model.minimum_size,
                batch.scale_level,
            )
            if first is None:
                first = clean[:1]
                first_levels = levels
            prepared = prepare_step(
                clean,
                jax.random.fold_in(jax.random.PRNGKey(train.seed + 2), index),
                levels,
            )
            parts = evaluate_step(model, prepared)
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value) / train.validation_batches
        self._log(writer, step, "validation", totals)
        # After one denoising iteration, widths have different noisy trajectories.
        # Evaluate complete autonomous samplers, not large-model states fed to small heads.
        count = getattr(model, "num_subnets", 1)
        generated = [
            sample_step(
                model.extract(i) if count > 1 else model,
                first,
                jax.random.PRNGKey(train.seed + 3),
                first_levels,
            )
            for i in range(count)
        ]
        panel = np.concatenate(
            [np.asarray(first[0]), *(np.asarray(value[0]) for value in generated)], axis=1
        )
        panel = np.rint(np.clip((panel[..., 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
        samples = self.run_dir / "samples"
        samples.mkdir(exist_ok=True)
        Image.fromarray(panel).save(samples / f"step-{step:08d}.png")
        writer.add_image("validation/target_and_sample", panel[None], step)
