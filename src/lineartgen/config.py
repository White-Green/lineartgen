"""Typed, strict experiment settings. Paths are relative to the project directory."""

import json
import math
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelConfig(Settings):
    channels: tuple[int, ...] = (4, 10, 20, 40)
    convolutions_per_block: int = Field(default=3, ge=1)
    kernel_size: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_architecture(self):
        if not self.channels or min(self.channels) < 1:
            raise ValueError("channels must contain positive channel counts")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return self

    @property
    def minimum_size(self) -> int:
        return 2 ** len(self.channels)


class DataConfig(Settings):
    root: Path = Path("dataset")
    scale_dirs: tuple[str, ...] = (
        "images",
        "images_2x",
        "images_4x",
        "images_8x",
        "images_16x",
        "images_32x",
    )
    # As in Burn, directories are ordered from 1x upwards unless explicitly mapped.
    scale_levels: tuple[int, ...] | None = None
    crop_size: int = Field(default=512, ge=16)
    validation_count: int = Field(default=64, ge=1)
    workers: int = Field(default=4, ge=0)
    prefetch_batches: int = Field(default=2, ge=1)
    augment: bool = True
    image_scale: float = Field(default=1.1, gt=0)
    alpha_mode: Literal["ignore", "white"] = "ignore"
    prepared_root: Path | None = None

    @model_validator(mode="after")
    def validate_scales(self):
        if not self.scale_dirs or len(set(self.scale_dirs)) != len(self.scale_dirs):
            raise ValueError("scale_dirs must be nonempty and unique")
        if self.scale_levels is not None and (
            len(self.scale_levels) != len(self.scale_dirs)
            or min(self.scale_levels) < 0
            or len(set(self.scale_levels)) != len(self.scale_levels)
        ):
            raise ValueError("scale_levels must give a unique nonnegative level per directory")
        return self

    @property
    def resolved_scale_levels(self):
        return (
            self.scale_levels
            if self.scale_levels is not None
            else tuple(range(len(self.scale_dirs)))
        )


class DiffusionConfig(Settings):
    original_steps: int = Field(default=1, ge=1)
    stop_size: int = Field(default=0, ge=0)
    min_noise: float = Field(default=0.001, gt=0, lt=1)
    identity_probability: float = Field(default=0.1, ge=0, le=1)
    pattern_weights: tuple[float, float, float, float] = (1, 1, 1, 1)
    scale_loss_multiplier: float = Field(default=0.0, ge=0)
    sample_steps: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_weights(self):
        if min(self.pattern_weights) < 0 or sum(self.pattern_weights) <= 0:
            raise ValueError("pattern_weights must be nonnegative with a positive sum")
        return self

    def levels_for_scale(self, crop_size, minimum_size, scale_level):
        """Burn's original_steps + scale_level rule, capped by model and stop sizes."""
        if scale_level < 0 or crop_size < minimum_size:
            raise ValueError("Invalid image size or dataset scale level")
        levels, size = 1, crop_size
        while levels < self.original_steps + scale_level:
            if (self.stop_size and size <= self.stop_size) or size // 2 < minimum_size:
                break
            size //= 2
            levels += 1
        return levels


class NoiseMatchingConfig(Settings):
    method: Literal["exact", "independent"] = "exact"
    solver: Literal["scipy", "optax"] = "scipy"
    pool_size: int = Field(default=256, ge=1)
    cost_resolution: int = Field(default=32, ge=1)


class LossConfig(Settings):
    balance_tones: bool = True
    identity_weight: float = Field(default=0.1, ge=0)
    edge_weight: float = Field(default=0.1, ge=0)
    contrast_weight: float = Field(default=0.01, ge=0)


class TrainConfig(Settings):
    steps: int = Field(default=100_000, ge=1)
    batch_size: int = Field(default=4, ge=1)
    accumulate_steps: int = Field(default=4, ge=1)
    # Base rate before sqrt(effective_batch_size); minimum remains absolute, as in Burn.
    learning_rate: float = Field(default=1e-4, gt=0)
    minimum_learning_rate: float = Field(default=1e-5, ge=0)
    warmup_steps: int = Field(default=500, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    ema_decay: float = Field(default=0.999, ge=0, lt=1)
    log_every: int = Field(default=10, ge=1)
    evaluate_every: int = Field(default=250, ge=1)
    checkpoint_every: int = Field(default=250, ge=1)
    keep_checkpoints: int = Field(default=3, ge=1)
    validation_batches: int = Field(default=4, ge=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @property
    def effective_batch_size(self):
        return self.batch_size * self.accumulate_steps

    @property
    def peak_learning_rate(self):
        return self.learning_rate * math.sqrt(self.effective_batch_size)


class ExperimentConfig(Settings):
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    matching: NoiseMatchingConfig = NoiseMatchingConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()

    @model_validator(mode="after")
    def validate_shapes_and_schedule(self):
        size = self.data.crop_size
        if size < self.model.minimum_size or size & (size - 1):
            raise ValueError("crop_size must be a power of two and at least the model minimum")
        if self.train.warmup_steps >= self.train.steps:
            raise ValueError("warmup_steps must be less than steps")
        if self.train.minimum_learning_rate > self.train.peak_learning_rate:
            raise ValueError("minimum_learning_rate must not exceed the sqrt(batch)-scaled peak")
        if (
            self.matching.method == "exact"
            and self.matching.pool_size < self.train.effective_batch_size
        ):
            raise ValueError("matching.pool_size must be at least batch_size * accumulate_steps")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if path.suffix == ".json":
        return ExperimentConfig.model_validate_json(path.read_text())
    with path.open("rb") as stream:
        return ExperimentConfig.model_validate(tomllib.load(stream))


def save_config(config: ExperimentConfig, path: Path) -> None:
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")
