"""Typed, strict experiment settings. Paths are relative to the project directory."""

import json
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
    crop_size: int = Field(default=512, ge=16)
    validation_count: int = Field(default=64, ge=1)
    workers: int = Field(default=4, ge=0)
    prefetch_batches: int = Field(default=2, ge=1)
    augment: bool = True
    image_scale: float = Field(default=1.1, gt=0)
    alpha_mode: Literal["ignore", "white"] = "ignore"
    prepared_root: Path | None = None


class DiffusionConfig(Settings):
    recursion_levels: int = Field(default=3, ge=1)
    min_noise: float = Field(default=0.001, gt=0, lt=1)
    identity_probability: float = Field(default=0.1, ge=0, le=1)
    pattern_weights: tuple[float, float, float, float] = (1, 1, 1, 1)
    scale_loss_decay: float = Field(default=0.5, ge=0, le=1)
    sample_steps: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_weights(self):
        if min(self.pattern_weights) < 0 or sum(self.pattern_weights) <= 0:
            raise ValueError("pattern_weights must be nonnegative with a positive sum")
        return self


class LossConfig(Settings):
    balance_tones: bool = True
    identity_weight: float = Field(default=0.1, ge=0)
    edge_weight: float = Field(default=0.1, ge=0)
    contrast_weight: float = Field(default=0.01, ge=0)


class TrainConfig(Settings):
    steps: int = Field(default=100_000, ge=1)
    batch_size: int = Field(default=4, ge=1)
    accumulate_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    minimum_learning_rate: float = Field(default=3e-5, ge=0)
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


class ExperimentConfig(Settings):
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()

    @model_validator(mode="after")
    def validate_shapes_and_schedule(self):
        divisor = self.model.minimum_size * 2 ** (self.diffusion.recursion_levels - 1)
        if self.data.crop_size % divisor:
            raise ValueError(f"crop_size must be divisible by {divisor} for this model/pyramid")
        if self.train.warmup_steps >= self.train.steps:
            raise ValueError("warmup_steps must be less than steps")
        if self.train.minimum_learning_rate > self.train.learning_rate:
            raise ValueError("minimum_learning_rate must not exceed learning_rate")
        if not self.data.scale_dirs:
            raise ValueError("at least one scale directory is required")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if path.suffix == ".json":
        return ExperimentConfig.model_validate_json(path.read_text())
    with path.open("rb") as stream:
        return ExperimentConfig.model_validate(tomllib.load(stream))


def save_config(config: ExperimentConfig, path: Path) -> None:
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")
