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
    architecture: Literal["unet", "nested"] = "unet"
    channels: tuple[int, ...] = (4, 10, 20, 40)
    convolutions_per_block: int = Field(default=3, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    # Cumulative encoder widths above the smallest model in `channels`.
    nested_channels: tuple[tuple[int, ...], ...] = ()
    nested_execution: Literal["fused", "blocks"] = "fused"
    output_mode: Literal["direct", "residual"] = "direct"
    # Experimental backward rule; both choices have the same clipped forward values.
    output_clip_gradient: Literal["hard", "straight_through"] = "hard"
    residual_scale_floor: float = Field(default=0.0, ge=0, le=1)
    # Experimental output anchor; the network still receives the sampled image.
    residual_base: Literal["noisy", "source"] = "noisy"
    feature_input: Literal["noisy", "source"] = "noisy"
    # Source-only restoration does not consume intermediate diffusion states.
    sampling_mode: Literal["diffusion", "one_pass"] = "diffusion"
    source_conditioning: bool = False
    progress_conditioning: bool = False
    guide_encoding: Literal["raw", "ink", "sqrt_ink"] = "raw"
    # Optional full-resolution output branch; zero preserves the original model.
    detail_channels: int = Field(default=0, ge=0)
    skip_mode: Literal["post_pool", "pre_pool"] = "post_pool"
    bottleneck_dilation: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def validate_architecture(self):
        if self.sampling_mode == "one_pass" and self.feature_input != "source":
            raise ValueError("One-pass restoration requires source feature input")
        if self.feature_input == "source" and self.residual_base != "source":
            raise ValueError("Source feature input requires the source residual base")
        if self.residual_base == "source" and (
            self.architecture != "unet"
            or self.output_mode != "residual"
            or not self.source_conditioning
            or self.progress_conditioning
        ):
            raise ValueError(
                "Source residual base requires a guided residual UNet without progress"
            )
        if self.progress_conditioning and (
            self.architecture != "unet"
            or not self.source_conditioning
            or self.output_mode != "residual"
        ):
            raise ValueError("progress_conditioning requires a source-conditioned residual UNet")
        if self.detail_channels and (self.architecture != "unet" or not self.source_conditioning):
            raise ValueError("detail_channels requires a source-conditioned UNet")
        if self.residual_scale_floor and (
            self.architecture != "unet" or self.output_mode != "residual"
        ):
            raise ValueError("residual_scale_floor requires a residual UNet")
        if not self.source_conditioning and self.guide_encoding != "raw":
            raise ValueError("guide_encoding requires source_conditioning=True")
        if self.architecture != "unet" and self.source_conditioning:
            raise ValueError("Source conditioning currently requires architecture='unet'")
        if self.architecture != "unet" and self.output_clip_gradient != "hard":
            raise ValueError("Surrogate output gradients currently require architecture='unet'")
        if not self.channels or min(self.channels) < 1:
            raise ValueError("channels must contain positive channel counts")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if self.architecture != "nested" and self.nested_channels:
            raise ValueError("nested_channels requires architecture='nested'")
        previous = self.channels
        for widths in self.nested_channels:
            if len(widths) != len(previous) or any(
                larger <= smaller for smaller, larger in zip(previous, widths, strict=True)
            ):
                raise ValueError("Each nested width must increase at every encoder stage")
            previous = widths
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
    final_test_count: int = Field(default=0, ge=0)
    split_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    workers: int = Field(default=4, ge=0)
    prefetch_batches: int = Field(default=2, ge=1)
    augment: bool = True
    image_scale: float = Field(default=1.1, gt=0)
    alpha_mode: Literal["ignore", "white"] = "ignore"
    prepared_root: Path | None = None
    # Read the level-0 teacher while independently retaining sampled recursion depth.
    native_training_targets: bool = False

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
        if self.native_training_targets and 0 not in self.resolved_scale_levels:
            raise ValueError("native_training_targets requires a level-0 directory")
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
    strength_sampling: Literal["uniform", "low_mixture"] = "uniform"
    low_strength_max: float = Field(default=0.5, gt=0, lt=1)
    low_strength_probability: float = Field(default=0.8, gt=0, lt=1)
    mask_sampling: Literal["uniform", "boundary_rect"] = "uniform"
    boundary_candidates: int = Field(default=8, ge=1, le=32)
    identity_probability: float = Field(default=0.1, ge=0, le=1)
    pattern_weights: tuple[float, float, float, float] = (1, 1, 1, 1)
    scale_loss_multiplier: float = Field(default=0.0, ge=0)
    sample_steps: int = Field(default=5, ge=1, le=20)
    training_path: Literal["rough", "bridge"] = "rough"
    progress_minimum: float = Field(default=0.05, gt=0, lt=1)
    progress_initial_probability: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self):
        if self.mask_sampling == "boundary_rect" and self.pattern_weights[3]:
            raise ValueError("boundary_rect requires the protected-lines pattern to be disabled")
        if self.strength_sampling == "low_mixture" and self.low_strength_max <= self.min_noise:
            raise ValueError("low_strength_max must exceed min_noise")
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
    # Draw a common random pool, then take an image-independent prefix before OT.
    # This keeps candidate draws identical across assignment ablations.
    generation_pool_size: int | None = Field(default=None, ge=1)


class SketchAugmentationConfig(Settings):
    """Corrupt only the input, at crop resolution before constructing the pyramid."""

    mode: Literal["none", "tone", "blur", "strokes", "mixed"] = "none"
    probability: float = Field(default=0.75, ge=0, le=1)
    opacity_min: float = Field(default=0.25, gt=0, le=1)
    opacity_max: float = Field(default=1.0, gt=0, le=1)
    blur_sigma_min: float = Field(default=0.0, ge=0, le=4)
    blur_sigma_max: float = Field(default=2.0, ge=0, le=4)
    dilation_radius: int = Field(default=2, ge=0, le=2)
    dilation_blend: float = Field(default=1.0, ge=0, le=1)
    displacement: float = Field(default=2.0, ge=0, le=8)
    strong_probability: float = Field(default=0.0, ge=0, le=1)
    strong_opacity_min: float = Field(default=0.2, gt=0, le=1)
    strong_opacity_max: float = Field(default=0.45, gt=0, le=1)
    strong_blur_sigma_min: float = Field(default=1.5, ge=0, le=4)
    strong_blur_sigma_max: float = Field(default=2.5, ge=0, le=4)
    strong_displacement: float = Field(default=3.0, ge=0, le=8)

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.opacity_min > self.opacity_max or self.blur_sigma_min > self.blur_sigma_max:
            raise ValueError("Sketch augmentation bounds must be ordered")
        if (
            self.strong_opacity_min > self.strong_opacity_max
            or self.strong_blur_sigma_min > self.strong_blur_sigma_max
        ):
            raise ValueError("Strong sketch augmentation bounds must be ordered")
        return self


class GeometryDataConfig(Settings):
    """Replace a fraction of clean training targets with independently generated paths."""

    probability: float = Field(default=0.0, ge=0, le=1)
    blank_probability: float = Field(default=0.05, ge=0, le=1)
    supersampling: int = Field(default=4, ge=2, le=8)
    minimum_width: float = Field(default=0.6, gt=0)
    maximum_width: float = Field(default=12.0, gt=0)
    # Conditional probability among crops not replaced by the original paths.
    coverage_probability: float = Field(default=0.0, ge=0, le=1)
    coverage_minimum_width: float = Field(default=16.0, gt=0)
    coverage_maximum_width: float = Field(default=128.0, gt=0)

    @model_validator(mode="after")
    def validate_widths(self):
        if self.minimum_width > self.maximum_width:
            raise ValueError("Geometry width bounds must be ordered")
        if self.coverage_minimum_width > self.coverage_maximum_width:
            raise ValueError("Coverage width bounds must be ordered")
        return self


class LossConfig(Settings):
    pixel_loss: Literal["clipped_mse", "interval_mse", "clipped_l1", "interval_l1"] = "clipped_mse"
    paper_peak_weight: float = Field(default=0.0, ge=0)
    paper_peak_tile: int = Field(default=8, ge=1, le=32)
    ink_keep_weight: float = Field(default=0.0, ge=0)
    opacity_deficit_weight: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    opacity_deficit_preclip: bool = False
    neighbor_keep_weight: float = Field(default=0.0, ge=0)
    neighbor_paper_weight: float = Field(default=0.0, ge=0)
    neighbor_radius: int = Field(default=1, ge=0, le=4)
    line_region_weight: float = Field(default=0.0, ge=0)
    line_region_radius: int = Field(default=6, ge=1, le=16)
    line_profile_weight: float = Field(default=0.0, ge=0)
    cldice_weight: float = Field(default=0.0, ge=0)
    cldice_iterations: int = Field(default=16, ge=1, le=64)
    boundary_weight: float = Field(default=0.0, ge=0)
    boundary_radius: int = Field(default=4, ge=1, le=16)
    balance_tones: bool = True
    tone_balance_power: float = Field(default=1.0, ge=0, le=1)
    identity_weight: float = Field(default=0.1, ge=0)
    edge_weight: float = Field(default=0.1, ge=0)
    contrast_weight: float = Field(default=0.01, ge=0)
    # Sketch degradation does not vanish with Gaussian variance. Bound its loss weight.
    noise_weight_floor: float = Field(default=0.0, ge=0, le=1)
    contrast_active_only: bool = False

    @model_validator(mode="after")
    def validate_opacity_preclip(self):
        if self.opacity_deficit_preclip and not self.opacity_deficit_weight:
            raise ValueError("opacity_deficit_preclip requires a positive opacity_deficit_weight")
        return self


class TrainConfig(Settings):
    steps: int = Field(default=100_000, ge=1)
    # Keep a completed decay schedule when extending a restored training run.
    learning_rate_decay_steps: int | None = Field(default=None, ge=1)
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
    model_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)

    @property
    def initialization_seed(self):
        return self.seed if self.model_seed is None else self.model_seed

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
    sketch: SketchAugmentationConfig = SketchAugmentationConfig()
    geometry: GeometryDataConfig = GeometryDataConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()

    @model_validator(mode="after")
    def validate_shapes_and_schedule(self):
        if self.diffusion.training_path == "bridge" and not self.model.progress_conditioning:
            raise ValueError("bridge training requires progress_conditioning")
        if self.model.progress_conditioning and self.matching.method != "independent":
            raise ValueError("Progress-conditioned training requires independent Gaussian noise")
        if self.loss.pixel_loss in ("interval_mse", "interval_l1") and (
            self.model.architecture != "unet" or self.model.output_clip_gradient != "hard"
        ):
            raise ValueError(
                f"{self.loss.pixel_loss} requires a single UNet with hard output clipping"
            )
        if self.loss.opacity_deficit_preclip and (
            self.model.architecture != "unet" or self.model.output_clip_gradient != "hard"
        ):
            raise ValueError("Preclip opacity requires a single UNet with hard output clipping")
        size = self.data.crop_size
        if size < self.model.minimum_size or size & (size - 1):
            raise ValueError("crop_size must be a power of two and at least the model minimum")
        if self.train.warmup_steps >= self.train.steps:
            raise ValueError("warmup_steps must be less than steps")
        decay_steps = self.train.learning_rate_decay_steps or self.train.steps
        if self.train.warmup_steps >= decay_steps:
            raise ValueError("warmup_steps must be less than learning_rate_decay_steps")
        if self.train.minimum_learning_rate > self.train.peak_learning_rate:
            raise ValueError("minimum_learning_rate must not exceed the sqrt(batch)-scaled peak")
        if self.sketch.mode != "none" and self.matching.method != "independent":
            raise ValueError("Sketch augmentation currently requires independent Gaussian noise")
        if (
            self.matching.method == "exact"
            and self.matching.pool_size < self.train.effective_batch_size
        ):
            raise ValueError("matching.pool_size must be at least batch_size * accumulate_steps")
        needed = (
            self.matching.pool_size
            if self.matching.method == "exact"
            else self.train.effective_batch_size
        )
        if (
            self.matching.generation_pool_size is not None
            and self.matching.generation_pool_size < needed
        ):
            raise ValueError("generation_pool_size must cover the assignment pool or full batch")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if path.suffix == ".json":
        return ExperimentConfig.model_validate_json(path.read_text())
    with path.open("rb") as stream:
        return ExperimentConfig.model_validate(tomllib.load(stream))


def save_config(config: ExperimentConfig, path: Path) -> None:
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")
