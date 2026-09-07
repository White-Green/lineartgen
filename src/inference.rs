use crate::model::{DiffusionModel, DiffusionModelConfig};
use crate::recursion::{forward_recursive, tensor_pyramid};
use burn::module::Module;
use burn::record::{HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const DEFAULT_RECURSION_STOP_SIZE: usize = 128;
pub const IMAGE_SCALE: f32 = 1.1;
pub const OUTPUT_ALPHA_THRESHOLD: f32 = 0.1;
pub const MAX_DENOISING_STEPS: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InferenceOptions {
    pub strength: f32,
    pub seed: u64,
    pub denoising_steps: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyramidPlan {
    pub original_size: [usize; 2],
    pub padded_size: [usize; 2],
    pub sizes: Vec<[usize; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    InvalidArgument(String),
    Backend(String),
    Model(String),
}

impl Display for InferenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(formatter, "{message}"),
            Self::Backend(message) => write!(formatter, "backend error: {message}"),
            Self::Model(message) => write!(formatter, "model error: {message}"),
        }
    }
}

impl Error for InferenceError {}

#[derive(Module, Debug)]
struct DiffusionCheckpoint<B: Backend> {
    model: DiffusionModel<B>,
}

pub struct InferenceSession<B: Backend> {
    model: DiffusionModel<B>,
    device: B::Device,
    recursion_stop_size: usize,
}

impl<B: Backend> InferenceSession<B> {
    pub fn from_embedded(
        model_config_json: &[u8],
        checkpoint: &[u8],
        device: B::Device,
    ) -> Result<Self, InferenceError> {
        let config = serde_json::from_slice::<DiffusionModelConfig>(model_config_json)
            .map_err(|error| InferenceError::Model(format!("invalid embedded model config: {error}")))?;
        let unloaded = DiffusionCheckpoint {
            model: config.init::<B>(&device),
        };
        let recorder = NamedMpkBytesRecorder::<HalfPrecisionSettings>::new();
        let record = recorder
            .load(checkpoint.to_vec(), &device)
            .map_err(|error| InferenceError::Model(format!("could not load embedded checkpoint: {error}")))?;
        let model = unloaded.load_record(record).model;

        Ok(Self {
            model,
            device,
            recursion_stop_size: DEFAULT_RECURSION_STOP_SIZE,
        })
    }

    #[cfg(test)]
    fn from_model(model: DiffusionModel<B>, device: B::Device) -> Self {
        Self {
            model,
            device,
            recursion_stop_size: DEFAULT_RECURSION_STOP_SIZE,
        }
    }

    pub fn infer_bgra(
        &self,
        scribble: &[u8],
        lineart: &[u8],
        width: usize,
        height: usize,
        options: InferenceOptions,
    ) -> Result<Vec<u8>, InferenceError> {
        validate_options(options)?;
        let byte_len = bgra_byte_len(width, height)?;
        validate_buffer("scribble", scribble, byte_len)?;
        validate_buffer("lineart", lineart, byte_len)?;

        let minimum = self.model.minimum_input_size();
        let plan = automatic_pyramid_plan([height, width], minimum, self.recursion_stop_size)?;
        let prepared = prepare_input(scribble, lineart, width, height, &plan, options.strength)?;
        let [padded_height, padded_width] = plan.padded_size;
        let shape = [1, 1, padded_height, padded_width];
        let clean = Tensor::<B, 4>::from_data(TensorData::new(prepared.clean, shape), &self.device);
        let noise_level = Tensor::<B, 4>::from_data(TensorData::new(prepared.noise_level, shape), &self.device);
        let noise = Tensor::<B, 4>::from_data(
            TensorData::new(gaussian_noise(padded_height * padded_width, options.seed), shape),
            &self.device,
        );
        let signal_scale = (noise_level.clone().neg() + 1.0).sqrt();
        let noise_scale = noise_level.clone().sqrt();
        let noisy = q_sample(clean, noise, signal_scale, noise_scale);
        let predicted = denoise_sample(
            &self.model,
            noisy,
            noise_level,
            options.strength,
            options.denoising_steps,
            &plan.sizes,
        );
        let values = predicted
            .try_into_data()
            .map_err(|error| InferenceError::Backend(error.to_string()))?
            .into_vec::<f32>()
            .map_err(|error| InferenceError::Backend(error.to_string()))?;

        predicted_values_to_bgra(&values, plan.padded_size, plan.original_size)
    }
}

pub fn automatic_pyramid_plan(
    original_size: [usize; 2],
    minimum_size: [usize; 2],
    stop_size: usize,
) -> Result<PyramidPlan, InferenceError> {
    if minimum_size.contains(&0) {
        return Err(InferenceError::Model(
            "model minimum input dimensions must be greater than zero".to_string(),
        ));
    }
    if stop_size < minimum_size[0].max(minimum_size[1]) {
        return Err(InferenceError::Model(format!(
            "recursion stop size {stop_size} is smaller than model minimum {minimum_size:?}"
        )));
    }
    if original_size[0] < minimum_size[0] || original_size[1] < minimum_size[1] {
        return Err(InferenceError::InvalidArgument(format!(
            "input size {original_size:?} must be at least the model minimum {minimum_size:?}"
        )));
    }

    let mut conceptual_size = original_size;
    let mut reductions = 0usize;
    while conceptual_size[0] > stop_size || conceptual_size[1] > stop_size {
        let next = [conceptual_size[0].div_ceil(2), conceptual_size[1].div_ceil(2)];
        if next[0] < minimum_size[0] || next[1] < minimum_size[1] {
            break;
        }
        conceptual_size = next;
        reductions += 1;
    }

    let shift = u32::try_from(reductions)
        .map_err(|_| InferenceError::InvalidArgument("input requires too many pyramid levels".to_string()))?;
    let alignment = [
        minimum_size[0]
            .checked_shl(shift)
            .ok_or_else(|| InferenceError::InvalidArgument("pyramid height alignment overflowed".to_string()))?,
        minimum_size[1]
            .checked_shl(shift)
            .ok_or_else(|| InferenceError::InvalidArgument("pyramid width alignment overflowed".to_string()))?,
    ];
    let padded_size = [
        round_up(original_size[0], alignment[0])?,
        round_up(original_size[1], alignment[1])?,
    ];
    let sizes = (0..=reductions)
        .map(|level| [padded_size[0] >> level, padded_size[1] >> level])
        .collect();

    Ok(PyramidPlan {
        original_size,
        padded_size,
        sizes,
    })
}

pub fn normalize_luma(value: u8) -> f32 {
    normalize_luma_unit(f32::from(value) / 255.0)
}

pub fn normalize_luma_unit(value: f32) -> f32 {
    ((value * 2.0 - 1.0) * IMAGE_SCALE).clamp(-1.0, 1.0)
}

pub fn prediction_to_luma(value: f32) -> u8 {
    (((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0).round() as u8
}

pub fn q_sample<B: Backend>(
    clean: Tensor<B, 4>,
    noise: Tensor<B, 4>,
    signal_scale: Tensor<B, 4>,
    noise_scale: Tensor<B, 4>,
) -> Tensor<B, 4> {
    clean * signal_scale + noise * noise_scale
}

pub fn denoise_sample<B: Backend>(
    model: &DiffusionModel<B>,
    mut sample: Tensor<B, 4>,
    start_noise_level_tensor: Tensor<B, 4>,
    start_noise_level: f32,
    denoising_steps: usize,
    sizes: &[[usize; 2]],
) -> Tensor<B, 4> {
    let schedule = denoising_schedule(start_noise_level, denoising_steps);

    for window in schedule.windows(2) {
        let noise_level_factor = relative_noise_level(window[0], start_noise_level);
        let next_noise_level_factor = relative_noise_level(window[1], start_noise_level);
        let noise_level = start_noise_level_tensor.clone() * noise_level_factor;
        let predicted_clean = forward_recursive(
            model,
            tensor_pyramid(sizes, sample.clone()),
            tensor_pyramid(sizes, noise_level),
        )
        .into_iter()
        .next()
        .expect("at least one sample scale is required");
        sample = denoise_next_sample(
            sample,
            predicted_clean,
            start_noise_level_tensor.clone(),
            noise_level_factor,
            next_noise_level_factor,
        );
    }

    sample
}

pub fn denoising_schedule(start_noise_level: f32, denoising_steps: usize) -> Vec<f32> {
    assert!(denoising_steps > 0, "denoising_steps must be greater than zero");
    let start_noise_level = start_noise_level.clamp(0.0, 1.0);
    if start_noise_level == 0.0 {
        return vec![0.0, 0.0];
    }

    (0..=denoising_steps)
        .map(|index| start_noise_level * (denoising_steps - index) as f32 / denoising_steps as f32)
        .collect()
}

fn relative_noise_level(noise_level: f32, start_noise_level: f32) -> f32 {
    if start_noise_level == 0.0 {
        0.0
    } else {
        noise_level / start_noise_level
    }
}

pub(crate) fn denoise_next_sample<B: Backend>(
    sample: Tensor<B, 4>,
    predicted_clean: Tensor<B, 4>,
    start_noise_level_tensor: Tensor<B, 4>,
    noise_level_factor: f32,
    next_noise_level_factor: f32,
) -> Tensor<B, 4> {
    let current_noise_level = start_noise_level_tensor.clone() * noise_level_factor;
    let next_noise_level = start_noise_level_tensor * next_noise_level_factor;
    let signal_scale = (current_noise_level.neg() + 1.0).sqrt();
    let next_signal_scale = (next_noise_level.neg() + 1.0).sqrt();
    let noise_ratio = if noise_level_factor <= 0.0 {
        0.0
    } else {
        (next_noise_level_factor / noise_level_factor).clamp(0.0, 1.0).sqrt() as f64
    };
    let predicted_clean_weight = next_signal_scale - signal_scale * noise_ratio;

    sample * noise_ratio + predicted_clean * predicted_clean_weight
}

fn validate_options(options: InferenceOptions) -> Result<(), InferenceError> {
    if !options.strength.is_finite() || !(0.0..=1.0).contains(&options.strength) {
        return Err(InferenceError::InvalidArgument(
            "strength must be finite and between 0 and 1".to_string(),
        ));
    }
    if !(1..=MAX_DENOISING_STEPS).contains(&options.denoising_steps) {
        return Err(InferenceError::InvalidArgument(format!(
            "denoising_steps must be between 1 and {MAX_DENOISING_STEPS}"
        )));
    }
    Ok(())
}

fn bgra_byte_len(width: usize, height: usize) -> Result<usize, InferenceError> {
    if width == 0 || height == 0 {
        return Err(InferenceError::InvalidArgument(
            "width and height must be greater than zero".to_string(),
        ));
    }
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| InferenceError::InvalidArgument("image byte length overflowed".to_string()))
}

fn validate_buffer(name: &str, buffer: &[u8], expected_len: usize) -> Result<(), InferenceError> {
    if buffer.len() != expected_len {
        return Err(InferenceError::InvalidArgument(format!(
            "{name} must contain exactly {expected_len} bytes, got {}",
            buffer.len()
        )));
    }
    Ok(())
}

fn round_up(value: usize, alignment: usize) -> Result<usize, InferenceError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| InferenceError::InvalidArgument("padded image dimension overflowed".to_string()))
    }
}

struct PreparedInput {
    clean: Vec<f32>,
    noise_level: Vec<f32>,
}

fn prepare_input(
    scribble: &[u8],
    lineart: &[u8],
    width: usize,
    height: usize,
    plan: &PyramidPlan,
    strength: f32,
) -> Result<PreparedInput, InferenceError> {
    let [padded_height, padded_width] = plan.padded_size;
    let padded_pixels = padded_height
        .checked_mul(padded_width)
        .ok_or_else(|| InferenceError::InvalidArgument("padded image area overflowed".to_string()))?;
    let mut clean = vec![1.0; padded_pixels];
    let mut noise_level = vec![0.0; padded_pixels];

    for y in 0..height {
        for x in 0..width {
            let source = (y * width + x) * 4;
            let target = y * padded_width + x;
            let scribble_pixel = &scribble[source..source + 4];
            let lineart_pixel = &lineart[source..source + 4];
            let scribble_luma = alpha_over(1.0, bgra_luma(scribble_pixel), alpha(scribble_pixel));
            let lineart_alpha = alpha(lineart_pixel);
            let combined_luma = alpha_over(scribble_luma, bgra_luma(lineart_pixel), lineart_alpha);

            clean[target] = normalize_luma_unit(combined_luma);
            noise_level[target] = strength * (1.0 - lineart_alpha);
        }
    }

    Ok(PreparedInput { clean, noise_level })
}

fn bgra_luma(pixel: &[u8]) -> f32 {
    let red = u32::from(pixel[2]);
    let green = u32::from(pixel[1]);
    let blue = u32::from(pixel[0]);
    (2126 * red + 7152 * green + 722 * blue) as f32 / (10_000.0 * 255.0)
}

fn alpha(pixel: &[u8]) -> f32 {
    f32::from(pixel[3]) / 255.0
}

fn alpha_over(background: f32, foreground: f32, alpha: f32) -> f32 {
    foreground * alpha + background * (1.0 - alpha)
}

fn gaussian_noise(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| StandardNormal.sample(&mut rng)).collect()
}

fn predicted_values_to_bgra(
    values: &[f32],
    padded_size: [usize; 2],
    original_size: [usize; 2],
) -> Result<Vec<u8>, InferenceError> {
    let [padded_height, padded_width] = padded_size;
    let [height, width] = original_size;
    let expected_values = padded_height
        .checked_mul(padded_width)
        .ok_or_else(|| InferenceError::Backend("prediction shape overflowed".to_string()))?;
    if values.len() != expected_values {
        return Err(InferenceError::Backend(format!(
            "prediction contains {} values, expected {expected_values}",
            values.len()
        )));
    }
    let output_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| InferenceError::InvalidArgument("output byte length overflowed".to_string()))?;
    let mut output = Vec::with_capacity(output_len);

    for y in 0..height {
        for x in 0..width {
            let luma = prediction_to_luma(values[y * padded_width + x]);
            let mut output_alpha = 255 - luma;
            if f32::from(output_alpha) / 255.0 < OUTPUT_ALPHA_THRESHOLD {
                output_alpha = 0;
            }
            output.extend_from_slice(&[0, 0, 0, output_alpha]);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DiffusionModelConfig;

    #[test]
    fn pyramid_levels_are_derived_from_image_size() {
        let plan = automatic_pyramid_plan([1024, 1024], [16, 16], 128).unwrap();

        assert_eq!(plan.padded_size, [1024, 1024]);
        assert_eq!(plan.sizes, [[1024, 1024], [512, 512], [256, 256], [128, 128]]);
    }

    #[test]
    fn images_at_the_stop_size_use_one_model_call() {
        let plan = automatic_pyramid_plan([112, 128], [16, 16], 128).unwrap();

        assert_eq!(plan.padded_size, [112, 128]);
        assert_eq!(plan.sizes, [[112, 128]]);
    }

    #[test]
    fn pyramid_padding_is_minimal_for_all_levels() {
        let plan = automatic_pyramid_plan([600, 1000], [16, 16], 128).unwrap();

        assert_eq!(plan.padded_size, [640, 1024]);
        assert_eq!(plan.sizes, [[640, 1024], [320, 512], [160, 256], [80, 128]]);
    }

    #[test]
    fn pyramid_stops_before_an_extreme_aspect_ratio_becomes_too_small() {
        let plan = automatic_pyramid_plan([128, 1024], [16, 16], 128).unwrap();

        assert_eq!(plan.sizes, [[128, 1024], [64, 512], [32, 256], [16, 128]]);
    }

    #[test]
    fn normalization_expands_and_clamps_without_inverse_output_scaling() {
        assert_eq!(normalize_luma(0), -1.0);
        assert_eq!(normalize_luma(255), 1.0);
        assert!(normalize_luma(64) < -0.5);
        assert_eq!(prediction_to_luma(0.0), 128);
    }

    #[test]
    fn lineart_alpha_protects_existing_lines_from_noise() {
        let plan = automatic_pyramid_plan([16, 16], [16, 16], 128).unwrap();
        let scribble = vec![255; 16 * 16 * 4];
        let mut lineart = vec![0; 16 * 16 * 4];
        lineart[3] = 255;
        lineart[7] = 128;

        let prepared = prepare_input(&scribble, &lineart, 16, 16, &plan, 0.8).unwrap();

        assert_eq!(prepared.noise_level[0], 0.0);
        assert!((prepared.noise_level[1] - 0.8 * (1.0 - 128.0 / 255.0)).abs() < 1.0e-6);
        assert_eq!(prepared.noise_level[2], 0.8);
    }

    #[test]
    fn host_noise_is_reproducible_and_seeded() {
        let first = gaussian_noise(32, 7);
        let repeated = gaussian_noise(32, 7);
        let different = gaussian_noise(32, 8);

        assert_eq!(first, repeated);
        assert_ne!(first, different);
    }

    #[test]
    fn inputs_smaller_than_the_model_minimum_are_rejected() {
        let error = automatic_pyramid_plan([15, 16], [16, 16], 128).unwrap_err();

        assert!(matches!(error, InferenceError::InvalidArgument(_)));
    }

    #[test]
    fn invalid_options_and_buffer_sizes_are_rejected() {
        assert!(
            validate_options(InferenceOptions {
                strength: f32::NAN,
                seed: 0,
                denoising_steps: 1,
            })
            .is_err()
        );
        assert!(
            validate_options(InferenceOptions {
                strength: 0.5,
                seed: 0,
                denoising_steps: 0,
            })
            .is_err()
        );
        assert!(
            validate_options(InferenceOptions {
                strength: 0.5,
                seed: 0,
                denoising_steps: MAX_DENOISING_STEPS + 1,
            })
            .is_err()
        );
        assert!(bgra_byte_len(0, 16).is_err());
        assert!(validate_buffer("image", &[0; 3], 4).is_err());
    }

    #[test]
    fn output_is_black_bgra_with_transparency_threshold() {
        let output = predicted_values_to_bgra(&[1.0, 0.9, -1.0], [1, 3], [1, 3]).unwrap();

        assert_eq!(output, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]);
    }

    #[cfg(any(feature = "inference-flex", feature = "training"))]
    #[test]
    fn model_pads_and_crops_arbitrary_non_power_of_two_sizes() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let session = InferenceSession::from_model(model, device);
        let pixels = vec![255; 17 * 33 * 4];
        let lineart = vec![0; 17 * 33 * 4];

        let output = session
            .infer_bgra(
                &pixels,
                &lineart,
                33,
                17,
                InferenceOptions {
                    strength: 0.0,
                    seed: 1,
                    denoising_steps: 1,
                },
            )
            .unwrap();

        assert_eq!(output.len(), pixels.len());
    }
}
