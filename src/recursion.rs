use crate::model::DiffusionModel;
use burn::config::Config;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::module::adaptive_avg_pool2d;

#[derive(Config, Debug)]
pub struct RecursionConfig {
    /// Number of model calls used for an original-scale image.
    #[config(default = "usize::MAX")]
    pub original_steps: usize,
    /// Do not add a smaller recursive input once both dimensions are at or below this size.
    /// Zero disables this limit.
    #[config(default = "0")]
    pub stop_size: usize,
}

impl RecursionConfig {
    pub fn validate(&self) {
        assert!(self.original_steps > 0, "original_steps must be greater than zero");
    }
}

pub fn input_sizes<B: Backend>(
    model: &DiffusionModel<B>,
    config: &RecursionConfig,
    base_size: [usize; 2],
    scale_level: usize,
) -> Vec<[usize; 2]> {
    config.validate();
    assert!(base_size[0].is_power_of_two(), "input height must be a power of two");
    assert!(base_size[1].is_power_of_two(), "input width must be a power of two");

    let minimum = model.minimum_input_size();
    assert!(
        base_size[0] >= minimum[0] && base_size[1] >= minimum[1],
        "input size must be at least the model minimum"
    );

    let max_steps = config.original_steps.saturating_add(scale_level);
    let mut sizes = Vec::new();
    let mut size = base_size;

    while sizes.len() < max_steps && size[0] >= minimum[0] && size[1] >= minimum[1] {
        sizes.push(size);
        if config.stop_size > 0 && size[0] <= config.stop_size && size[1] <= config.stop_size {
            break;
        }
        size = model.expected_insert_size(size);
    }

    sizes
}

pub fn tensor_pyramid<B: Backend>(sizes: &[[usize; 2]], tensor: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
    let [_, _, height, width] = tensor.dims();

    sizes
        .iter()
        .map(|&size| {
            if size == [height, width] {
                tensor.clone()
            } else {
                adaptive_avg_pool2d(tensor.clone(), size)
            }
        })
        .collect()
}

pub fn forward_recursive<B: Backend>(
    model: &DiffusionModel<B>,
    inputs: Vec<Tensor<B, 4>>,
    noise_levels: Vec<Tensor<B, 4>>,
) -> Vec<Tensor<B, 4>> {
    validate_pyramids(&inputs, &noise_levels);

    let mut insert = None;
    let mut outputs = Vec::with_capacity(inputs.len());
    for (input, noise_level) in inputs.into_iter().zip(noise_levels).rev() {
        let output = model.forward(input, noise_level, insert);
        insert = Some(output.clone());
        outputs.push(output);
    }
    outputs.reverse();
    outputs
}

pub fn forward_teacher_forced<B: Backend>(
    model: &DiffusionModel<B>,
    inputs: Vec<Tensor<B, 4>>,
    noise_levels: Vec<Tensor<B, 4>>,
    clean: &[Tensor<B, 4>],
) -> Vec<Tensor<B, 4>> {
    validate_pyramids(&inputs, &noise_levels);
    assert_eq!(inputs.len(), clean.len(), "clean pyramid length must match inputs");
    assert!(
        inputs
            .iter()
            .zip(clean)
            .all(|(input, clean)| input.dims() == clean.dims()),
        "clean pyramid shapes must match inputs"
    );

    inputs
        .into_iter()
        .zip(noise_levels)
        .enumerate()
        .map(|(index, (input, noise_level))| model.forward(input, noise_level, clean.get(index + 1).cloned()))
        .collect()
}

fn validate_pyramids<B: Backend>(inputs: &[Tensor<B, 4>], noise_levels: &[Tensor<B, 4>]) {
    assert!(!inputs.is_empty(), "at least one scale is required");
    assert_eq!(
        inputs.len(),
        noise_levels.len(),
        "noise level pyramid length must match inputs"
    );
    assert!(
        inputs
            .iter()
            .zip(noise_levels)
            .all(|(input, noise_level)| input.dims() == noise_level.dims()),
        "noise level pyramid shapes must match inputs"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DiffusionModelConfig;

    #[test]
    fn scale_level_adds_one_recursive_step_per_dataset_scale() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let mut config = RecursionConfig::new();
        config.original_steps = 1;

        for scale_level in 0..=5 {
            assert_eq!(
                input_sizes(&model, &config, [512, 512], scale_level).len(),
                scale_level + 1
            );
        }
    }

    #[test]
    fn recursion_is_capped_by_stop_and_minimum_sizes() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let mut config = RecursionConfig::new();
        config.stop_size = 64;

        assert_eq!(
            input_sizes(&model, &config, [512, 512], 5),
            vec![[512, 512], [256, 256], [128, 128], [64, 64]]
        );

        config.stop_size = 0;
        assert_eq!(
            input_sizes(&model, &config, [512, 512], usize::MAX),
            vec![[512, 512], [256, 256], [128, 128], [64, 64], [32, 32], [16, 16]]
        );
    }

    #[test]
    fn recursive_and_teacher_forced_execution_return_every_scale() {
        let device = Default::default();
        let model = DiffusionModelConfig::new().init::<burn::backend::Flex>(&device);
        let mut config = RecursionConfig::new();
        config.original_steps = 2;
        let sizes = input_sizes(&model, &config, [32, 32], 0);
        let input = Tensor::<_, 4>::zeros([1, 1, 32, 32], &device);
        let inputs = tensor_pyramid(&sizes, input.clone());
        let noise_levels = tensor_pyramid(&sizes, input.clone());
        let clean = tensor_pyramid(&sizes, input);

        let recursive = forward_recursive(&model, inputs.clone(), noise_levels.clone());
        let teacher_forced = forward_teacher_forced(&model, inputs, noise_levels, &clean);

        assert_eq!(recursive.len(), 2);
        assert_eq!(teacher_forced.len(), 2);
        assert_eq!(recursive[0].dims(), [1, 1, 32, 32]);
        assert_eq!(recursive[1].dims(), [1, 1, 16, 16]);
    }
}
