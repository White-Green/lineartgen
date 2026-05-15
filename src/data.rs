use crate::image_io::{Result, read_lineart_values};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct LineartItem {
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct LineartDataset {
    items: Vec<LineartItem>,
}

impl LineartDataset {
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        let mut paths = fs::read_dir(path)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;

        paths.retain(|path| is_image_file(path));
        paths.sort();

        Ok(Self {
            items: paths.into_iter().map(|path| LineartItem { path }).collect(),
        })
    }
}

impl Dataset<LineartItem> for LineartDataset {
    fn get(&self, index: usize) -> Option<LineartItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone, Debug)]
pub struct LineartBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,
}

#[derive(Clone, Debug)]
pub struct LineartBatcher;

impl<B: Backend> Batcher<B, LineartItem, LineartBatch<B>> for LineartBatcher {
    fn batch(&self, items: Vec<LineartItem>, device: &B::Device) -> LineartBatch<B> {
        let tensors = items
            .into_iter()
            .map(|item| {
                let file = File::open(&item.path)
                    .unwrap_or_else(|err| panic!("failed to open image {}: {err}", item.path.display()));
                let (values, height, width) = read_lineart_values(file)
                    .unwrap_or_else(|err| panic!("failed to read image {}: {err}", item.path.display()));
                let data = TensorData::new(values, [1, height, width]);
                Tensor::<B, 3>::from_data(data, device)
            })
            .collect::<Vec<_>>();

        let inputs = Tensor::stack::<4>(tensors, 0);

        LineartBatch { inputs }
    }
}

pub fn lineart_dataloader<B, D>(
    dataset: D,
    batch_size: usize,
    device: B::Device,
    shuffle_seed: Option<u64>,
    num_workers: usize,
) -> Arc<dyn DataLoader<B, LineartBatch<B>>>
where
    B: Backend,
    D: Dataset<LineartItem> + 'static,
{
    let builder = DataLoaderBuilder::new(LineartBatcher)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .set_device(device);

    match shuffle_seed {
        Some(seed) => builder.shuffle(seed).build(dataset),
        None => builder.build(dataset),
    }
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| matches!(extension.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BurnBackend;

    #[test]
    fn creates_batches_from_dataset_images() {
        let dataset = LineartDataset::from_dir("dataset/images").unwrap();
        let device = Default::default();
        let loader = lineart_dataloader::<BurnBackend, _>(dataset, 2, device, None, 0);

        let batch = loader.iter().next().unwrap();

        assert_eq!(batch.inputs.dims(), [2, 1, 1024, 1024]);
    }
}
