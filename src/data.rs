use crate::image_io::{ImageCrop, Result, read_lineart_crop_values, read_lineart_values};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineartItem {
    pub path: PathBuf,
    pub crop: Option<ImageCrop>,
    pub scale_level: usize,
}

#[derive(Clone, Debug)]
pub struct LineartDataset {
    items: LineartDatasetItems,
}

#[derive(Clone, Debug)]
enum LineartDatasetItems {
    Fixed(Vec<LineartItem>),
    RandomScale {
        sources: Vec<LineartSource>,
        crop_size: usize,
        seed: u64,
        access_counts: Arc<[AtomicU64]>,
    },
}

#[derive(Clone, Debug)]
struct LineartSource {
    index: usize,
    variants: Vec<LineartVariant>,
}

#[derive(Clone, Debug)]
struct LineartVariant {
    path: PathBuf,
    width: usize,
    height: usize,
    scale_level: usize,
}

impl LineartDataset {
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        let mut paths = fs::read_dir(path)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;

        paths.retain(|path| is_image_file(path));
        paths.sort();

        Ok(Self {
            items: LineartDatasetItems::Fixed(
                paths
                    .into_iter()
                    .map(|path| LineartItem {
                        path,
                        crop: None,
                        scale_level: 0,
                    })
                    .collect(),
            ),
        })
    }
}

impl Dataset<LineartItem> for LineartDataset {
    fn get(&self, index: usize) -> Option<LineartItem> {
        match &self.items {
            LineartDatasetItems::Fixed(items) => items.get(index).cloned(),
            LineartDatasetItems::RandomScale {
                sources,
                crop_size,
                seed,
                access_counts,
            } => {
                let source = sources.get(index)?;
                let access = access_counts[index].fetch_add(1, Ordering::Relaxed);
                let mut rng = StdRng::seed_from_u64(derive_seed(*seed, source.index, access as usize));
                let variant_index = rng.random_range(..source.variants.len());
                let variant = &source.variants[variant_index];

                Some(LineartItem {
                    path: variant.path.clone(),
                    crop: Some(random_crop(variant, *crop_size, &mut rng)),
                    scale_level: variant.scale_level,
                })
            }
        }
    }

    fn len(&self) -> usize {
        match &self.items {
            LineartDatasetItems::Fixed(items) => items.len(),
            LineartDatasetItems::RandomScale { sources, .. } => sources.len(),
        }
    }
}

pub fn multiscale_lineart_datasets<P: AsRef<Path>>(
    base_dir: impl AsRef<Path>,
    scale_dirs: &[P],
    crop_size: usize,
    valid_count: usize,
    seed: u64,
) -> Result<(LineartDataset, LineartDataset)> {
    if crop_size == 0 || !crop_size.is_power_of_two() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("dataset_crop_size must be a positive power of two, got {crop_size}"),
        )
        .into());
    }

    let base_dir = base_dir.as_ref();
    let base_images = images_by_filename(base_dir)?;
    if base_images.is_empty() {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("no images found in base dataset directory {}", base_dir.display()),
        )
        .into());
    }

    let mut image_sets = Vec::with_capacity(scale_dirs.len() + 1);
    image_sets.push(base_images);
    for scale_dir in scale_dirs {
        let scale_dir = scale_dir.as_ref();
        let images = images_by_filename(scale_dir)?;
        validate_matching_filenames(&image_sets[0], &images, scale_dir)?;
        image_sets.push(images);
    }

    let mut sources = Vec::with_capacity(image_sets[0].len());
    for (source_index, file_name) in image_sets[0].keys().enumerate() {
        let mut variants = Vec::with_capacity(image_sets.len());
        let base_path = image_sets[0]
            .get(file_name)
            .expect("base filename must exist in the base image set");
        let (base_width, base_height) = image::image_dimensions(base_path)?;
        let base_width = base_width as usize;
        let base_height = base_height as usize;

        for (scale_level, images) in image_sets.iter().enumerate() {
            let path = images
                .get(file_name)
                .expect("validated scale directories must contain every base filename")
                .clone();
            let (width, height) = image::image_dimensions(&path)?;
            let width = width as usize;
            let height = height as usize;
            let scale = 1usize.checked_shl(scale_level as u32).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("dataset scale level {scale_level} is too large"),
                )
            })?;
            let expected_width = base_width.checked_mul(scale).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "expected width overflow for {} at scale level {scale_level}",
                        path.display()
                    ),
                )
            })?;
            let expected_height = base_height.checked_mul(scale).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "expected height overflow for {} at scale level {scale_level}",
                        path.display()
                    ),
                )
            })?;
            if [width, height] != [expected_width, expected_height] {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "image {} is {width}x{height}; scale level {scale_level} requires exactly {expected_width}x{expected_height} from base {}x{}",
                        path.display(),
                        base_width,
                        base_height
                    ),
                )
                .into());
            }
            if width < crop_size || height < crop_size {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "image {} is {}x{}, smaller than dataset_crop_size {crop_size}",
                        path.display(),
                        width,
                        height
                    ),
                )
                .into());
            }
            variants.push(LineartVariant {
                path,
                width,
                height,
                scale_level,
            });
        }
        sources.push(LineartSource {
            index: source_index,
            variants,
        });
    }

    if valid_count >= sources.len() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "dataset must contain more than {valid_count} source images to create a validation split; found {}",
                sources.len()
            ),
        )
        .into());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    sources.shuffle(&mut rng);
    let train_sources = sources.split_off(valid_count);
    let valid_sources = sources;
    let access_counts = (0..train_sources.len())
        .map(|_| AtomicU64::new(0))
        .collect::<Vec<_>>()
        .into();
    let train_dataset = LineartDataset {
        items: LineartDatasetItems::RandomScale {
            sources: train_sources,
            crop_size,
            seed,
            access_counts,
        },
    };

    let mut valid_items = Vec::with_capacity(valid_count * image_sets.len());
    for source in valid_sources {
        for (variant_index, variant) in source.variants.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(derive_seed(seed ^ 0x7661_6c69_6461_7465, source.index, variant_index));
            valid_items.push(LineartItem {
                path: variant.path.clone(),
                crop: Some(random_crop(&variant, crop_size, &mut rng)),
                scale_level: variant.scale_level,
            });
        }
    }
    let valid_dataset = LineartDataset {
        items: LineartDatasetItems::Fixed(valid_items),
    };

    Ok((train_dataset, valid_dataset))
}

fn images_by_filename(path: &Path) -> Result<BTreeMap<OsString, PathBuf>> {
    let mut images = BTreeMap::new();
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if is_image_file(&path) {
            let file_name = path.file_name().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("image path has no filename: {}", path.display()),
                )
            })?;
            images.insert(file_name.to_os_string(), path);
        }
    }
    Ok(images)
}

fn validate_matching_filenames(
    base: &BTreeMap<OsString, PathBuf>,
    scale: &BTreeMap<OsString, PathBuf>,
    scale_dir: &Path,
) -> Result<()> {
    let missing = base
        .keys()
        .filter(|file_name| !scale.contains_key(*file_name))
        .map(|file_name| file_name.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let extra = scale
        .keys()
        .filter(|file_name| !base.contains_key(*file_name))
        .map(|file_name| file_name.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if missing.is_empty() && extra.is_empty() {
        return Ok(());
    }

    Err(Error::new(
        ErrorKind::InvalidData,
        format!(
            "dataset filenames in {} do not match the base directory: missing [{}], extra [{}]",
            scale_dir.display(),
            summarize_filenames(&missing),
            summarize_filenames(&extra)
        ),
    )
    .into())
}

fn summarize_filenames(file_names: &[String]) -> String {
    const DISPLAY_LIMIT: usize = 8;
    let mut summary = file_names
        .iter()
        .take(DISPLAY_LIMIT)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if file_names.len() > DISPLAY_LIMIT {
        summary.push_str(&format!(", ... ({} total)", file_names.len()));
    }
    summary
}

fn random_crop(variant: &LineartVariant, crop_size: usize, rng: &mut impl rand::Rng) -> ImageCrop {
    ImageCrop {
        x: rng.random_range(..=variant.width - crop_size),
        y: rng.random_range(..=variant.height - crop_size),
        size: crop_size,
    }
}

fn derive_seed(seed: u64, source_index: usize, sequence: usize) -> u64 {
    splitmix64(seed ^ splitmix64(source_index as u64) ^ splitmix64(sequence as u64 ^ 0x9e37_79b9_7f4a_7c15))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Debug)]
pub struct LineartBatch<B: Backend> {
    pub groups: Vec<LineartScaleBatch<B>>,
}

impl<B: Backend> LineartBatch<B> {
    pub fn item_count(&self) -> usize {
        self.groups.iter().map(LineartScaleBatch::item_count).sum()
    }
}

#[derive(Clone, Debug)]
pub struct LineartScaleBatch<B: Backend> {
    pub scale_level: usize,
    pub inputs: Tensor<B, 4>,
}

impl<B: Backend> LineartScaleBatch<B> {
    pub fn item_count(&self) -> usize {
        self.inputs.dims()[0]
    }
}

#[derive(Clone, Debug)]
pub struct LineartBatcher;

impl<B: Backend> Batcher<B, LineartItem, LineartBatch<B>> for LineartBatcher {
    fn batch(&self, items: Vec<LineartItem>, device: &B::Device) -> LineartBatch<B> {
        let mut grouped_tensors = BTreeMap::<usize, Vec<Tensor<B, 3>>>::new();
        for item in items {
            let scale_level = item.scale_level;
            let tensor = {
                let (values, height, width) = match item.crop {
                    Some(crop) => read_lineart_crop_values(&item.path, crop),
                    None => {
                        let file = File::open(&item.path)
                            .unwrap_or_else(|err| panic!("failed to open image {}: {err}", item.path.display()));
                        read_lineart_values(file)
                    }
                }
                .unwrap_or_else(|err| panic!("failed to read image {}: {err}", item.path.display()));
                let data = TensorData::new(values, [1, height, width]);
                Tensor::<B, 3>::from_data(data, device)
            };
            grouped_tensors.entry(scale_level).or_default().push(tensor);
        }

        let groups = grouped_tensors
            .into_iter()
            .map(|(scale_level, tensors)| LineartScaleBatch {
                scale_level,
                inputs: Tensor::stack::<4>(tensors, 0),
            })
            .collect();

        // Batching runs on loader workers, each of which owns a separate CubeCL stream.
        B::memory_cleanup(device);

        LineartBatch { groups }
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
    use burn::backend::Flex;
    use image::{GrayImage, Luma};
    use std::collections::HashSet;
    use std::sync::atomic::AtomicU64;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDataset {
        root: PathBuf,
        base: PathBuf,
        scales: Vec<PathBuf>,
    }

    impl TestDataset {
        fn new(source_count: usize, variant_count: usize, image_size: usize) -> Self {
            assert!(variant_count > 0);
            let root = std::env::temp_dir().join(format!(
                "lineartgen-data-test-{}-{}",
                std::process::id(),
                NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            let base = root.join("base");
            fs::create_dir_all(&base).unwrap();
            let scales = (1..variant_count)
                .map(|index| root.join(format!("scale-{index}")))
                .collect::<Vec<_>>();
            for scale in &scales {
                fs::create_dir_all(scale).unwrap();
            }

            for source_index in 0..source_count {
                let filename = format!("source-{source_index:02}.png");
                for (variant_index, directory) in std::iter::once(&base).chain(scales.iter()).enumerate() {
                    let variant_size = image_size * (1usize << variant_index);
                    let image = GrayImage::from_fn(variant_size as u32, variant_size as u32, |x, y| {
                        Luma([((source_index * 31 + variant_index * 17 + x as usize + y as usize) % 256) as u8])
                    });
                    image.save(directory.join(&filename)).unwrap();
                }
            }

            Self { root, base, scales }
        }
    }

    impl Drop for TestDataset {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn filename(item: &LineartItem) -> OsString {
        item.path.file_name().unwrap().to_os_string()
    }

    #[test]
    fn rejects_scale_directories_with_missing_and_extra_filenames() {
        let dataset = TestDataset::new(3, 2, 8);
        fs::remove_file(dataset.scales[0].join("source-01.png")).unwrap();
        GrayImage::from_pixel(8, 8, Luma([255]))
            .save(dataset.scales[0].join("extra.png"))
            .unwrap();

        let error = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 8, 1, 42)
            .unwrap_err()
            .to_string();

        assert!(error.contains("missing [source-01.png]"));
        assert!(error.contains("extra [extra.png]"));
    }

    #[test]
    fn rejects_non_power_of_two_crop_sizes() {
        let dataset = TestDataset::new(3, 1, 8);

        let error = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 3, 1, 42)
            .unwrap_err()
            .to_string();

        assert!(error.contains("positive power of two"));
    }

    #[test]
    fn rejects_scale_images_with_incorrect_dimensions() {
        let dataset = TestDataset::new(3, 2, 8);
        GrayImage::from_pixel(15, 16, Luma([255]))
            .save(dataset.scales[0].join("source-00.png"))
            .unwrap();

        let error = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 8, 1, 42)
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires exactly 16x16"));
    }

    #[test]
    fn splits_source_ids_before_expanding_scales() {
        let dataset = TestDataset::new(8, 6, 16);
        let (train, valid) = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 8, 2, 42).unwrap();

        assert_eq!(train.len(), 6);
        assert_eq!(valid.len(), 12);
        let train_ids = (0..train.len())
            .map(|index| filename(&train.get(index).unwrap()))
            .collect::<HashSet<_>>();
        let valid_ids = (0..valid.len())
            .map(|index| filename(&valid.get(index).unwrap()))
            .collect::<HashSet<_>>();

        assert_eq!(train_ids.len(), 6);
        assert_eq!(valid_ids.len(), 2);
        assert!(train_ids.is_disjoint(&valid_ids));
    }

    #[test]
    fn training_sampling_changes_per_access_and_is_reproducible() {
        let dataset = TestDataset::new(5, 6, 32);
        let (first, _) = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 8, 1, 123).unwrap();
        let (second, _) = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 8, 1, 123).unwrap();

        let first_sequence = (0..12).map(|_| first.get(0).unwrap()).collect::<Vec<_>>();
        let second_sequence = (0..12).map(|_| second.get(0).unwrap()).collect::<Vec<_>>();

        assert_eq!(first_sequence, second_sequence);
        assert!(first_sequence.iter().skip(1).any(|item| item != &first_sequence[0]));
        assert!(first_sequence.iter().all(|item| item.crop.unwrap().size == 8));
    }

    #[test]
    fn validation_uses_every_scale_and_fixed_crops() {
        let dataset = TestDataset::new(5, 6, 32);
        let (_, first) = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 8, 2, 123).unwrap();
        let (_, second) = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 8, 2, 123).unwrap();

        let first_items = (0..first.len())
            .map(|index| first.get(index).unwrap())
            .collect::<Vec<_>>();
        let second_items = (0..second.len())
            .map(|index| second.get(index).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(first.len(), 2 * 6);
        assert_eq!(first_items, second_items);
        for source_items in first_items.chunks_exact(6) {
            let directories = source_items
                .iter()
                .map(|item| item.path.parent().unwrap().to_path_buf())
                .collect::<HashSet<_>>();
            assert_eq!(directories.len(), 6);
            assert_eq!(
                source_items.iter().map(|item| item.scale_level).collect::<Vec<_>>(),
                vec![0, 1, 2, 3, 4, 5]
            );
        }
    }

    #[test]
    fn creates_512_pixel_batches_from_crops() {
        let dataset = TestDataset::new(3, 2, 512);
        let (_, valid) = multiscale_lineart_datasets(&dataset.base, &dataset.scales, 512, 1, 42).unwrap();
        let device = Default::default();
        let loader = lineart_dataloader::<Flex, _>(valid, 2, device, None, 0);

        let batch = loader.iter().next().unwrap();

        assert_eq!(batch.item_count(), 2);
        assert_eq!(batch.groups.len(), 2);
        assert_eq!(batch.groups[0].scale_level, 0);
        assert_eq!(batch.groups[0].inputs.dims(), [1, 1, 512, 512]);
        assert_eq!(batch.groups[1].scale_level, 1);
        assert_eq!(batch.groups[1].inputs.dims(), [1, 1, 512, 512]);
    }
}
