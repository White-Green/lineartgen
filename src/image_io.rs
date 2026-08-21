use crate::{BurnBackend, LineartTensor};
use burn::tensor::{Tensor, TensorData};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageBuffer, ImageEncoder, Luma, Pixel, Rgb};
use png::{BitDepth, ColorType as PngColorType, Decoder, Transformations};
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Read, Write};
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const IMAGE_SCALE: f32 = 1.1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageCrop {
    pub x: usize,
    pub y: usize,
    pub size: usize,
}

pub fn read_lineart_image(mut reader: impl Read) -> Result<LineartTensor> {
    let (values, height, width) = read_lineart_values(&mut reader)?;

    let device = Default::default();
    let data = TensorData::new(values, [height, width]);
    Ok(Tensor::<BurnBackend, 2>::from_data(data, &device))
}

pub fn read_lineart_values(mut reader: impl Read) -> Result<(Vec<f32>, usize, usize)> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    let image = image::load_from_memory(&bytes)?.to_luma8();
    let (width, height) = image.dimensions();
    let values = image.pixels().map(|pixel| normalize_luma(pixel[0])).collect::<Vec<_>>();

    Ok((values, height as usize, width as usize))
}

pub fn read_lineart_crop_values(path: impl AsRef<Path>, crop: ImageCrop) -> Result<(Vec<f32>, usize, usize)> {
    let path = path.as_ref();
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => read_png_crop_values(path, crop),
        _ => read_fully_decoded_crop_values(path, crop),
    }
}

fn read_png_crop_values(path: &Path, crop: ImageCrop) -> Result<(Vec<f32>, usize, usize)> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file));
    decoder.set_transformations(Transformations::normalize_to_color8());
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    let width = info.width as usize;
    let height = info.height as usize;
    if info.interlaced {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "interlaced PNGs are not supported for streaming crops: {}",
                path.display()
            ),
        )
        .into());
    }
    validate_crop(crop, width, height, path)?;

    let (color_type, bit_depth) = reader.output_color_type();
    if bit_depth != BitDepth::Eight {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("PNG decoder did not produce 8-bit samples for {}", path.display()),
        )
        .into());
    }

    let mut values = Vec::with_capacity(crop.size * crop.size);
    let crop_bottom = crop.y + crop.size;
    for row_index in 0..crop_bottom {
        let row = reader.next_row()?.ok_or_else(|| {
            Error::new(
                ErrorKind::UnexpectedEof,
                format!("PNG ended before row {row_index}: {}", path.display()),
            )
        })?;
        if row_index >= crop.y {
            append_png_crop_row(&mut values, row.data(), color_type, crop, path)?;
        }
    }

    Ok((values, crop.size, crop.size))
}

fn append_png_crop_row(
    values: &mut Vec<f32>,
    row: &[u8],
    color_type: PngColorType,
    crop: ImageCrop,
    path: &Path,
) -> Result<()> {
    let channels = color_type.samples();
    let start = crop.x.checked_mul(channels).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("crop byte offset overflowed for {}", path.display()),
        )
    })?;
    let end = (crop.x + crop.size).checked_mul(channels).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("crop byte offset overflowed for {}", path.display()),
        )
    })?;
    let pixels = row.get(start..end).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            format!("decoded PNG row is shorter than expected: {}", path.display()),
        )
    })?;

    match color_type {
        PngColorType::Grayscale => values.extend(pixels.iter().map(|value| normalize_luma(*value))),
        PngColorType::GrayscaleAlpha => values.extend(pixels.chunks_exact(2).map(|pixel| normalize_luma(pixel[0]))),
        PngColorType::Rgb => values.extend(
            pixels
                .chunks_exact(3)
                .map(|pixel| normalize_luma(Rgb([pixel[0], pixel[1], pixel[2]]).to_luma()[0])),
        ),
        PngColorType::Rgba => values.extend(
            pixels
                .chunks_exact(4)
                .map(|pixel| normalize_luma(Rgb([pixel[0], pixel[1], pixel[2]]).to_luma()[0])),
        ),
        PngColorType::Indexed => {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("PNG palette expansion failed for {}", path.display()),
            )
            .into());
        }
    }

    Ok(())
}

fn read_fully_decoded_crop_values(path: &Path, crop: ImageCrop) -> Result<(Vec<f32>, usize, usize)> {
    let image = image::open(path)?.to_luma8();
    let (width, height) = image.dimensions();
    validate_crop(crop, width as usize, height as usize, path)?;

    let mut values = Vec::with_capacity(crop.size * crop.size);
    for y in crop.y..crop.y + crop.size {
        for x in crop.x..crop.x + crop.size {
            values.push(normalize_luma(image.get_pixel(x as u32, y as u32)[0]));
        }
    }

    Ok((values, crop.size, crop.size))
}

fn validate_crop(crop: ImageCrop, width: usize, height: usize, path: &Path) -> Result<()> {
    let right = crop.x.checked_add(crop.size);
    let bottom = crop.y.checked_add(crop.size);
    if crop.size == 0 || right.is_none_or(|right| right > width) || bottom.is_none_or(|bottom| bottom > height) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "crop x={}, y={}, size={} is outside {}x{} image {}",
                crop.x,
                crop.y,
                crop.size,
                width,
                height,
                path.display()
            ),
        )
        .into());
    }
    Ok(())
}

fn normalize_luma(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    ((value * 2.0 - 1.0) * IMAGE_SCALE).clamp(-1.0, 1.0)
}

pub fn write_lineart_image(tensor: LineartTensor, writer: impl Write) -> Result<()> {
    let data = tensor.try_into_data()?;
    let shape = data.shape.clone();
    if shape.len() != 2 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("expected a 2D image tensor, got shape {shape:?}"),
        )
        .into());
    }

    let height = shape[0];
    let width = shape[1];
    let pixels = data
        .into_vec::<f32>()?
        .into_iter()
        .map(|value| (((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0).round() as u8)
        .collect::<Vec<_>>();

    let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(width as u32, height as u32, pixels)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "tensor data length does not match shape"))?;

    PngEncoder::new(writer).write_image(image.as_raw(), width as u32, height as u32, ColorType::L8.into())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_FILE: AtomicU64 = AtomicU64::new(0);

    struct TestFile(PathBuf);

    impl TestFile {
        fn new(extension: &str) -> Self {
            Self(std::env::temp_dir().join(format!(
                "lineartgen-image-io-test-{}-{}.{}",
                std::process::id(),
                NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed),
                extension
            )))
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    use std::path::PathBuf;

    #[test]
    fn streams_exact_rgb_png_crop() {
        let file = TestFile::new("png");
        let pixels = (0..12)
            .flat_map(|index| {
                let value = index as u8 * 20;
                [value, value.saturating_add(5), value.saturating_add(10)]
            })
            .collect::<Vec<_>>();
        PngEncoder::new(File::create(&file.0).unwrap())
            .write_image(&pixels, 4, 3, ColorType::Rgb8.into())
            .unwrap();

        let (values, height, width) = read_lineart_crop_values(&file.0, ImageCrop { x: 1, y: 1, size: 2 }).unwrap();
        let expected = [5usize, 6, 9, 10]
            .into_iter()
            .map(|index| {
                let offset = index * 3;
                normalize_luma(Rgb([pixels[offset], pixels[offset + 1], pixels[offset + 2]]).to_luma()[0])
            })
            .collect::<Vec<_>>();

        assert_eq!((height, width), (2, 2));
        assert_eq!(values, expected);
    }

    #[test]
    fn streams_exact_grayscale_png_crop() {
        let file = TestFile::new("png");
        let pixels = (0..16).map(|index| index as u8 * 16).collect::<Vec<_>>();
        PngEncoder::new(File::create(&file.0).unwrap())
            .write_image(&pixels, 4, 4, ColorType::L8.into())
            .unwrap();

        let (values, height, width) = read_lineart_crop_values(&file.0, ImageCrop { x: 2, y: 1, size: 2 }).unwrap();
        let expected = [6usize, 7, 10, 11]
            .into_iter()
            .map(|index| normalize_luma(pixels[index]))
            .collect::<Vec<_>>();

        assert_eq!((height, width), (2, 2));
        assert_eq!(values, expected);
    }

    #[test]
    fn jpeg_crop_uses_full_decode_fallback() {
        let file = TestFile::new("jpg");
        let pixels = (0..16).map(|index| index as u8 * 16).collect::<Vec<_>>();
        JpegEncoder::new_with_quality(File::create(&file.0).unwrap(), 100)
            .encode(&pixels, 4, 4, ColorType::L8.into())
            .unwrap();

        let (values, height, width) = read_lineart_crop_values(&file.0, ImageCrop { x: 1, y: 1, size: 2 }).unwrap();

        assert_eq!((height, width), (2, 2));
        assert_eq!(values.len(), 4);
        assert!(values.iter().all(|value| (-1.0..=1.0).contains(value)));
    }
}
