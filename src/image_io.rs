use crate::{BurnBackend, LineartTensor};
use burn::tensor::{Tensor, TensorData};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageBuffer, ImageEncoder, Luma};
use std::io::{Error, ErrorKind, Read, Write};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const IMAGE_SCALE: f32 = 1.1;

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
    let values = image
        .pixels()
        .map(|pixel| {
            let value = f32::from(pixel[0]) / 255.0;
            ((value * 2.0 - 1.0) * IMAGE_SCALE).clamp(-1.0, 1.0)
        })
        .collect::<Vec<_>>();

    Ok((values, height as usize, width as usize))
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
