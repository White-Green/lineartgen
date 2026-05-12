use image_io::{Result, read_line_art_image, write_line_art_image};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

mod image_io;

pub type BurnBackend = burn::backend::Wgpu;
pub type LineArtTensor = burn::tensor::Tensor<BurnBackend, 2>;

const INPUT_DIR: &str = "dataset/images";
const OUTPUT_DIR: &str = "tmp/images";

fn main() -> Result<()> {
    fs::create_dir_all(OUTPUT_DIR)?;

    let mut input_paths = fs::read_dir(INPUT_DIR)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    input_paths.sort();

    let mut count = 0usize;
    for input_path in input_paths.into_iter().filter(|path| is_image_file(path)) {
        let output_path = output_path_for(&input_path)?;

        let input = File::open(&input_path)?;
        let tensor = read_line_art_image(input)?;

        let output = File::create(&output_path)?;
        write_line_art_image(tensor, output)?;

        count += 1;
    }

    println!("wrote {count} images to {OUTPUT_DIR}");
    Ok(())
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| matches!(extension.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
        .unwrap_or(false)
}

fn output_path_for(input_path: &Path) -> Result<PathBuf> {
    let file_name = input_path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("input path has no file name: {}", input_path.display()),
        )
    })?;

    Ok(Path::new(OUTPUT_DIR).join(file_name))
}
