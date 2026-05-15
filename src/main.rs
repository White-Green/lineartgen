use clap::{Args, Parser, Subcommand};
use image_io::{Result, read_lineart_image, write_lineart_image};
use std::fs;
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
mod data;
mod image_io;
#[allow(dead_code)]
mod model;
mod train;

pub type BurnBackend = burn::backend::Wgpu;
pub type LineartTensor = burn::tensor::Tensor<BurnBackend, 2>;

const INPUT_DIR: &str = "dataset/images";
const OUTPUT_DIR: &str = "tmp/images";

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train(TrainArgs),
    Roundtrip,
}

#[derive(Args, Debug)]
struct TrainArgs {
    #[arg(long, value_name = "EPOCH")]
    resume: Option<NonZeroUsize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Train(args)) => {
            let mut config = train::TrainingConfig::new();
            config.resume_epoch = args.resume.map(NonZeroUsize::get);
            train::train_diffusion_with_config(config)
        }
        Some(Command::Roundtrip) | None => roundtrip_images(),
    }
}

fn roundtrip_images() -> Result<()> {
    fs::create_dir_all(OUTPUT_DIR)?;

    let mut input_paths = fs::read_dir(INPUT_DIR)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    input_paths.sort();

    let mut count = 0usize;
    for input_path in input_paths.into_iter().filter(|path| is_image_file(path)) {
        let output_path = output_path_for(&input_path)?;

        let input = File::open(&input_path)?;
        let tensor = read_lineart_image(input)?;

        let output = File::create(&output_path)?;
        write_lineart_image(tensor, output)?;

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
