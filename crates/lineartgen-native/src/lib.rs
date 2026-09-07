use lineartgen::inference::{InferenceError, InferenceOptions, InferenceSession};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Mutex;

#[cfg(feature = "cube-cpu")]
type NativeBackend = burn::backend::Cpu;
#[cfg(not(feature = "cube-cpu"))]
type NativeBackend = burn::backend::Flex;

const MODEL_MANIFEST: &[u8] = include_bytes!("../assets/manifest.json");
const MODEL_CHECKPOINT: &[u8] = include_bytes!("../assets/model.mpk");

#[pyclass]
pub struct LineartModel {
    session: Mutex<InferenceSession<NativeBackend>>,
    thread_pool: ThreadPool,
}

#[pymethods]
impl LineartModel {
    #[new]
    #[pyo3(signature = (num_threads=None))]
    fn new(num_threads: Option<usize>) -> PyResult<Self> {
        let num_threads = num_threads.unwrap_or_else(default_thread_count);
        if num_threads == 0 {
            return Err(PyValueError::new_err("num_threads must be greater than zero"));
        }
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("lineartgen-{index}"))
            .build()
            .map_err(|error| PyRuntimeError::new_err(format!("could not create CPU thread pool: {error}")))?;
        let session = thread_pool
            .install(|| {
                InferenceSession::<NativeBackend>::from_embedded(MODEL_MANIFEST, MODEL_CHECKPOINT, Default::default())
            })
            .map_err(inference_error_to_python)?;

        Ok(Self {
            session: Mutex::new(session),
            thread_pool,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (scribble, lineart, width, height, strength, seed, denoise_steps))]
    fn infer<'py>(
        &self,
        py: Python<'py>,
        scribble: PyBackedBytes,
        lineart: PyBackedBytes,
        width: usize,
        height: usize,
        strength: f32,
        seed: u64,
        denoise_steps: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let output = py
            .detach(|| {
                self.thread_pool.install(|| {
                    let session = self
                        .session
                        .lock()
                        .map_err(|_| InferenceError::Backend("inference lock was poisoned".to_string()))?;
                    session.infer_bgra(
                        scribble.as_ref(),
                        lineart.as_ref(),
                        width,
                        height,
                        InferenceOptions {
                            strength,
                            seed,
                            denoising_steps: denoise_steps,
                        },
                    )
                })
            })
            .map_err(inference_error_to_python)?;

        Ok(PyBytes::new(py, &output))
    }
}

fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .saturating_sub(1)
        .max(1)
}

fn inference_error_to_python(error: InferenceError) -> PyErr {
    match error {
        InferenceError::InvalidArgument(message) => PyValueError::new_err(message),
        InferenceError::Backend(message) | InferenceError::Model(message) => PyRuntimeError::new_err(message),
    }
}

#[pymodule]
fn lineartgen_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LineartModel>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use std::time::Instant;

    fn session() -> InferenceSession<NativeBackend> {
        InferenceSession::from_embedded(MODEL_MANIFEST, MODEL_CHECKPOINT, Default::default()).unwrap()
    }

    fn inputs(size: usize) -> (Vec<u8>, Vec<u8>) {
        let mut scribble = vec![255; size * size * 4];
        let lineart = vec![0; size * size * 4];
        for pixel in scribble.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        (scribble, lineart)
    }

    #[test]
    fn embedded_model_loads_and_infers() {
        let session = session();
        let (scribble, lineart) = inputs(16);

        let output = session
            .infer_bgra(
                &scribble,
                &lineart,
                16,
                16,
                InferenceOptions {
                    strength: 0.5,
                    seed: 42,
                    denoising_steps: 1,
                },
            )
            .unwrap();

        assert_eq!(output.len(), 16 * 16 * 4);
        assert!(output.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]));
    }

    #[test]
    fn fixed_seed_is_reproducible() {
        let session = session();
        let (scribble, lineart) = inputs(16);
        let options = InferenceOptions {
            strength: 1.0,
            seed: 7,
            denoising_steps: 1,
        };

        let first = session.infer_bgra(&scribble, &lineart, 16, 16, options).unwrap();
        let second = session.infer_bgra(&scribble, &lineart, 16, 16, options).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    #[ignore = "manual release-mode CPU benchmark"]
    fn benchmark_embedded_model() {
        let backend = if cfg!(feature = "cube-cpu") { "cube-cpu" } else { "flex" };
        benchmark_session(backend, session());
    }

    #[cfg(feature = "vulkan-benchmark")]
    #[test]
    #[ignore = "manual release-mode Vulkan benchmark"]
    fn benchmark_vulkan_model() {
        let session = InferenceSession::<burn::backend::Vulkan>::from_embedded(
            MODEL_MANIFEST,
            MODEL_CHECKPOINT,
            Default::default(),
        )
        .unwrap();
        benchmark_session("vulkan", session);
    }

    fn benchmark_session<B: Backend>(backend: &str, session: InferenceSession<B>) {
        const WARM_ITERATIONS: u32 = 5;
        let options = InferenceOptions {
            strength: 0.5,
            seed: 42,
            denoising_steps: 1,
        };
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(default_thread_count())
            .build()
            .unwrap();

        for size in [256, 512, 1024] {
            let (scribble, lineart) = inputs(size);
            let cold_start = Instant::now();
            thread_pool
                .install(|| session.infer_bgra(&scribble, &lineart, size, size, options))
                .unwrap();
            let cold = cold_start.elapsed();
            let warm_start = Instant::now();
            for _ in 0..WARM_ITERATIONS {
                thread_pool
                    .install(|| session.infer_bgra(&scribble, &lineart, size, size, options))
                    .unwrap();
            }
            let warm = warm_start.elapsed() / WARM_ITERATIONS;
            println!("{backend} {size}x{size}: cold={cold:?}, warm_avg={warm:?}");
        }
    }
}
