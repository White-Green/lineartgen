#![recursion_limit = "256"]

use burn::tensor::backend::BackendTypes;
use lineartgen::inference::{InferenceError, InferenceOptions, InferenceSession};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

#[cfg(all(feature = "wgpu", any(feature = "flex", feature = "cube-cpu")))]
compile_error!("Select wgpu or a CPU backend; use --no-default-features for CPU builds.");
#[cfg(not(any(feature = "wgpu", feature = "flex", feature = "cube-cpu")))]
compile_error!("Enable a native backend: wgpu, flex, or cube-cpu.");

#[cfg(feature = "wgpu")]
type NativeBackend = burn::backend::Wgpu;
#[cfg(feature = "cube-cpu")]
type NativeBackend = burn::backend::Cpu;
#[cfg(all(feature = "flex", not(feature = "cube-cpu")))]
type NativeBackend = burn::backend::Flex;

const BACKEND_NAME: &str = if cfg!(feature = "wgpu") {
    "wgpu"
} else if cfg!(feature = "cube-cpu") {
    "cube-cpu"
} else {
    "flex"
};

const MODEL_MANIFEST: &[u8] = include_bytes!("../assets/manifest.json");
const MODEL_CHECKPOINT: &[u8] = include_bytes!("../assets/model.mpk");

#[pyclass]
pub struct LineartModel {
    session: Mutex<InferenceSession<NativeBackend>>,
    thread_pool: ThreadPool,
    #[pyo3(get)]
    device: String,
}

#[pymethods]
impl LineartModel {
    #[new]
    #[pyo3(signature = (num_threads=None))]
    fn new(py: Python<'_>, num_threads: Option<usize>) -> PyResult<Self> {
        let num_threads = num_threads.unwrap_or_else(default_thread_count);
        if num_threads == 0 {
            return Err(PyValueError::new_err("num_threads must be greater than zero"));
        }
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("lineartgen-{index}"))
            .build()
            .map_err(|error| PyRuntimeError::new_err(format!("could not create inference thread pool: {error}")))?;
        let (session, device) = py
            .detach(|| {
                thread_pool.install(|| {
                    catch_backend_panic(|| {
                        let (device, description) = native_device()?;
                        let session =
                            InferenceSession::<NativeBackend>::from_embedded(MODEL_MANIFEST, MODEL_CHECKPOINT, device)?;
                        Ok((session, description))
                    })
                })
            })
            .map_err(inference_error_to_python)?;

        Ok(Self {
            session: Mutex::new(session),
            thread_pool,
            device,
        })
    }

    #[getter]
    fn backend(&self) -> &'static str {
        BACKEND_NAME
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
                catch_backend_panic(|| {
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
            })
            .map_err(inference_error_to_python)?;

        Ok(PyBytes::new(py, &output))
    }
}

fn default_thread_count() -> usize {
    if cfg!(feature = "wgpu") {
        return 1;
    }
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .saturating_sub(1)
        .max(1)
}

type NativeDevice = <NativeBackend as BackendTypes>::Device;

#[cfg(feature = "wgpu")]
fn native_device() -> Result<(NativeDevice, String), InferenceError> {
    use burn::backend::wgpu;
    use std::sync::OnceLock;

    // Share one initialized adapter across models, including multiple Krita docks.
    static DEVICE: OnceLock<Result<(NativeDevice, String), InferenceError>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            catch_backend_panic(|| {
                let device = NativeDevice::default();
                #[cfg(target_os = "windows")]
                type GraphicsApi = wgpu::graphics::Dx12;
                #[cfg(not(target_os = "windows"))]
                type GraphicsApi = wgpu::graphics::AutoGraphicsApi;
                let setup = wgpu::init_setup::<GraphicsApi>(&device, Default::default());
                let info = setup.adapter.get_info();
                let description = format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend);
                Ok((device, description))
            })
        })
        .clone()
}

#[cfg(not(feature = "wgpu"))]
fn native_device() -> Result<(NativeDevice, String), InferenceError> {
    Ok((Default::default(), "CPU".to_string()))
}

fn catch_backend_panic<T>(operation: impl FnOnce() -> Result<T, InferenceError>) -> Result<T, InferenceError> {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or_else(|error| {
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("unknown backend failure");
        Err(InferenceError::Backend(format!(
            "{BACKEND_NAME}: {message}. Check your graphics driver and restart Krita before retrying."
        )))
    })
}

fn inference_error_to_python(error: InferenceError) -> PyErr {
    match error {
        InferenceError::InvalidArgument(message) => PyValueError::new_err(message),
        InferenceError::Backend(message) | InferenceError::Model(message) => PyRuntimeError::new_err(message),
    }
}

#[pymodule]
fn lineartgen_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("BACKEND", BACKEND_NAME)?;
    module.add_class::<LineartModel>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use std::time::Instant;

    fn session() -> InferenceSession<NativeBackend> {
        let (device, _) = native_device().unwrap();
        InferenceSession::from_embedded(MODEL_MANIFEST, MODEL_CHECKPOINT, device).unwrap()
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
    #[ignore = "manual release-mode native backend benchmark"]
    fn benchmark_embedded_model() {
        benchmark_session(BACKEND_NAME, session());
    }

    #[test]
    fn backend_panics_become_reportable_errors() {
        let error = catch_backend_panic::<()>(|| panic!("adapter initialization failed")).unwrap_err();
        assert!(matches!(error, InferenceError::Backend(_)));
        assert!(error.to_string().contains("adapter initialization failed"));
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
