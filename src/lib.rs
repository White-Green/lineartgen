pub mod inference;
pub mod model;
pub mod recursion;

#[cfg(feature = "training")]
pub mod data;
#[cfg(feature = "training")]
pub mod image_io;
#[cfg(feature = "training")]
pub mod train;

#[cfg(feature = "training")]
pub type BurnBackend = burn::backend::Vulkan;
#[cfg(feature = "training")]
pub type LineartTensor = burn::tensor::Tensor<BurnBackend, 2>;
