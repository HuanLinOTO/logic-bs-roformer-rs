//! All device kernels, one #[cuda_module] host per file.

#[cfg_attr(not(target_os = "none"), path = "frame.rs")]
pub mod frame;