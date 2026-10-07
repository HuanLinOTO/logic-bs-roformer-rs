//! Exe-relative runtime layout probing: portable bundles ship every
//! dynamically loaded library next to the executable, so loaders probe
//! that directory ahead of toolkit paths and bare loader fallbacks.

/// Directory holding the running executable, if determinable.
pub fn exe_dir() -> Option<std::path::PathBuf> {
    std::env::current_exe().ok()?.parent().map(|p| p.to_path_buf())
}

/// Exe directory as a display string (forward slashes, matching the
/// CUDA toolchain path convention used across this crate).
pub fn exe_dir_str() -> Option<String> {
    exe_dir().map(|d| d.display().to_string())
}
