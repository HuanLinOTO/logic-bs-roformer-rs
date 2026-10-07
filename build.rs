//! Embed the exact build-source digest, including builds from rsync snapshots.
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};
fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("source directory") {
        let p = entry.unwrap().path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|s| s == "rs") {
            out.push(p);
        }
    }
}
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let revision = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned())
        .unwrap_or_default();
    println!("cargo:rustc-env=LBRR_BUILD_REVISION={revision}");
    let mut paths = Vec::new();
    collect(&root.join("src"), &mut paths);
    for p in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "rust-toolchain.toml",
        ".cargo/config.toml",
    ] {
        let p = root.join(p);
        if p.exists() {
            paths.push(p);
        }
    }
    paths.sort();
    let mut snapshot = Vec::new();
    for p in paths {
        println!("cargo:rerun-if-changed={}", p.display());
        let name = p
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let data = fs::read(&p).unwrap();
        snapshot.extend_from_slice(name.as_bytes());
        snapshot.push(0);
        snapshot.extend_from_slice(&(data.len() as u64).to_le_bytes());
        snapshot.extend_from_slice(&data);
    }
    let file = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("source.snapshot");
    fs::write(&file, snapshot).unwrap();
    let result = if cfg!(windows) {
        Command::new("certutil")
            .arg("-hashfile")
            .arg(&file)
            .arg("SHA256")
            .output()
    } else {
        Command::new("sha256sum").arg(&file).output()
    }
    .expect("OS SHA256 utility");
    assert!(result.status.success(), "cannot hash build source");
    let text = String::from_utf8_lossy(&result.stdout);
    let hash = text
        .split_whitespace()
        .find(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .expect("SHA256 output");
    println!("cargo:rustc-env=LBRR_BUILD_SOURCE_SHA256={hash}");
}
