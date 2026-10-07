//! Host-only benchmark contract and numerical checks. No GPU work is timed here.
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Stage {
    #[default]
    Waveform,
    Frames,
}
impl Stage {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "waveform" => Ok(Self::Waveform),
            "frames" => Ok(Self::Frames),
            _ => Err(format!("invalid --bench-stage {s}")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Waveform => "waveform",
            Self::Frames => "frames",
        }
    }
}
#[derive(Debug, Clone)]
pub struct Options {
    pub reference: Option<PathBuf>,
    pub stage: Stage,
    pub warmup: usize,
    pub json: Option<PathBuf>,
    pub output: Option<PathBuf>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            reference: None,
            stage: Stage::Waveform,
            warmup: 5,
            json: None,
            output: None,
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Metric {
    pub finite: bool,
    pub reference_rms: f64,
    pub snr_db: Option<f64>,
    pub max_abs_error: f64,
    pub bit_equal: bool,
}
pub fn metric(actual: &[f32], reference: &[f32]) -> Metric {
    assert_eq!(actual.len(), reference.len());
    assert!(!actual.is_empty());
    let (mut signal, mut noise, mut max_error) = (0.0f64, 0.0f64, 0.0f64);
    let mut finite = true;
    let mut bit_equal = true;
    for (&a, &b) in actual.iter().zip(reference) {
        finite &= a.is_finite() && b.is_finite();
        bit_equal &= a.to_bits() == b.to_bits();
        let d = a as f64 - b as f64;
        signal += (b as f64).powi(2);
        noise += d * d;
        max_error = max_error.max(d.abs());
    }
    let rms = (signal / actual.len() as f64).sqrt();
    Metric {
        finite,
        reference_rms: rms,
        snr_db: if finite && rms >= 1e-8 {
            Some(10.0 * (signal / noise.max(1e-300)).log10())
        } else {
            None
        },
        max_abs_error: max_error,
        bit_equal,
    }
}
pub fn correctness(actual: &[f32], reference: &[f32], len: usize) -> Value {
    assert_eq!(actual.len(), 12 * len);
    assert_eq!(reference.len(), actual.len());
    let stems: Vec<_> = actual
        .chunks_exact(2 * len)
        .zip(reference.chunks_exact(2 * len))
        .map(|(a, b)| metric(a, b))
        .collect();
    let all = metric(actual, reference);
    json!({ "finite": all.finite, "shape": [1,6,2,len], "stems": stems, "overall": all,
        "relative_to_b0": null, "relative_to_b0_reason": "computed by benchmark_matrix compare", "tail_covered": true })
}
pub fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    if v.len() % 2 == 0 {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) * 0.5
    } else {
        v[v.len() / 2]
    }
}
pub fn matrix_flops(t: usize) -> u64 {
    let t = t as u64;
    let m = 62 * t;
    let trunk = 24
        * (2 * m * 1536 * 256
            + 2 * m * 256 * 512
            + 2 * m * 1024 * 256
            + 2 * m * 256 * 1024
            + 2 * m * 8 * 256);
    let attn = 12 * 4 * 62 * 8 * t * t * 64 + 12 * 4 * t * 8 * 62 * 62 * 64;
    let mask = 6 * (2 * m * 1024 * 256 + 2 * t * 8200 * 1024);
    trunk + attn + mask + 2 * t * 4100 * 256
}
pub fn mfu(device: &str, t: usize, ms: f64) -> Value {
    let peak = match device {
        "NVIDIA GeForce RTX 3080" => Some(59.53536),
        "NVIDIA GeForce RTX 4060 Ti" => Some(44.12928),
        _ => None,
    };
    let flops = matrix_flops(t);
    json!({"matrix_flops": flops, "nominal_peak_tflops": peak,
        "nominal_mfu": peak.map(|p| flops as f64 / (ms * 1e9 * p)),
        "reason": if peak.is_none() { Some("no configured nominal dense FP16/FP32 peak for this device") } else { None }})
}
pub fn istft_inverse(window: &[f32], len: usize, frames: usize) -> Vec<f32> {
    let mut energy = vec![0.0f32; len + 2048];
    for t in 0..frames {
        for (n, &w) in window.iter().enumerate() {
            if t * 512 + n < energy.len() {
                energy[t * 512 + n] += w * w;
            }
        }
    }
    energy
        .into_iter()
        .map(|x| if x > 1e-8 { 1.0 / x } else { 0.0 })
        .collect()
}
/// Historical host OLA is retained as an independent check of GPU gather/crop.
pub fn host_ola(pcm: &[f32], win: &[f32], len: usize, frames: usize) -> Vec<f32> {
    let padded = len + 2048;
    let mut counter = vec![0.0f32; padded];
    let mut result = vec![0.0f32; 12 * padded];
    for t in 0..frames {
        for n in 0..2048 {
            let p = t * 512 + n;
            if p >= padded {
                continue;
            }
            counter[p] += win[n] * win[n];
            for sc in 0..12 {
                result[sc * padded + p] +=
                    pcm[(sc * frames + t) * 2048 + n] * win[n] * (1.0 / 2048.0);
            }
        }
    }
    (0..12 * len)
        .map(|i| {
            let sc = i / len;
            let p = i % len + 1024;
            if counter[p] > 1e-8 {
                result[sc * padded + p] / counter[p]
            } else {
                0.0
            }
        })
        .collect()
}
pub fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let data = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, data).map_err(|e| format!("{}: {e}", path.display()))
}
/// Use the OS SHA256 utility to keep dependencies limited to serde/serde_json.
pub fn sha256(path: &Path) -> Result<String, String> {
    let p = path.canonicalize().map_err(|e| e.to_string())?;
    let output = if cfg!(windows) {
        Command::new("certutil")
            .arg("-hashfile")
            .arg(&p)
            .arg("SHA256")
            .output()
    } else {
        Command::new("sha256sum").arg(&p).output()
    }
    .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!("hash failed for {}", p.display()));
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .find(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_owned)
        .ok_or_else(|| "SHA256 missing from utility output".into())
}
fn cuda_runtime_version() -> Value {
    let root = std::env::var("CUDA_HOME").unwrap_or_default();
    let names = if cfg!(windows) {
        vec![
            format!("{root}/bin/x64/cudart64_13.dll"),
            format!("{root}/bin/cudart64_12.dll"),
            "cudart64_13.dll".into(),
            "cudart64_12.dll".into(),
        ]
    } else {
        vec![
            format!("{root}/lib64/libcudart.so.13"),
            format!("{root}/lib64/libcudart.so.12"),
            "libcudart.so".into(),
        ]
    };
    for path in names {
        unsafe {
            if let Ok(lib) = libloading::Library::new(&path) {
                if let Ok(get) =
                    lib.get::<unsafe extern "C" fn(*mut i32) -> i32>(b"cudaRuntimeGetVersion")
                {
                    let mut version = 0;
                    if get(&mut version) == 0 {
                        return json!({"version":version,"library":path});
                    }
                }
            }
        }
    }
    json!({"version":null,"reason":"CUDA runtime version symbol unavailable"})
}

pub fn identity(
    model: &Path,
    reference: &Path,
    device: &str,
    sm: (i32, i32),
) -> Result<Value, String> {
    let source = env!("LBRR_BUILD_SOURCE_SHA256");
    let revision = option_env!("LBRR_BUILD_REVISION").filter(|s| !s.is_empty());
    let driver = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|p| p.status.success())
        .map(|p| String::from_utf8_lossy(&p.stdout).trim().to_string());
    Ok(
        json!({"source_revision":revision, "source_sha256":source, "source_hash_reason":"embedded at compilation from source, Cargo manifests and build configuration",
        "binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,
        "model_sha256":sha256(&model.join("model.safetensors"))?, "config_sha256":sha256(&model.join("logic_bs_roformer.yaml"))?,
        "reference_sha256":sha256(reference)?, "fixture_id":reference.parent().and_then(Path::file_name).map(|s|s.to_string_lossy()),
        "device":device, "sm":format!("{}.{}",sm.0,sm.1), "driver":driver,"cuda_runtime":cuda_runtime_version()}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verified_flop_counts() {
        assert_eq!(matrix_flops(1151), 6_069_010_774_016);
        assert_eq!(matrix_flops(259), 1_013_639_108_608);
    }
    #[test]
    fn silence_and_nonfinite_are_explicit() {
        let m = metric(&[0.0; 4], &[0.0; 4]);
        assert!(m.finite && m.bit_equal);
        assert!(m.snr_db.is_none());
        assert!(!metric(&[f32::NAN], &[0.0]).finite);
    }
}
