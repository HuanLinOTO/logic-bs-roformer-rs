//! lbrr — BS-RoFormer six-stem music separation, pure Rust + cuda-oxide.
//!
//! CLI:
//!   lbrr --model-dir assets --input song.wav --outdir out/
//!   lbrr --self-test          # vecadd smoke test on the local GPU
//!   lbrr --print-config       # parse and dump the model YAML

mod audio;
mod config;

use std::path::PathBuf;

#[derive(Debug, Default)]
struct Args {
    model_dir: Option<PathBuf>,
    input: Option<PathBuf>,
    outdir: Option<PathBuf>,
    device: usize,
    self_test: bool,
    print_config: bool,
    bench: bool,
    stems: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut need = |name: &str| -> Result<String, String> {
            it.next().ok_or_else(|| format!("missing value for {name}"))
        };
        match a.as_str() {
            "--model-dir" => args.model_dir = Some(PathBuf::from(need("--model-dir")?)),
            "--input" => args.input = Some(PathBuf::from(need("--input")?)),
            "--outdir" => args.outdir = Some(PathBuf::from(need("--outdir")?)),
            "--device" => args.device = need("--device")?.parse().map_err(|_| "bad --device")?,
            "--stems" => args.stems = Some(need("--stems")?.parse().map_err(|_| "bad --stems")?),
            "--self-test" => args.self_test = true,
            "--print-config" => args.print_config = true,
            "--bench" => args.bench = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(args)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("lbrr: {e}");
            eprintln!("usage: lbrr --model-dir DIR --input song.wav --outdir out/ [--device N] [--bench]");
            eprintln!("       lbrr --self-test | --print-config --model-dir DIR");
            std::process::exit(2);
        }
    };

    if args.self_test {
        self_test(args.device);
        return;
    }

    let model_dir = args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets"));
    let yaml_path = model_dir.join("logic_bs_roformer.yaml");
    let yaml_text = match std::fs::read_to_string(&yaml_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("lbrr: cannot read {}: {e}", yaml_path.display());
            std::process::exit(1);
        }
    };
    let cfg = match config::ModelConfig::parse(&yaml_text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("lbrr: bad config: {e}");
            std::process::exit(1);
        }
    };

    if args.print_config {
        println!("{cfg:#?}");
        return;
    }

    let Some(input) = args.input.clone() else {
        eprintln!("lbrr: --input required for inference");
        std::process::exit(2);
    };
    let wav = match audio::read_wav(&input) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("lbrr: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "input: {} Hz, {} ch, {} frames ({:.1}s)",
        wav.sample_rate,
        wav.channels,
        wav.samples.len() / wav.channels,
        wav.samples.len() as f64 / wav.channels as f64 / wav.sample_rate as f64
    );
    println!("model: dim {} depth {} bands {} stems {:?}", cfg.dim, cfg.depth, cfg.num_bands(), cfg.instruments);
    eprintln!("lbrr: inference not implemented yet (Phase 3)");
}

// ---------------------------------------------------------------------------
// Toolchain smoke test: vecadd on the local GPU.
// ---------------------------------------------------------------------------

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// c[i] = a[i] + b[i] — toolchain smoke test.
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = c.get_mut(idx) {
            *out = a[i] + b[i];
        }
    }
}

fn self_test(device: usize) {
    let ctx = CudaContext::new(device).expect("create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1 << 20;
    let a: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32).sin()).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("load embedded module");
    // SAFETY: 1-D launch, one thread per output element, buffers all length N.
    unsafe {
        module.vecadd(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }
    .expect("launch vecadd");

    let c = c_dev.to_host_vec(&stream).unwrap();
    let max_err = (0..N)
        .map(|i| (c[i] - (a[i] + b[i])).abs())
        .fold(0.0f32, f32::max);
    println!("lbrr self-test: vecadd N={N} max_err={max_err:e}");
    assert!(max_err < 1e-6, "vecadd mismatch");
    println!("OK");
}
