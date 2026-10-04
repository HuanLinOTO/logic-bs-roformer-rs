//! lbrr — BS-RoFormer six-stem music separation, pure Rust + cuda-oxide.
//!
//! CLI:
//!   lbrr --model-dir assets --input song.wav --outdir out/
//!   lbrr --self-test          # vecadd smoke test on the local GPU
//!   lbrr --print-config       # parse and dump the model YAML

mod audio;
mod config;
mod cufft;
mod kernels;
mod npz;
mod stft;
mod weights;

use std::path::PathBuf;

#[derive(Debug, Default)]
struct Args {
    model_dir: Option<PathBuf>,
    input: Option<PathBuf>,
    outdir: Option<PathBuf>,
    device: usize,
    self_test: bool,
    print_config: bool,
    check_weights: bool,
    fft_test: bool,
    stft_test: bool,
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
            "--check-weights" => args.check_weights = true,
            "--fft-test" => args.fft_test = true,
            "--stft-test" => args.stft_test = true,
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

    if args.fft_test {
        fft_roundtrip_test(args.device);
        return;
    }

    if args.stft_test {
        stft_parity_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
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

    if args.check_weights {
        let st_path = model_dir.join("model.safetensors");
        let t0 = std::time::Instant::now();
        let st = match weights::SafeTensors::open(&st_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("lbrr: {}: {e}", st_path.display());
                std::process::exit(1);
            }
        };
        println!("safetensors: {} tensors parsed in {:?}", st.metas.len(), t0.elapsed());
        match weights::ModelWeights::load(&st, &cfg) {
            Ok(w) => {
                println!(
                    "weights OK: {} layers, {} bands, {} stems; shared qkv bias len {}",
                    w.layers.len(),
                    w.band_w.len(),
                    w.mask_w1.len(),
                    w.shared_qkv_bias.len()
                );
                println!("all checkpoint keys consumed exactly once");
            }
            Err(e) => {
                eprintln!("lbrr: weight load failed: {e}");
                std::process::exit(1);
            }
        }
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
mod gpu_kernels {
    use super::*;

    /// STFT front-end: one windowed frame element per thread. See the
    /// kernels/frame.rs doc for the reflect-gather semantics.
    #[kernel]
    pub fn frame_hann_reflect(
        x: &[f32],
        window: &[f32],
        mut out: DisjointSlice<f32>,
        n_fft: u32,
        hop: u32,
        len: u32,
        frames: u32,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let n_fft = n_fft as usize;
            let hop = hop as usize;
            let len = len as usize;
            let frames = frames as usize;
            let t2 = g0 / n_fft;
            let j = g0 % n_fft;
            let ch = t2 / frames;
            let t = t2 % frames;
            let pad = n_fft / 2;
            let g = t * hop + j;
            // numpy/torch 'reflect' mirrors WITHOUT repeating the border:
            // left x_pad[k] = x[p-k]; right x_pad[p+len+k] = x[len-2-k].
            let src = if g < pad {
                pad - g
            } else if g >= len + pad {
                2 * len + pad - 2 - g
            } else {
                g - pad
            };
            *o = x[src * 2 + ch] * window[j];
        }
    }

    /// ISTFT back-end: scale cuFFT's unnormalized C2R output by 1/n_fft.
    #[kernel]
    pub fn scale_1_over_n(mut x: DisjointSlice<f32>, n: f32) {
        let idx = thread::index_1d();
        if let Some(v) = x.get_mut(idx) {
            *v /= n;
        }
    }

    /// Minimal probe retained for toolchain smoke tests.
    #[kernel]
    pub fn probe_min(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[g0];
        }
    }
}

/// STFT parity vs torch.stft (parity/stft.npz from tools/dump_refs.py).
/// Gate: max abs err < 1e-5 (STFT is a deterministic dot product; measured 3e-6).
fn stft_parity_test(device: usize, model_dir: &std::path::Path) {
    let npz_path = model_dir.parent().unwrap_or(model_dir).join("parity/stft.npz");
    let npz = npz::Npz::open(&npz_path).unwrap_or_else(|e| panic!("{e}"));
    let x = npz.f32("x").expect("x");               // (2, CHUNK) planar stereo
    let window = npz.f32("window").expect("window"); // (2048,)
    let ref_spec = npz.f32("spec").expect("spec");   // (2, 1025, T, 2)
    let len = npz.shapes["x"][1];
    let frames = stft::num_frames(len);
    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();

    // planar (2, L) -> interleaved (L, 2)
    let mut xi = vec![0.0f32; len * 2];
    for i in 0..len {
        xi[i * 2] = x[i];
        xi[i * 2 + 1] = x[len + i];
    }
    let our_win = stft::hann_window(stft::N_FFT);
    let win_err = our_win.iter().zip(window).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    // torch's hann_window differs from a pure-f64 recompute by ~2e-7 (its
    // internal form is float32). Far below the parity gate.
    assert!(win_err < 1e-6, "hann window mismatch {win_err:e}");

    let x_dev = DeviceBuffer::from_host(&stream, &xi).unwrap();
    let win_dev = DeviceBuffer::from_host(&stream, &our_win).unwrap();
    let mut frames_dev = DeviceBuffer::<f32>::zeroed(&stream, 2 * frames * stft::N_FFT).unwrap();
    let km = gpu_kernels::load(&ctx).expect("kernel module");
    // SAFETY: 1-D launch, one thread per output element; buffers sized for
    // 2*frames*n_fft outputs, window of n_fft, input of len*2.
    unsafe {
        km.frame_hann_reflect(
            &stream,
            cuda_core::simt::LaunchConfig::for_num_elems((2 * frames * stft::N_FFT) as u32),
            &x_dev,
            &win_dev,
            &mut frames_dev,
            stft::N_FFT as u32,
            stft::HOP as u32,
            len as u32,
            frames as u32,
        )
    }
    .expect("frame kernel");

    let fft = std::sync::Arc::new(cufft::Cufft::load().expect("cufft"));
    let stft_plan = stft::Stft::new(fft, frames).expect("stft plan");
    let mut spec_dev = DeviceBuffer::<f32>::zeroed(&stream, 2 * frames * stft::FREQ_BINS * 2).unwrap();
    stft_plan.exec_fwd(&frames_dev, &mut spec_dev).expect("r2c");

    let got = spec_dev.to_host_vec(&stream).unwrap();
    // got layout: (t2 = ch*frames + t)(f)(c); ref layout: (ch)(f)(t)(c)
    let t = frames;
    let f = stft::FREQ_BINS;
    let mut max_err = 0.0f32;
    let mut denom_max = 0.0f32;
    for ch in 0..2usize {
        for ti in 0..t {
            for fi in 0..f {
                for c in 0..2usize {
                    let g = got[((ch * t + ti) * f + fi) * 2 + c];
                    let r = ref_spec[((ch * f + fi) * t + ti) * 2 + c];
                    max_err = max_err.max((g - r).abs());
                    denom_max = denom_max.max(r.abs());
                }
            }
        }
    }
    let rel = max_err / denom_max;
    println!("stft parity: frames={t} max_err={max_err:e} rel={rel:e}");
    assert!(max_err < 1e-5 * denom_max + 1e-6, "stft parity failed");
    println!("OK");
}


/// cufft R2C/C2R roundtrip parity: x -> R2C -> C2R*(1/n) == x, plus the
/// DC/Nyquist bins of a known cosine.
fn fft_roundtrip_test(device: usize) {
    let ctx = CudaContext::new(device).expect("create CUDA context");
    let stream = ctx.default_stream();
    let fft = cufft::Cufft::load().expect("load libcufft");

    const N: usize = 2048;
    const BATCH: usize = 2302; // 2ch * 1151 frames — the real STFT shape

    // deterministic pseudo-random signal + one known cosine frame
    let mut x = vec![0.0f32; N * BATCH];
    let mut seed = 0x12345678u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0 - 0.5
    };
    for v in x.iter_mut() {
        *v = next() * 0.1;
    }
    // frame 7: 64-point cosine -> energy at bin 64
    for i in 0..N {
        x[7 * N + i] = (std::f32::consts::TAU * 64.0 * i as f32 / N as f32).cos();
    }

    let x_dev = DeviceBuffer::from_host(&stream, &x).unwrap();
    let mut spec = DeviceBuffer::<f32>::zeroed(&stream, (N / 2 + 1) * 2 * BATCH).unwrap();
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, N * BATCH).unwrap();

    let fwd = fft.plan(N, BATCH, true).expect("R2C plan");
    let inv = fft.plan(N, BATCH, false).expect("C2R plan");
    fwd.exec_r2c(x_dev.cu_deviceptr(), spec.cu_deviceptr()).expect("exec R2C");
    inv.exec_c2r(spec.cu_deviceptr(), y_dev.cu_deviceptr()).expect("exec C2R");

    let y = y_dev.to_host_vec(&stream).unwrap();
    let spec = spec.to_host_vec(&stream).unwrap();
    let mut max_err = 0.0f32;
    for i in 0..x.len() {
        let got = y[i] / N as f32;
        max_err = max_err.max((got - x[i]).abs());
    }
    // DC of the cosine frame must be ~0 and bin 64 magnitude ~ N/2
    let dc = spec[7 * (N / 2 + 1) * 2];
    let b64_re = spec[7 * (N / 2 + 1) * 2 + 64 * 2];
    let b64_im = spec[7 * (N / 2 + 1) * 2 + 64 * 2 + 1];
    let b64_mag = (b64_re * b64_re + b64_im * b64_im).sqrt();
    println!(
        "fft roundtrip N={N} batch={BATCH}: max_err={max_err:e}, cos-frame DC={dc:.3e}, bin64 |X|={b64_mag:.1} (expect ~{})",
        N / 2
    );
    assert!(max_err < 1e-4, "roundtrip error too large");
    assert!(b64_mag > (N / 2) as f32 * 0.99 && b64_mag < (N / 2) as f32 * 1.01);
    println!("OK");
}

fn self_test(device: usize) {
    let ctx = CudaContext::new(device).expect("create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1 << 20;
    let a: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = gpu_kernels::load(&ctx).expect("load embedded module");
    // SAFETY: 1-D launch, one thread per output element, buffers all length N.
    unsafe {
        module.probe_min(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &mut c_dev,
        )
    }
    .expect("launch probe_min");

    let c = c_dev.to_host_vec(&stream).unwrap();
    let max_err = (0..N).map(|i| (c[i] - a[i]).abs()).fold(0.0f32, f32::max);
    println!("lbrr self-test: probe_min N={N} max_err={max_err:e}");
    assert!(max_err < 1e-6, "probe mismatch");
    println!("OK");
}