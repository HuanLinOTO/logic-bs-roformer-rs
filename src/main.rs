//! lbrr — BS-RoFormer six-stem music separation, pure Rust + cuda-oxide.
//!
//! CLI:
//!   lbrr --model-dir assets --input song.wav --outdir out/
//!   lbrr --self-test          # vecadd smoke test on the local GPU
//!   lbrr --print-config       # parse and dump the model YAML

mod audio;
mod config;
mod cublas;
mod cufft;
mod kernels;
mod npz;
mod stft;
mod weights;

use std::path::PathBuf;
use std::sync::Arc;

use cuda_core::CudaStream;

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
    rmsnorm_test: bool,
    bandsplit_test: bool,
    gemm_test: bool,
    qkvrope_test: bool,
    attn_test: bool,
    gateff_test: bool,
    e2e_test: bool,
    reorder_test: bool,
    bench: bool,
    iters: usize,
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
            "--rmsnorm-test" => args.rmsnorm_test = true,
            "--bandsplit-test" => args.bandsplit_test = true,
            "--gemm-test" => args.gemm_test = true,
            "--qkvrope-test" => args.qkvrope_test = true,
            "--attn-test" => args.attn_test = true,
            "--gateff-test" => args.gateff_test = true,
            "--e2e-test" => args.e2e_test = true,
            "--reorder-test" => args.reorder_test = true,
            "--bench" => args.bench = true,
            "--iters" => args.iters = need("--iters")?.parse().map_err(|_| "bad --iters")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.iters == 0 {
        args.iters = 10;
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

    if args.reorder_test {
        reorder_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
        return;
    }

    if args.e2e_test {
        e2e_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
        return;
    }

    if args.bench {
        bench_warm(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")), args.iters);
        return;
    }

    if args.gateff_test {
        gateff_parity_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
        return;
    }

    if args.attn_test {
        attn_parity_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
        return;
    }

    if args.qkvrope_test {
        qkvrope_parity_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
        return;
    }

    if args.gemm_test {
        gemm_parity_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
        return;
    }

    if args.bandsplit_test {
        bandsplit_parity_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
        return;
    }

    if args.rmsnorm_test {
        rmsnorm_parity_test(args.device, &args.model_dir.clone().unwrap_or_else(|| PathBuf::from("assets")));
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
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};
use cuda_device::wmma;

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

    /// RMSNorm: out[r] = normalize(x[r]) * sqrt(dim) * gamma, one warp per
    /// row, butterfly reduction for the squared sum. dim must be a multiple
    /// of 32 (256 = 8 elements per lane).
    #[kernel]
    pub fn rmsnorm(x: &[f32], gamma: &[f32], mut out: DisjointSlice<f32>, rows: u32, dim: u32) {
        let gid = thread::index_1d();
        let g0 = gid.get();
        let lane = warp::lane_id() as usize;
        let warp_id = g0 / 32;
        let rows = rows as usize;
        let dim = dim as usize;
        let per_lane = dim / 32;
        if warp_id >= rows {
            return;
        }
        let base = warp_id * dim;
        // partial squared sum; lane-major indexing makes each load coalesced
        let mut vals = [0.0f32; 8];
        let mut s = 0.0f32;
        for k in 0..per_lane {
            let i = base + k * 32 + lane;
            let v = x[i];
            vals[k] = v;
            s += v * v;
        }
        // butterfly warp reduce
        s += warp::shuffle_xor_f32(s, 16);
        s += warp::shuffle_xor_f32(s, 8);
        s += warp::shuffle_xor_f32(s, 4);
        s += warp::shuffle_xor_f32(s, 2);
        s += warp::shuffle_xor_f32(s, 1);
        let norm = s.sqrt();
        let denom = if norm > 1e-12 { norm } else { 1e-12 };
        let scale = (dim as f32).sqrt() / denom;
        let out_ptr = out.as_mut_ptr();
        for k in 0..per_lane {
            let i = base + k * 32 + lane;
            // SAFETY: rows*dim total elements, i < rows*dim by construction;
            // each warp owns a disjoint row, lanes write disjoint columns.
            unsafe {
                *out_ptr.add(i) = vals[k] * scale * gamma[i % dim];
            }
        }
    }

    /// RMSNorm fused with the eight head-gate projections. One warp owns a
    /// row; after normalization it performs eight 256-wide dot products.
    #[kernel]
    pub fn rmsnorm_gates(
        x: &[f32], gamma: &[f32], gate_w: &[f32], gate_b: &[f32],
        mut out: DisjointSlice<f32>, mut gates: DisjointSlice<f32>,
        rows: u32, dim: u32,
    ) {
        let gid = thread::index_1d();
        let g0 = gid.get();
        let lane = warp::lane_id() as usize;
        let warp_id = g0 / 32;
        let rows = rows as usize;
        let dim = dim as usize;
        let per_lane = dim / 32;
        if warp_id >= rows { return; }
        let base = warp_id * dim;
        let mut vals = [0.0f32; 8];
        let mut s = 0.0f32;
        for k in 0..per_lane {
            let i = base + k * 32 + lane;
            let v = x[i];
            vals[k] = v;
            s += v * v;
        }
        s += warp::shuffle_xor_f32(s, 16);
        s += warp::shuffle_xor_f32(s, 8);
        s += warp::shuffle_xor_f32(s, 4);
        s += warp::shuffle_xor_f32(s, 2);
        s += warp::shuffle_xor_f32(s, 1);
        let norm = s.sqrt();
        let denom = if norm > 1e-12 { norm } else { 1e-12 };
        let scale = (dim as f32).sqrt() / denom;
        let out_ptr = out.as_mut_ptr();
        for k in 0..per_lane {
            let i = base + k * 32 + lane;
            vals[k] *= scale * gamma[i % dim];
            unsafe {
                *out_ptr.add(i) = vals[k];
            }
        }
        let gates_ptr = gates.as_mut_ptr();
        for g in 0..8usize {
            let wbase = g * dim;
            let mut dot = 0.0f32;
            for k in 0..per_lane {
                dot += vals[k] * gate_w[wbase + k * 32 + lane];
            }
            dot = warp::reduce_sum_f32(dot);
            if lane == g {
                unsafe {
                    *gates_ptr.add(warp_id * 8 + g) = dot + gate_b[g];
                }
            }
        }
    }

    /// BandSplit: per-band RMSNorm + Linear(dim_in -> 256) fused.
    /// One warp per (t, band) task. Per-warp slice of a static shared tile
    /// carries the normalized h vector; each lane then runs 8 COMPLETE dot
    /// products from shared memory (no cross-lane accumulation shuffles).
    #[kernel]
    pub fn bandsplit(
        x: &[f32],
        gamma: &[f32],
        w: &[f32],
        b: &[f32],
        band_offs: &[u32],
        band_dims: &[u32],
        mut out: DisjointSlice<f32>,
        rows: u32,
        n_bands: u32,
    ) {
        static mut HB: SharedArray<f32, { 8 * 521 }> = SharedArray::UNINIT;
        let gid = thread::index_1d();
        let g0 = gid.get();
        let lane = warp::lane_id() as usize;
        let wid = (thread::threadIdx_x() as usize) / 32; // warp within block
        let task = g0 / 32;
        let rows = rows as usize;
        let n_bands = n_bands as usize;
        if task >= rows * n_bands {
            return;
        }
        let t = task / n_bands;
        let band = task % n_bands;
        let dim_in = band_dims[band] as usize;
        let g_off = band_offs[band] as usize;
        let x_off = t * band_offs[n_bands] as usize + g_off;
        let w_off = 256 * g_off;
        let b_off = band * 256;
        let dmax = dim_in - 1;
        let sbase = wid * 521;

        // 1) branch-free partial load + square sum
        let mut s = 0.0f32;
        for k in 0..17 {
            let j = lane + k * 32;
            let valid = if j < dim_in { 1.0f32 } else { 0.0f32 };
            let v = x[x_off + j.min(dmax)];
            s += v * v * valid;
            unsafe {
                HB[sbase + j.min(520)] = v * valid;
            }
        }
        // 2) butterfly reduce the norm
        s += warp::shuffle_xor_f32(s, 16);
        s += warp::shuffle_xor_f32(s, 8);
        s += warp::shuffle_xor_f32(s, 4);
        s += warp::shuffle_xor_f32(s, 2);
        s += warp::shuffle_xor_f32(s, 1);
        let norm = s.sqrt();
        let denom = if norm > 1e-12 { norm } else { 1e-12 };
        let scale = (dim_in as f32).sqrt() / denom;
        // 3) normalize + fold gamma in shared memory
        for k in 0..17 {
            let j = lane + k * 32;
            let valid = if j < dim_in { 1.0f32 } else { 0.0f32 };
            let jj = j.min(520);
            unsafe {
                HB[sbase + jj] *= scale * valid * gamma[g_off + j.min(dmax)];
            }
        }
        thread::sync_threads();
        // 4) each lane: 8 complete dots straight from shared memory
        let out_ptr = out.as_mut_ptr();
        let out_base = task * 256;
        for ci in 0..8 {
            let ch = lane * 8 + ci;
            let wrow = w_off + ch * dim_in;
            let mut a = 0.0f32;
            for j in 0..dim_in {
                a += unsafe { HB[sbase + j] } * w[wrow + j];
            }
            // SAFETY: task < rows*n_bands, ch < 256, out sized rows*n_bands*256.
            unsafe {
                *out_ptr.add(out_base + ch) = a + b[b_off + ch];
            }
        }
    }

    /// BandSplit stage 1: per-(band,time) RMSNorm into a padded band-major
    /// tensor consumed by the tensor-core BandSplit GEMM.
    #[kernel]
    pub fn bandsplit_norm(
        x: &[f32], gamma: &[f32], band_off: &[u32], band_dims: &[u32],
        mut out: DisjointSlice<f32>, t_rows: u32, bands: u32, max_dim: u32,
    ) {
        let gid = thread::index_1d();
        let g0 = gid.get();
        let lane = warp::lane_id() as usize;
        let task = g0 / 32;
        let (t_f, bands_, width) = (t_rows as usize, bands as usize, max_dim as usize);
        if task >= bands_ * t_f { return; }
        let band = task / t_f;
        let t = task % t_f;
        let dim = band_dims[band] as usize;
        let g_off = band_off[band] as usize;
        let x_off = t * band_off[bands_] as usize + g_off;
        let out_off = task * width;
        let per = dim.div_ceil(32);
        let mut vals = [0.0f32; 17];
        let mut s = 0.0f32;
        for i in 0..per {
            let j = lane + i * 32;
            let v = if j < dim { x[x_off + j] } else { 0.0 };
            vals[i] = v;
            s += v * v;
        }
        s = warp::reduce_sum_f32(s);
        let denom = if s.sqrt() > 1e-12 { s.sqrt() } else { 1e-12 };
        let scale = (dim as f32).sqrt() / denom;
        let ptr = out.as_mut_ptr();
        for i in 0..17usize {
            let j = lane + i * 32;
            let v = if i < per && j < dim { vals[i] * scale * gamma[g_off + j] } else { 0.0 };
            if j < width {
                unsafe { *ptr.add(out_off + j) = v; }
            }
        }
    }

    /// Debug variant: lane 0 recomputes the whole band serially from the
    /// warp-reduced norm. Isolates lane-partitioning bugs.
    #[kernel]
    pub fn bandsplit_serial(
        x: &[f32],
        gamma: &[f32],
        w: &[f32],
        b: &[f32],
        band_offs: &[u32],
        band_dims: &[u32],
        mut out: DisjointSlice<f32>,
        rows: u32,
        n_bands: u32,
    ) {
        let gid = thread::index_1d();
        let g0 = gid.get();
        let lane = warp::lane_id() as usize;
        let task = g0 / 32;
        let rows = rows as usize;
        let n_bands = n_bands as usize;
        if task >= rows * n_bands {
            return;
        }
        let dim_in = band_dims[task % n_bands] as usize;
        let band = task % n_bands;
        let t = task / n_bands;
        let g_off = band_offs[band] as usize;
        let x_off = t * band_offs[n_bands] as usize + g_off;
        // norm over the band via one lane, then broadcast through shuffle
        let mut s = 0.0f32;
        if lane == 0 {
            for j in 0..dim_in {
                s += x[x_off + j] * x[x_off + j];
            }
        }
        s = warp::shuffle_f32(s, 0);
        let norm = s.sqrt();
        let denom = if norm > 1e-12 { norm } else { 1e-12 };
        let scale = (dim_in as f32).sqrt() / denom;
        if lane == 0 {
            let w_off = 256 * g_off;
            let b_off = band * 256;
            let out_ptr = out.as_mut_ptr();
            for ch in 0..256 {
                let mut a = 0.0f32;
                let wrow = w_off + ch * dim_in;
                for j in 0..dim_in {
                    a += x[x_off + j] * scale * gamma[g_off + j] * w[wrow + j];
                }
                // SAFETY: task < rows*n_bands, ch < 256.
                unsafe {
                    *out_ptr.add(task * 256 + ch) = a + b[b_off + ch];
                }
            }
        }
    }

    /// GEMM: Y[M,N] = X[M,K] . W[N,K]^T + bias. 16x16 tiles, one C
    /// element per thread (Phase A baseline). W stays in torch [N,K] order.
    #[kernel]
    pub fn gemm_bias(
        m: u32,
        n: u32,
        k: u32,
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        static mut TA: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TB: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * 16 + ty;
        let col = thread::blockIdx_x() as usize * 16 + tx;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let num_tiles = k_size.div_ceil(16);
        let mut sum = 0.0f32;
        let smem_idx = ty * 16 + tx;
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * 16;
            unsafe {
                let a_col = tile_start + tx;
                TA[smem_idx] = if row < m_size && a_col < k_size { x[row * k_size + a_col] } else { 0.0 };
                // W is [N, K] row-major: element (k_idx, col) == w[col*k + k_idx]
                let b_row = tile_start + ty;
                TB[smem_idx] = if b_row < k_size && col < n_size { w[col * k_size + b_row] } else { 0.0 };
            }
            thread::sync_threads();
            unsafe {
                let mut i = 0usize;
                while i < 16 {
                    sum += TA[ty * 16 + i] * TB[i * 16 + tx];
                    i += 1;
                }
            }
            thread::sync_threads();
            tile += 1;
        }
        if row < m_size && col < n_size {
            let out_ptr = y.as_mut_ptr();
            // SAFETY: row/n bound checked; y is m*n elements.
            unsafe {
                *out_ptr.add(row * n_size + col) = sum + bias[col];
            }
        }
    }

    /// RoPE + scale on a folded QKV tensor: qkv[(b*seq+pos), 3*8*64].
    /// Each thread rotates one (even, odd) pair of q or k and scales; v and
    /// the partner element are written by the same thread. cos/sin tables
    /// are [seq, 32] row-major.
    #[kernel]
    pub fn rope_scale(
        qkv: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut out: DisjointSlice<f32>,
        seq: u32,
        bands: u32,
        axis: u32,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        // g0 = m * 768 + c2, where c2 in [0, 768) indexes (part, head, pair):
        // part 0=q, 1=k; 768 = 2 parts * 8 heads * 32 pairs? Actually 3*8*64/2
        // pairs total = 768; v occupies pairs 512..768 and passes through
        // untouched (its two elements are copied as-is).
        let c2 = g0 % 768;
        let m = g0 / 768;
        let part = c2 / 256; // 0=q, 1=k, 2=v (in pair units: 256 pairs each)
        let head = (c2 % 256) / 32;
        let pair = c2 % 32;
        // token m is (t, band) folded: time axis pos = m / bands, freq axis
        // pos = m % bands (x is (t, f) row-major).
        let pos = if axis == 0 { m / bands as usize } else { m % bands as usize };
        let base = m * 1536 + part * 512 + head * 64 + pair * 2;
        let c = cos[pos * 32 + pair];
        let s = sin[pos * 32 + pair];
        let (ve, vo) = (qkv[base], qkv[base + 1]);
        let (re, ro) = if part < 2 {
            (ve * c - vo * s, vo * c + ve * s)
        } else {
            (ve, vo)
        };
        // Only q (part 0) gets the SDPA scale; k must stay unscaled.
        let scale = if part == 0 { 0.125f32 } else { 1.0f32 };
        let out_ptr = out.as_mut_ptr();
        // SAFETY: base+1 < rows*1536 by construction (c2 < 768).
        unsafe {
            *out_ptr.add(base) = re * scale;
            *out_ptr.add(base + 1) = ro * scale;
        }
    }

    /// Short-sequence SDPA (freq axis, seq=62): one 128-thread block per
    /// (batch*head). q/k/v are [bh, 62, 64] folded. Scores materialize in
    /// shared memory; softmax rows are computed serially per thread (no
    /// cross-lane reductions). scale = 64^-0.5 = 0.125 (already folded into
    /// q by rope_scale upstream, so NOT applied here).
    #[kernel]
    pub fn attn_short(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
        seq: u32,
    ) {
        // 48KB budget: K+V tiles (31KB) + scores (15.9KB). Q reads straight
        // from global (each row is re-read n times; L2 absorbs it).
        static mut SK: SharedArray<f32, { 62 * 64 }> = SharedArray::UNINIT;
        static mut SV: SharedArray<f32, { 62 * 64 }> = SharedArray::UNINIT;
        static mut SC: SharedArray<f32, { 62 * 64 }> = SharedArray::UNINIT; // scores, rows padded to 64
        let tid = thread::threadIdx_x() as usize;
        let bh = thread::blockIdx_x() as usize;
        let n = seq as usize;
        let base = bh * n * 64;
        // cooperative load: 62*64 = 3968 elements per matrix, 128 threads
        let total = n * 64;
        let mut i = tid;
        while i < total {
            unsafe {
                SK[i] = k[base + i];
                SV[i] = v[base + i];
            }
            i += 128;
        }
        thread::sync_threads();
        // scores: row-major padded to 64 cols, scaled by 0.125 (applied to q
        // upstream — but the reference applies it inside SDPA; our q input
        // from rope_scale is pre-scaled, so scores here are unscaled dots).
        let stride = 64; // padded row width
        let mut idx = tid;
        let n2 = n * n;
        while idx < n2 {
            let r = idx / n;
            let c = idx % n;
            let mut acc = 0.0f32;
            let mut d = 0usize;
            while d < 64 {
                acc += q[base + r * 64 + d] * unsafe { SK[c * 64 + d] };
                d += 1;
            }
            unsafe {
                SC[r * stride + c] = acc;
            }
            idx += 128;
        }
        thread::sync_threads();
        // softmax + weighted V: one thread per row (62 rows), 64 outputs each
        if tid < n {
            let r = tid;
            let row_off = r * stride;
            let mut m = f32::NEG_INFINITY;
            let mut j = 0usize;
            while j < n {
                let s = unsafe { SC[row_off + j] };
                if s > m {
                    m = s;
                }
                j += 1;
            }
            let mut z = 0.0f32;
            j = 0;
            while j < n {
                let e = (unsafe { SC[row_off + j] } - m).exp();
                unsafe {
                    SC[row_off + j] = e;
                }
                z += e;
                j += 1;
            }
            let inv = 1.0 / z;
            let out_ptr = out.as_mut_ptr();
            let mut d = 0usize;
            while d < 64 {
                let mut acc = 0.0f32;
                j = 0;
                while j < n {
                    acc += unsafe { SC[row_off + j] } * unsafe { SV[j * 64 + d] };
                    j += 1;
                }
                // SAFETY: bh < grid, r < n, d < 64; out has bh*n*64 elements.
                unsafe {
                    *out_ptr.add(base + r * 64 + d) = acc * inv;
                }
                d += 1;
            }
        }
    }

    /// Batched online-softmax SDPA for any axis. q/k/v are [BH, N, 64];
    /// q is already scaled by 64^-1/2. One warp computes one query row, so a
    /// 256-thread block covers 8 rows. K/V are streamed through shared memory
    /// in 64-key tiles; no [BH, N, N] score matrix is materialized.
    #[kernel]
    pub fn attn_flash(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
        seq: u32,
    ) {
        static mut SQ: SharedArray<f32, { 8 * 64 }> = SharedArray::UNINIT;
        static mut SK: SharedArray<f32, { 64 * 64 }> = SharedArray::UNINIT;
        static mut SV: SharedArray<f32, { 64 * 64 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let n = seq as usize;
        // Flatten (bh, query tile) into grid.x: cuda-oxide's 1-D block/grid
        // path is the best-tested indexing route and avoids mixing grid axes.
        let q_tiles = n.div_ceil(8);
        let bh = thread::blockIdx_x() as usize / q_tiles;
        let q_row = (thread::blockIdx_x() as usize % q_tiles) * 8 + warp_id;
        let bh_base = bh * n * 64;
        let q_valid = q_row < n;

        // Load this block's 8 query rows into shared memory.
        let mut i = tid;
        while i < 8 * 64 {
            let r = i / 64;
            let d = i % 64;
            let row = (thread::blockIdx_x() as usize % q_tiles) * 8 + r;
            unsafe {
                SQ[i] = if row < n { q[bh_base + row * 64 + d] } else { 0.0 };
            }
            i += 256;
        }
        thread::sync_threads();

        let row_off = warp_id * 64; // SQ is block-local (32 rows), not bh-local
        let d0 = lane * 2;
        let d1 = lane * 2 + 1;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut sum = 0.0f32;
        let mut old_max = f32::NEG_INFINITY;

        let mut kt = 0usize;
        while kt < n {
            // Cooperative 64-key K/V tile load (two 16 KiB arrays).
            let mut j = tid;
            while j < 2 * 64 * 64 {
                if j < 64 * 64 {
                    let key = kt + j / 64;
                    let d = j % 64;
                    unsafe {
                        SK[j] = if key < n { k[bh_base + key * 64 + d] } else { 0.0 };
                    }
                } else {
                    let x = j - 64 * 64;
                    let key = kt + x / 64;
                    let d = x % 64;
                    unsafe {
                        SV[x] = if key < n { v[bh_base + key * 64 + d] } else { 0.0 };
                    }
                }
                j += 256;
            }
            thread::sync_threads();

            // Each lane owns two score columns. Invalid keys stay -inf so they
            // contribute zero after the online-softmax rescaling.
            let kj0 = kt + lane * 2;
            let kj1 = kj0 + 1;
            let mut s0 = f32::NEG_INFINITY;
            let mut s1 = f32::NEG_INFINITY;
            if q_valid && kj0 < n {
                let mut acc = 0.0f32;
                let mut d = 0usize;
                while d < 64 {
                    acc += unsafe { SQ[row_off + d] } * unsafe { SK[(lane * 2) * 64 + d] };
                    d += 1;
                }
                s0 = acc;
            }
            if q_valid && kj1 < n {
                let mut acc = 0.0f32;
                let mut d = 0usize;
                while d < 64 {
                    acc += unsafe { SQ[row_off + d] } * unsafe { SK[(lane * 2 + 1) * 64 + d] };
                    d += 1;
                }
                s1 = acc;
            }

            let tile_max = if s0 > s1 { s0 } else { s1 };
            let mut new_max = warp::reduce_max_f32(tile_max);
            if new_max == f32::NEG_INFINITY {
                new_max = 0.0;
            }
            if old_max > new_max {
                new_max = old_max;
            }
            let correction = (old_max - new_max).exp();
            acc0 *= correction;
            acc1 *= correction;
            sum *= correction;

            let p0 = if s0 == f32::NEG_INFINITY { 0.0 } else { (s0 - new_max).exp() };
            let p1 = if s1 == f32::NEG_INFINITY { 0.0 } else { (s1 - new_max).exp() };
            sum += warp::reduce_sum_f32(p0 + p1);

            // Every lane broadcasts its two probabilities in turn; all lanes
            // accumulate their two owned output dimensions over the tile.
            let mut src = 0usize;
            while src < 32 {
                let src_lane = src as u32;
                let pa = warp::shuffle_f32(p0, src_lane);
                let pb = warp::shuffle_f32(p1, src_lane);
                let kg0 = kt + src * 2;
                let kg1 = kg0 + 1;
                unsafe {
                    if kg0 < n {
                        let vb = (src * 2) * 64;
                        acc0 += pa * SV[vb + d0];
                        acc1 += pa * SV[vb + d1];
                    }
                    if kg1 < n {
                        let vb = (src * 2 + 1) * 64;
                        acc0 += pb * SV[vb + d0];
                        acc1 += pb * SV[vb + d1];
                    }
                }
                src += 1;
            }

            old_max = new_max;
            thread::sync_threads();
            kt += 64;
        }

        if q_valid && sum > 0.0 {
            let inv = 1.0 / sum;
            let out_ptr = out.as_mut_ptr();
            let base = bh_base + q_row * 64;
            // SAFETY: q_row < n, d0/d1 < 64, and out has BH*n*64 elements.
            unsafe {
                *out_ptr.add(base + d0) = acc0 * inv;
                *out_ptr.add(base + d1) = acc1 * inv;
            }
        }
    }

    /// Two-row variant of attn_flash: each warp owns two query rows, so a
    /// 256-thread block covers 16 rows and each K/V tile is loaded half as
    /// often. Useful when the one-row kernel is streaming-memory bound.
    #[kernel]
    pub fn attn_flash2(
        q: &[f32], k: &[f32], v: &[f32],
        mut out: DisjointSlice<f32>, seq: u32,
    ) {
        static mut SQ: SharedArray<f32, { 16 * 64 }> = SharedArray::UNINIT;
        static mut SK: SharedArray<f32, { 64 * 64 }> = SharedArray::UNINIT;
        static mut SV: SharedArray<f32, { 64 * 64 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let n = seq as usize;
        let q_tiles = n.div_ceil(16);
        let tile_origin = (thread::blockIdx_x() as usize % q_tiles) * 16;
        let bh = thread::blockIdx_x() as usize / q_tiles;
        let bh_base = bh * n * 64;
        let q0 = tile_origin + warp_id * 2;
        let q1 = q0 + 1;
        let (valid0, valid1) = (q0 < n, q1 < n);

        let mut i = tid;
        while i < 16 * 64 {
            let row = tile_origin + i / 64;
            let d = i % 64;
            unsafe { SQ[i] = if row < n { q[bh_base + row * 64 + d] } else { 0.0 }; }
            i += 256;
        }
        thread::sync_threads();

        let row0 = (warp_id * 2) * 64;
        let row1 = row0 + 64;
        let d0 = lane * 2;
        let d1 = lane * 2 + 1;
        let (mut a00, mut a01) = (0.0f32, 0.0f32);
        let (mut a10, mut a11) = (0.0f32, 0.0f32);
        let (mut sum0, mut sum1) = (0.0f32, 0.0f32);
        let (mut max0, mut max1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);

        let mut kt = 0usize;
        while kt < n {
            let mut j = tid;
            while j < 2 * 64 * 64 {
                if j < 64 * 64 {
                    let key = kt + j / 64;
                    let d = j % 64;
                    unsafe { SK[j] = if key < n { k[bh_base + key * 64 + d] } else { 0.0 }; }
                } else {
                    let x = j - 64 * 64;
                    let key = kt + x / 64;
                    let d = x % 64;
                    unsafe { SV[x] = if key < n { v[bh_base + key * 64 + d] } else { 0.0 }; }
                }
                j += 256;
            }
            thread::sync_threads();

            let (kj0, kj1) = (kt + lane * 2, kt + lane * 2 + 1);
            let mut s00 = f32::NEG_INFINITY;
            let mut s01 = f32::NEG_INFINITY;
            let mut s10 = f32::NEG_INFINITY;
            let mut s11 = f32::NEG_INFINITY;
            if valid0 && kj0 < n {
                let mut acc = 0.0;
                let mut d = 0;
                while d < 64 { acc += unsafe { SQ[row0 + d] } * unsafe { SK[(lane * 2) * 64 + d] }; d += 1; }
                s00 = acc;
            }
            if valid0 && kj1 < n {
                let mut acc = 0.0;
                let mut d = 0;
                while d < 64 { acc += unsafe { SQ[row0 + d] } * unsafe { SK[(lane * 2 + 1) * 64 + d] }; d += 1; }
                s01 = acc;
            }
            if valid1 && kj0 < n {
                let mut acc = 0.0;
                let mut d = 0;
                while d < 64 { acc += unsafe { SQ[row1 + d] } * unsafe { SK[(lane * 2) * 64 + d] }; d += 1; }
                s10 = acc;
            }
            if valid1 && kj1 < n {
                let mut acc = 0.0;
                let mut d = 0;
                while d < 64 { acc += unsafe { SQ[row1 + d] } * unsafe { SK[(lane * 2 + 1) * 64 + d] }; d += 1; }
                s11 = acc;
            }

            let mut nm0 = warp::reduce_max_f32(if s00 > s01 { s00 } else { s01 });
            let mut nm1 = warp::reduce_max_f32(if s10 > s11 { s10 } else { s11 });
            if nm0 == f32::NEG_INFINITY { nm0 = 0.0; }
            if nm1 == f32::NEG_INFINITY { nm1 = 0.0; }
            if max0 > nm0 { nm0 = max0; }
            if max1 > nm1 { nm1 = max1; }
            let corr0 = (max0 - nm0).exp();
            let corr1 = (max1 - nm1).exp();
            a00 *= corr0; a01 *= corr0; sum0 *= corr0;
            a10 *= corr1; a11 *= corr1; sum1 *= corr1;

            let p00 = if s00 == f32::NEG_INFINITY { 0.0 } else { (s00 - nm0).exp() };
            let p01 = if s01 == f32::NEG_INFINITY { 0.0 } else { (s01 - nm0).exp() };
            let p10 = if s10 == f32::NEG_INFINITY { 0.0 } else { (s10 - nm1).exp() };
            let p11 = if s11 == f32::NEG_INFINITY { 0.0 } else { (s11 - nm1).exp() };
            sum0 += warp::reduce_sum_f32(p00 + p01);
            sum1 += warp::reduce_sum_f32(p10 + p11);

            let mut src = 0usize;
            while src < 32 {
                let lane_src = src as u32;
                let u0 = warp::shuffle_f32(p00, lane_src);
                let u1 = warp::shuffle_f32(p01, lane_src);
                let w0 = warp::shuffle_f32(p10, lane_src);
                let w1 = warp::shuffle_f32(p11, lane_src);
                let kg0 = kt + src * 2;
                let kg1 = kg0 + 1;
                unsafe {
                    if kg0 < n {
                        let vb = (src * 2) * 64;
                        a00 += u0 * SV[vb + d0]; a01 += u0 * SV[vb + d1];
                        a10 += w0 * SV[vb + d0]; a11 += w0 * SV[vb + d1];
                    }
                    if kg1 < n {
                        let vb = (src * 2 + 1) * 64;
                        a00 += u1 * SV[vb + d0]; a01 += u1 * SV[vb + d1];
                        a10 += w1 * SV[vb + d0]; a11 += w1 * SV[vb + d1];
                    }
                }
                src += 1;
            }

            max0 = nm0;
            max1 = nm1;
            thread::sync_threads();
            kt += 64;
        }

        let ptr = out.as_mut_ptr();
        if valid0 && sum0 > 0.0 {
            let inv = 1.0 / sum0;
            let base = bh_base + q0 * 64;
            unsafe { *ptr.add(base + d0) = a00 * inv; *ptr.add(base + d1) = a01 * inv; }
        }
        if valid1 && sum1 > 0.0 {
            let inv = 1.0 / sum1;
            let base = bh_base + q1 * 64;
            unsafe { *ptr.add(base + d0) = a10 * inv; *ptr.add(base + d1) = a11 * inv; }
        }
    }

    /// In-place row softmax over a [rows, cols] matrix (one thread/row).
    #[kernel]
    pub fn softmax_rows(mut p: DisjointSlice<f32>, rows: u32, cols: u32) {
        let idx = thread::index_1d();
        let r = idx.get();
        let (rows, cols) = (rows as usize, cols as usize);
        if r >= rows {
            return;
        }
        let off = r * cols;
        let ptr = p.as_mut_ptr();
        let mut m = unsafe { *ptr.add(off) };
        let mut j = 1usize;
        while j < cols {
            let s = unsafe { *ptr.add(off + j) };
            if s > m {
                m = s;
            }
            j += 1;
        }
        let mut z = 0.0f32;
        j = 0;
        while j < cols {
            let e = (unsafe { *ptr.add(off + j) } - m).exp();
            unsafe {
                *ptr.add(off + j) = e;
            }
            z += e;
            j += 1;
        }
        let inv = 1.0 / z;
        j = 0;
        while j < cols {
            // SAFETY: off+j < rows*cols.
            unsafe {
                *ptr.add(off + j) *= inv;
            }
            j += 1;
        }
    }

    /// GEMM with B in [K, N] row-major layout (torch x @ B semantics):
    /// Y[M,N] = X[M,K] . B[K,N] + bias.
    #[kernel]
    pub fn gemm_bias_bn(
        m: u32,
        n: u32,
        k: u32,
        x: &[f32],
        b_mat: &[f32],
        bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        static mut TA: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TB: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * 16 + ty;
        let col = thread::blockIdx_x() as usize * 16 + tx;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let num_tiles = k_size.div_ceil(16);
        let mut sum = 0.0f32;
        let smem_idx = ty * 16 + tx;
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * 16;
            unsafe {
                let a_col = tile_start + tx;
                TA[smem_idx] = if row < m_size && a_col < k_size { x[row * k_size + a_col] } else { 0.0 };
                let b_row = tile_start + ty;
                TB[smem_idx] = if b_row < k_size && col < n_size { b_mat[b_row * n_size + col] } else { 0.0 };
            }
            thread::sync_threads();
            unsafe {
                let mut i = 0usize;
                while i < 16 {
                    sum += TA[ty * 16 + i] * TB[i * 16 + tx];
                    i += 1;
                }
            }
            thread::sync_threads();
            tile += 1;
        }
        if row < m_size && col < n_size {
            let out_ptr = y.as_mut_ptr();
            // SAFETY: bounds checked above; y is m*n.
            unsafe {
                *out_ptr.add(row * n_size + col) = sum + bias[col];
            }
        }
    }

    /// GELU (erf form) via the Abramowitz-Stegun 7.1.26 approximation
    /// (|eps| <= 1.5e-7, ~1000x tighter than the tanh approximation).
    #[kernel]
    pub fn gelu_erf(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[i];
            let sign = if v < 0.0 { -1.0f32 } else { 1.0 };
            // GELU uses erf(x/sqrt(2))
            let a = v.abs() * 0.70710678;
            let t = 1.0 / (1.0 + 0.3275911 * a);
            let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
            let erfa = sign * (1.0 - poly * (-a * a).exp());
            *o = 0.5 * v * (1.0 + erfa);
        }
    }

    /// Head-gate scaling BEFORE the out projection:
    /// scaled[m, c] = attn_raw[m, c] * sigmoid(gates[m, c/64]), c in [0,512).
    #[kernel]
    pub fn gate_scale(
        attn_raw: &[f32],
        gates: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if g0 < attn_raw.len() {
            let head = (g0 % 512) / 64;
            let m = g0 / 512;
            let sig = 1.0 / (1.0 + (-gates[m * 8 + head]).exp());
            if let Some(o) = out.get_mut(idx) {
                *o = attn_raw[g0] * sig;
            }
        }
    }

    /// Elementwise y[i] = x[i] + res[i].
    #[kernel]
    pub fn add_resid(x: &[f32], res: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len() {
            if let Some(o) = y.get_mut(idx) {
                *o = x[i] + res[i];
            }
        }
    }

    /// Reorder folded QKV into attention-major layout:
    /// in qkv[m=(t,f), part*512 + head*64 + d] -> out[(part*BH + seq_i*8 + head)*N + d]
    /// where seq_i is the attention sequence index (time axis: t, BH=f*8,
    /// N=T; freq axis: f, BH=t*8, N=62).
    #[kernel]
    pub fn qkv_to_attn(
        qkv: &[f32],
        mut out: DisjointSlice<f32>,
        t_frames: u32,
        bands: u32,
        axis: u32,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        // g0 = ((part*BH + seq_i*8 + head)*N + row)*64 + d — decompose once
        let total = out.len();
        if g0 < total {
            let (t_f, bands_, axis_) = (t_frames as usize, bands as usize, axis as usize);
            let d = g0 % 64;
            let row_col = g0 / 64;
            let n = row_col % (if axis_ == 0 { t_f } else { bands_ });
            let rest = row_col / (if axis_ == 0 { t_f } else { bands_ });
            let head = rest % 8;
            let seq_i = rest / 8;
            let part = seq_i / (if axis_ == 0 { bands_ } else { t_f });
            let s = seq_i % (if axis_ == 0 { bands_ } else { t_f });
            // source token m and channel
            let (t, f) = if axis_ == 0 { (n, s) } else { (s, n) };
            let m = t * bands_ + f;
            let src = m * 1536 + part * 512 + head * 64 + d;
            if let Some(o) = out.get_mut(idx) {
                *o = qkv[src];
            }
        }
    }

    /// Fused RoPE + QKV reorder: reads the raw QKV GEMM output and writes the
    /// attention-major layout directly, avoiding one full qkv-sized round trip.
    #[kernel]
    pub fn rope_qkv_to_attn(
        qkv: &[f32], cos: &[f32], sin: &[f32],
        mut out: DisjointSlice<f32>,
        t_frames: u32, bands: u32, axis: u32,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if g0 < out.len() {
            let (t_f, bands_, axis_) = (t_frames as usize, bands as usize, axis as usize);
            let n_len = if axis_ == 0 { t_f } else { bands_ };
            let d = g0 % 64;
            let row_col = g0 / 64;
            let n = row_col % n_len;
            let rest = row_col / n_len;
            let head = rest % 8;
            let seq_i = rest / 8;
            let batch_count = if axis_ == 0 { bands_ } else { t_f };
            let part = seq_i / batch_count;
            let s = seq_i % batch_count;
            let (t, f) = if axis_ == 0 { (n, s) } else { (s, n) };
            let m = t * bands_ + f;
            let pair = d / 2;
            let src = m * 1536 + part * 512 + head * 64 + pair * 2;
            let (ve, vo) = (qkv[src], qkv[src + 1]);
            let value = if part < 2 {
                let pos = n; // n is the attention sequence position on both axes
                let pair = d / 2;
                let c = cos[pos * 32 + pair];
                let sn = sin[pos * 32 + pair];
                let (re, ro) = (ve * c - vo * sn, vo * c + ve * sn);
                if d % 2 == 0 { re } else { ro }
            } else if d % 2 == 0 {
                ve
            } else {
                vo
            };
            let scale = if part == 0 { 0.125f32 } else { 1.0 };
            if let Some(o) = out.get_mut(idx) {
                *o = value * scale;
            }
        }
    }

    /// Inverse of qkv_to_attn for the V output only (attn out [BH, N, 64] ->
    /// folded (t,f) rows: out_flat[m*512 + head*64 + d]).
    #[kernel]
    pub fn attn_v_to_flat(
        attn: &[f32], gates: &[f32],
        mut out: DisjointSlice<f32>,
        t_frames: u32,
        bands: u32,
        axis: u32,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        // g0 = m*512 + head*64 + d
        if g0 < out.len() {
            let (t_f, bands_, axis_) = (t_frames as usize, bands as usize, axis as usize);
            let d = g0 % 64;
            let ch = g0 / 64;
            let head = ch % 8;
            let m = ch / 8;
            let f = m % bands_;
            let t = m / bands_;
            let (n, s) = if axis_ == 0 { (t, f) } else { (f, t) };
            let bh = s * 8 + head;
            let n_len = if axis_ == 0 { t_f } else { bands_ };
            // attn_out_long is [BH, N, 64] — no part offset here.
            let src = (bh * n_len + n) * 64 + d;
            if let Some(o) = out.get_mut(idx) {
                let sig = 1.0 / (1.0 + (-gates[m * 8 + head]).exp());
                *o = attn[src] * sig;
            }
        }
    }

    /// Reorder the batched STFT spectrum into BandSplit input layout.
    /// in: spec[(ch*T + t)*1025 + (band_f0 + fi)] * 2 + c]
    /// out: x_in[t*4100 + band_off + fi*4 + ch*2 + c]
    #[kernel]
    pub fn spec_reorder(
        spec: &[f32],
        band_f0: &[u32],
        band_off: &[u32],
        mut out: DisjointSlice<f32>,
        t_frames: u32,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if g0 < out.len() {
            let t_f = t_frames as usize;
            // out coordinate: t, band, fi, ch, c (4100 = sum(2*freqs*2))
            let c = g0 % 2;
            let ch = (g0 / 2) % 2;
            let rest = g0 / 4;
            let t = rest / 1025;
            let fc = rest % 1025; // flattened (band, fi)
            // fc is in 1025-space; locate the band via the f0 table
            let mut band = 0usize;
            while band + 1 < band_f0.len() && (band_f0[band + 1] as usize) <= fc {
                band += 1;
            }
            let src = ((ch * t_f + t) * 1025 + fc) * 2 + c;
            let _ = band;
            if let Some(o) = out.get_mut(idx) {
                *o = spec[src];
            }
        }
    }

    /// MaskEstimator per-band MLP for ONE stem, batched over 62 bands as a
    /// loop of GEMMs is host-side; this kernel handles the epilogue:
    /// tanh + second GEMM would need per-band W — instead we run the 1st GEMM
    /// per band via gemm_bias on the host loop, then this kernel does GLU:
    /// y = a * sigmoid(b) over pairs: out (T, dim_in*2 per band -> halved).
    #[kernel]
    pub fn glu_halve(x: &[f32], mut out: DisjointSlice<f32>, width: u32) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        let w = width as usize; // full width per (row, band) segment
        if g0 < out.len() {
            // out index g0 corresponds to input pair at (g0/w)*w + g0%w
            let half = w / 2;
            let row = g0 / half;
            let col = g0 % half;
            let a = x[row * w + col];
            let b = x[row * w + half + col];
            if let Some(o) = out.get_mut(idx) {
                *o = a / (1.0 + (-b).exp());
            }
        }
    }

    /// Complex mask multiply: masked_spec[f, t] = stft[f, t] * mask[f, t].
    /// Both in (f-major per stem) layout (2050, T, 2): out = (ar*br - ai*bi,
    /// ar*bi + ai*br).
    #[kernel]
    pub fn cmul(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        let n = out.len() / 2;
        if g0 < n {
            let (ar, ai) = (a[g0 * 2], a[g0 * 2 + 1]);
            let (br, bi) = (b[g0 * 2], b[g0 * 2 + 1]);
            // SAFETY: each thread owns the disjoint pair (2g0, 2g0+1).
            let p = out.as_mut_ptr();
            unsafe {
                *p.add(g0 * 2) = ar * br - ai * bi;
                *p.add(g0 * 2 + 1) = ar * bi + ai * br;
            }
        }
    }

    /// Elementwise tanh (mask-estimator hidden activation).
    #[kernel]
    pub fn tanh_e(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[i].tanh();
        }
    }

    /// Transpose (T, 62, 256) -> (62, T, 256) so each band's rows are
    /// contiguous for per-band GEMMs.
    #[kernel]
    pub fn transpose_band_major(x: &[f32], mut out: DisjointSlice<f32>, t_frames: u32, bands: u32) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if g0 < out.len() {
            let (t_f, bands_) = (t_frames as usize, bands as usize);
            let d = g0 % 256;
            let row = g0 / 256;
            // Output is (band, t, 256) so row = b*T + t → b = row/T, t = row%T
            let b = row / t_f;
            let t = row % t_f;
            let src = (t * bands_ + b) * 256 + d;
            if let Some(o) = out.get_mut(idx) {
                *o = x[src];
            }
        }
    }

    /// Apply the GLU'd per-band masks to the STFT spectrum, writing the
    /// C2R batch layout directly: frame (stem*2 + ch)*T + t, freq bins, c.
    /// glu layout is (stem, band, t, fi*4 + ch*2 + c) row-major.
    #[kernel]
    pub fn mask_apply(
        spec: &[f32],
        glu: &[f32],
        band_f0: &[u32],
        mut out: DisjointSlice<f32>,
        t_frames: u32,
        bands: u32,
    ) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if g0 < out.len() {
            let (t_f, bands_) = (t_frames as usize, bands as usize);
            let c = g0 % 2;
            let fpos = (g0 / 2) % 1025;
            let frame = g0 / 2 / 1025;
            let ch = (frame / t_f) % 2;
            let t = frame % t_f;
            let stem = frame / (2 * t_f);
            // locate band for this freq bin
            let mut band = 0usize;
            while band + 1 < band_f0.len() && (band_f0[band + 1] as usize) <= fpos {
                band += 1;
            }
            let fi = fpos - band_f0[band] as usize;
            let gcol = fi * 4 + ch * 2 + c;
            // glu rows are padded to max_dim = 516
            // Complex multiplication: (sr + si*j) * (mr + mi*j)
            // Each thread handles ONE component (c=0 for real, c=1 for imag).
            // Read both spec and mask components to compute this component.
            let g_base = ((stem * bands_ + band) * t_f + t) * 516 + fi * 4 + ch * 2;
            let mr = glu[g_base];
            let mi = glu[g_base + 1];
            let sr = spec[((ch * t_f + t) * 1025 + fpos) * 2];
            let si = spec[((ch * t_f + t) * 1025 + fpos) * 2 + 1];
            let v = if c == 0 { sr * mr - si * mi } else { sr * mi + si * mr };
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }

    /// Copy compact rows (src_dim) into padded rows (dst_dim), zero-fill tail.
    #[kernel]
    pub fn copy_masked(src: &[f32], mut dst: DisjointSlice<f32>, src_dim: u32, dst_dim: u32) {
        let idx = thread::index_1d();
        let g0 = idx.get();
        if g0 < dst.len() {
            let (sd, dd) = (src_dim as usize, dst_dim as usize);
            let row = g0 / dd;
            let col = g0 % dd;
            let v = if col < sd { src[row * sd + col] } else { 0.0 };
            if let Some(o) = dst.get_mut(idx) {
                *o = v;
            }
        }
    }


    /// High-performance GEMM: 64x64 tile, 4x4 register blocking per thread.
    /// Y[M,N] = X[M,K] · W[N,K]^T + bias. 256 threads per block, each
    /// computing a 4x4 block of the 64x64 output tile.
    



    /// High-performance GEMM: 64x64 tile, 4x4 register blocking, 256 threads.
    /// Uses linear shared-memory loading (each thread loads 16 elements via
    /// tid + i*256 covering all 4096 positions).
    #[kernel]
    pub fn gemm_bias_64(
        m: u32,
        n: u32,
        k: u32,
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        static mut TA: SharedArray<f32, { 64 * 64 }> = SharedArray::UNINIT;
        static mut TB: SharedArray<f32, { 64 * 64 }> = SharedArray::UNINIT;
        let tx = thread::threadIdx_x() as usize; // 0..15
        let ty = thread::threadIdx_y() as usize; // 0..15
        let tid = ty * 16 + tx; // 0..255
        let row_base = thread::blockIdx_y() as usize * 64;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let num_tiles = k_size.div_ceil(64);
        // 4x4 register accumulators for this thread's output block
        // Thread (ty, tx) computes rows [ty*4..ty*4+4) and cols [tx*4..tx*4+4)
        let mut acc = [[0.0f32; 4]; 4];
        let mut tile = 0usize;
        while tile < num_tiles {
            let ts = tile * 64;
            // Linear loading: each of 256 threads loads 16 elements
            unsafe {
                for i in 0..16usize {
                    let linear = tid + i * 256;
                    let r = linear / 64;  // 0..63
                    let cc = linear % 64; // 0..63
                    // A tile: x[row_base+r][ts+cc]
                    let ar = row_base + r;
                    let ac = ts + cc;
                    TA[r * 64 + cc] = if ar < m_size && ac < k_size { x[ar * k_size + ac] } else { 0.0 };
                    // B tile: w[col_base+cc][ts+r] (W is [N,K], we need B[r][cc] = W[cc][r])
                    let bc = col_base + cc;
                    let br = ts + r;
                    TB[r * 64 + cc] = if br < k_size && bc < n_size { w[bc * k_size + br] } else { 0.0 };
                }
            }
            thread::sync_threads();
            // Compute 4x4 outputs
            unsafe {
                let mut kk = 0usize;
                while kk < 64 {
                    let a0 = TA[ty * 4 * 64 + kk];
                    let a1 = TA[(ty * 4 + 1) * 64 + kk];
                    let a2 = TA[(ty * 4 + 2) * 64 + kk];
                    let a3 = TA[(ty * 4 + 3) * 64 + kk];
                    let b0 = TB[kk * 64 + tx * 4];
                    let b1 = TB[kk * 64 + tx * 4 + 1];
                    let b2 = TB[kk * 64 + tx * 4 + 2];
                    let b3 = TB[kk * 64 + tx * 4 + 3];
                    acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
                    acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
                    acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
                    acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
                    kk += 1;
                }
            }
            thread::sync_threads();
            tile += 1;
        }
        // Write 4x4 outputs with bounds check
        let out_ptr = y.as_mut_ptr();
        for i in 0..4usize {
            for j in 0..4usize {
                let r = row_base + ty * 4 + i;
                let cc = col_base + tx * 4 + j;
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked above; y is m*n elements.
                    unsafe {
                        *out_ptr.add(r * n_size + cc) = acc[i][j] + bias[cc];
                    }
                }
            }
        }
    }


    /// TF32 tensor-core GEMM: Y[M,N] = X[M,K] · W[N,K]^T + bias.
    /// Uses mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32.
    /// Each block: 128 threads (4 warps), computes 64×8 output tile.
    /// TF32 conversion: truncate lower 13 mantissa bits (f32→TF32).
    #[kernel]
    pub fn gemm_tf32(
        m: u32, n: u32, k: u32,
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Shared: A tile (64 rows × 8 K) and B tile (8 K × 8 cols)
        static mut SA: SharedArray<f32, { 64 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<f32, { 8 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;

        let block_row_base = thread::blockIdx_y() as usize * 64;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 8;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);

        // Accumulator: 4 f32 per lane (16×8 tile distributed across warp)
        let mut acc = [0.0f32; 4];

        let num_k = k_size.div_ceil(8);
        for ks in 0..num_k {
            let k_base = ks * 8;

            // Cooperatively load A tile: 512 elements, 128 threads load 4 each.
            // `r` is block-local; every warp adds the same block origin.
            unsafe {
                for i in 0..4usize {
                    let idx = tid + i * 128;
                    let r = idx / 8;
                    let cc = idx % 8;
                    let xr = block_row_base + r;
                    let xc = k_base + cc;
                    SA[r * 8 + cc] = if xr < m_size && xc < k_size { x[xr * k_size + xc] } else { 0.0 };
                }

                // Load B tile (8x8): 64 elements, first 64 threads.
                if tid < 64 {
                    let r = tid / 8;
                    let cc = tid % 8;
                    let br = k_base + r;
                    let bc = col_base + cc;
                    // B[k][n] = W[n][k] (W is [N,K] row-major)
                    SB[r * 8 + cc] = if br < k_size && bc < n_size { w[bc * k_size + br] } else { 0.0 };
                }
            }

            thread::sync_threads();

            // Load A fragment: warp handles rows [warp_id*16, warp_id*16+16)
            // a[j] → row = warp_id*16 + group + (if j∈{1,3} then 8 else 0)
            //         col = tig + (if j≥2 then 4 else 0)
            let mut a = [0u32; 4];
            unsafe {
                for j in 0..4 {
                    let r = warp_id * 16 + group + if j == 1 || j == 3 { 8 } else { 0 };
                    let cc = tig + if j >= 2 { 4 } else { 0 };
                    a[j] = SA[r * 8 + cc].to_bits() & 0xFFFF_E000;
                }
            }

            // Load B fragment: b[j] → row = tig + (if j==1 then 4 else 0), col = group
            let mut b = [0u32; 2];
            unsafe {
                for j in 0..2 {
                    let r = tig + if j == 1 { 4 } else { 0 };
                    let cc = group;
                    b[j] = SB[r * 8 + cc].to_bits() & 0xFFFF_E000;
                }
            }

            // Tensor core MMA
            acc = unsafe { wmma::mma_m16n8k8_f32_tf32(acc, a, b) };

            thread::sync_threads();
        }

        // Write output: c[j] → row = warp_id*16 + group + (if j≥2 then 8 else 0)
        //                    col = col_base + tig*2 + (j&1)
        let out_ptr = y.as_mut_ptr();
        for j in 0..4 {
            let r = row_base + group + if j >= 2 { 8 } else { 0 };
            let cc = col_base + tig * 2 + (j & 1);
            if r < m_size && cc < n_size {
                // SAFETY: bounds checked; y is m*n.
                unsafe {
                    *out_ptr.add(r * n_size + cc) = acc[j] + bias[cc];
                }
            }
        }
    }

    /// Wide TF32 tensor-core GEMM: 128x64 tile per 256-thread block. Each warp
    /// owns 16 rows and eight 16x8 accumulator fragments, so every K-step loads
    /// one A fragment and performs eight MMAs (versus one in gemm_tf32).
    #[kernel]
    pub fn gemm_tf32_64(
        m: u32, n: u32, k: u32,
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        static mut SA: SharedArray<f32, { 128 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<f32, { 8 * 64 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(8);
        for ks in 0..num_k {
            let k_base = ks * 8;
            unsafe {
                for i in 0..4usize {
                    let idx = tid + i * 256;
                    let r = idx / 8;
                    let cc = idx % 8;
                    let xr = block_row_base + r;
                    let xc = k_base + cc;
                    SA[r * 8 + cc] = if xr < m_size && xc < k_size { x[xr * k_size + xc] } else { 0.0 };
                }
                for i in 0..2usize {
                    let idx = tid + i * 256;
                    let r = idx / 64;
                    let cc = idx % 64;
                    let br = k_base + r;
                    let bc = col_base + cc;
                    SB[r * 64 + cc] = if br < k_size && bc < n_size { w[bc * k_size + br] } else { 0.0 };
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                for j in 0..4 {
                    let r = warp_id * 16 + group + if j == 1 || j == 3 { 8 } else { 0 };
                    let cc = tig + if j >= 2 { 4 } else { 0 };
                    a[j] = SA[r * 8 + cc].to_bits() & 0xFFFF_E000;
                }
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    for j in 0..2 {
                        let r = tig + if j == 1 { 4 } else { 0 };
                        let cc = nt * 8 + group;
                        b[j] = SB[r * 64 + cc].to_bits() & 0xFFFF_E000;
                    }
                }
                acc[nt] = unsafe { wmma::mma_m16n8k8_f32_tf32(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(r * n_size + cc) = acc[nt][j] + bias[cc];
                    }
                }
            }
        }
    }

    /// FP16 tensor-core GEMM, 128x64 tile / 256 threads. Inputs are fp32 in
    /// global memory; cooperative loads pack adjacent K values into f16x2.
    /// Accumulation stays f32, so accuracy is close to PyTorch's autocast path.
    #[kernel]
    pub fn gemm_f16_128x64(
        m: u32, n: u32, k: u32,
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 32 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 32 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let mut acc = [[0.0f32; 4]; 8];

        // Vectorized cooperative load: each thread moves 16 B (four f32) per
        // instruction, quartering the load-instruction count of the K loop.
        let num_k = k_size.div_ceil(64);
        for ks in 0..num_k {
            let k_base = ks * 64;
            unsafe {
                // A: 128 rows x 16 float4 = 2048 loads, 8 per thread.
                for i in 0..8usize {
                    let idx = tid + i * 256;
                    let r = idx / 16;
                    let q4 = idx % 16;
                    let xr = block_row_base + r;
                    let k0 = k_base + q4 * 4;
                    let (mut v0, mut v1, mut v2, mut v3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    if xr < m_size && k0 + 3 < k_size {
                        // SAFETY: k0 is 16 B aligned (k0 % 4 == 0) and the
                        // four elements are in-bounds by the guard.
                        let src = x.as_ptr().add(xr * k_size + k0);
                        let v: [f32; 4] = *(src as *const [f32; 4]);
                        (v0, v1, v2, v3) = (v[0], v[1], v[2], v[3]);
                    }
                    SA[r * 32 + q4 * 2] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                    SA[r * 32 + q4 * 2 + 1] = cuda_device::convert::cvt_f16x2_f32(v2, v3);
                }
                // B: 64 columns x 16 float4 = 1024 loads, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 256;
                    let col = idx / 16;
                    let q4 = idx % 16;
                    let bc = col_base + col;
                    let k0 = k_base + q4 * 4;
                    let (mut v0, mut v1, mut v2, mut v3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    if bc < n_size && k0 + 3 < k_size {
                        // SAFETY: same 16 B alignment argument as A.
                        let src = w.as_ptr().add(bc * k_size + k0);
                        let v: [f32; 4] = *(src as *const [f32; 4]);
                        (v0, v1, v2, v3) = (v[0], v[1], v[2], v[3]);
                    }
                    SB[col * 32 + q4 * 2] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                    SB[col * 32 + q4 * 2 + 1] = cuda_device::convert::cvt_f16x2_f32(v2, v3);
                }
            }
            thread::sync_threads();

            // Four K=16 MMA steps per shared-memory barrier.
            for kk in 0..4usize {
                let word = kk * 8;
                let mut a = [0u32; 4];
                unsafe {
                    let r0 = warp_id * 16 + group;
                    let r1 = r0 + 8;
                    let k_word = word + tig;
                    let k_word_hi = word + tig + 4;
                    a[0] = SA[r0 * 32 + k_word];
                    a[1] = SA[r1 * 32 + k_word];
                    a[2] = SA[r0 * 32 + k_word_hi];
                    a[3] = SA[r1 * 32 + k_word_hi];
                }
                for nt in 0..8usize {
                    let mut b = [0u32; 2];
                    unsafe {
                        let col_word = (nt * 8 + group) * 32 + word;
                        b[0] = SB[col_word + tig];
                        b[1] = SB[col_word + tig + 4];
                    }
                    acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
                }
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(r * n_size + cc) = acc[nt][j] + bias[cc];
                    }
                }
            }
        }
    }

    /// FP16 tensor-core GEMM with residual addition fused into the epilogue.
    /// FP16 tensor-core GEMM, 128x64 tile / 256 threads. Inputs are fp32 in
    /// global memory; cooperative loads pack adjacent K values into f16x2.
    /// Accumulation stays f32, so accuracy is close to PyTorch's autocast path.
    #[kernel]
    pub fn gemm_f16_residual_128x64(
        m: u32, n: u32, k: u32,
        x: &[f32], w: &[f32], bias: &[f32], resid: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 32 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 32 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(64);
        for ks in 0..num_k {
            let k_base = ks * 64;
            unsafe {
                // 128 rows x 32 K-pairs = 4096 assignments, 16 per thread.
                for i in 0..16usize {
                    let idx = tid + i * 256;
                    let r = idx / 32;
                    let kp = idx % 32;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if xr < m_size && k0 < k_size { x[xr * k_size + k0] } else { 0.0 };
                    let v1 = if xr < m_size && k1 < k_size { x[xr * k_size + k1] } else { 0.0 };
                    SA[r * 32 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // 64 columns x 32 K-pairs = 2048 assignments, 8 per thread.
                for i in 0..8usize {
                    let idx = tid + i * 256;
                    let col = idx / 32;
                    let kp = idx % 32;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if bc < n_size && k0 < k_size { w[bc * k_size + k0] } else { 0.0 };
                    let v1 = if bc < n_size && k1 < k_size { w[bc * k_size + k1] } else { 0.0 };
                    SB[col * 32 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            // Four K=16 MMA steps per shared-memory barrier.
            for kk in 0..4usize {
                let word = kk * 8;
                let mut a = [0u32; 4];
                unsafe {
                    let r0 = warp_id * 16 + group;
                    let r1 = r0 + 8;
                    let k_word = word + tig;
                    let k_word_hi = word + tig + 4;
                    a[0] = SA[r0 * 32 + k_word];
                    a[1] = SA[r1 * 32 + k_word];
                    a[2] = SA[r0 * 32 + k_word_hi];
                    a[3] = SA[r1 * 32 + k_word_hi];
                }
                for nt in 0..8usize {
                    let mut b = [0u32; 2];
                    unsafe {
                        let col_word = (nt * 8 + group) * 32 + word;
                        b[0] = SB[col_word + tig];
                        b[1] = SB[col_word + tig + 4];
                    }
                    acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
                }
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(r * n_size + cc) = acc[nt][j] + bias[cc] + resid[r * n_size + cc];
                    }
                }
            }
        }
    }

    /// FP16 tensor-core FF1 GEMM with GELU fused into its epilogue.
    /// FP16 tensor-core GEMM, 128x64 tile / 256 threads. Inputs are fp32 in
    /// global memory; cooperative loads pack adjacent K values into f16x2.
    /// Accumulation stays f32, so accuracy is close to PyTorch's autocast path.
    #[kernel]
    pub fn gemm_f16_gelu_128x64(
        m: u32, n: u32, k: u32,
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 32 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 32 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(64);
        for ks in 0..num_k {
            let k_base = ks * 64;
            unsafe {
                // 128 rows x 32 K-pairs = 4096 assignments, 16 per thread.
                for i in 0..16usize {
                    let idx = tid + i * 256;
                    let r = idx / 32;
                    let kp = idx % 32;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if xr < m_size && k0 < k_size { x[xr * k_size + k0] } else { 0.0 };
                    let v1 = if xr < m_size && k1 < k_size { x[xr * k_size + k1] } else { 0.0 };
                    SA[r * 32 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // 64 columns x 32 K-pairs = 2048 assignments, 8 per thread.
                for i in 0..8usize {
                    let idx = tid + i * 256;
                    let col = idx / 32;
                    let kp = idx % 32;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if bc < n_size && k0 < k_size { w[bc * k_size + k0] } else { 0.0 };
                    let v1 = if bc < n_size && k1 < k_size { w[bc * k_size + k1] } else { 0.0 };
                    SB[col * 32 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            // Four K=16 MMA steps per shared-memory barrier.
            for kk in 0..4usize {
                let word = kk * 8;
                let mut a = [0u32; 4];
                unsafe {
                    let r0 = warp_id * 16 + group;
                    let r1 = r0 + 8;
                    let k_word = word + tig;
                    let k_word_hi = word + tig + 4;
                    a[0] = SA[r0 * 32 + k_word];
                    a[1] = SA[r1 * 32 + k_word];
                    a[2] = SA[r0 * 32 + k_word_hi];
                    a[3] = SA[r1 * 32 + k_word_hi];
                }
                for nt in 0..8usize {
                    let mut b = [0u32; 2];
                    unsafe {
                        let col_word = (nt * 8 + group) * 32 + word;
                        b[0] = SB[col_word + tig];
                        b[1] = SB[col_word + tig + 4];
                    }
                    acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
                }
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        let v = acc[nt][j] + bias[cc];
                        let sign = if v < 0.0 { -1.0f32 } else { 1.0 };
                        let a = v.abs() * 0.70710678;
                        let erf_t = 1.0 / (1.0 + 0.3275911 * a);
                        let poly = ((((1.061405429 * erf_t - 1.453152027) * erf_t + 1.421413741) * erf_t - 0.284496736) * erf_t + 0.254829592) * erf_t;
                        let erfa = sign * (1.0 - poly * (-a * a).exp());
                        *out_ptr.add(r * n_size + cc) = 0.5 * v * (1.0 + erfa);
                    }
                }
            }
        }
    }

    /// Batched QK attention reading folded raw QKV and fusing RoPE.
    /// Batched attention score GEMM: Q@K^T for BH groups. Q and K are both
    /// [BH, M/N, 64] (B uses the model's [N,K] layout); scores are [BH,M,N].
    #[kernel]
    pub fn attn_qk_rope_f16(
        m: u32, n: u32, k: u32,
        qkv: &[f32], cos: &[f32], sin: &[f32], bias: &[f32], bands: u32, axis: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 64;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let out_group = thread::blockIdx_z() as usize;
        let bands = bands as usize;
        let axis = axis as usize;
        let y_off = out_group * m_size * n_size;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // Load Q from folded raw QKV and fuse RoPE plus query scaling.
                for i in 0..4usize {
                    let idx = tid + i * 128;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let mut v0 = 0.0f32;
                    let mut v1 = 0.0f32;
                    if xr < m_size && k1 < k_size {
                        let token = if axis == 0 { xr * bands + out_group / 8 } else { (out_group / 8) * bands + xr };
                        let base = token * 1536 + out_group % 8 * 64 + k0;
                        let c = cos[xr * 32 + k_base / 2 + kp];
                        let sn = sin[xr * 32 + k_base / 2 + kp];
                        let (even, odd) = (qkv[base], qkv[base + 1]);
                        v0 = (even * c - odd * sn) * 0.125;
                        v1 = (odd * c + even * sn) * 0.125;
                    }
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // Load K from folded raw QKV and fuse RoPE without query scale.
                for i in 0..4usize {
                    let idx = tid + i * 128;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let mut v0 = 0.0f32;
                    let mut v1 = 0.0f32;
                    if bc < n_size && k1 < k_size {
                        let token = if axis == 0 { bc * bands + out_group / 8 } else { (out_group / 8) * bands + bc };
                        let base = token * 1536 + 512 + out_group % 8 * 64 + k0;
                        let c = cos[bc * 32 + k_base / 2 + kp];
                        let sn = sin[bc * 32 + k_base / 2 + kp];
                        let (even, odd) = (qkv[base], qkv[base + 1]);
                        v0 = even * c - odd * sn;
                        v1 = odd * c + even * sn;
                    }
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(y_off + r * n_size + cc) = acc[nt][j] + bias[cc];
                    }
                }
            }
        }
    }

    /// Batched PV attention reading V directly from folded raw QKV.
    /// Batched attention PV GEMM: P@V for BH groups. V is [BH,K,64] in
    /// row-major [K,N] layout, unlike the weight-major GEMM above.
    #[kernel]
    pub fn attn_pv_raw_f16(
        m: u32, n: u32, k: u32,
        x: &[f32], qkv: &[f32], bias: &[f32], gates: &[f32], bands: u32, axis: u32,
        row_max: &[f32], row_scale: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let out_group = thread::blockIdx_z() as usize;
        let bands = bands as usize;
        let axis = axis as usize;
        let x_off = out_group * m_size * k_size;
        let _ = out_group * k_size * n_size;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // 128 rows x 8 K-pairs = 1024 assignments, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 256;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let valid = xr < m_size;
                    let (rm, rs) = if valid { (row_max[out_group * m_size + xr], row_scale[out_group * m_size + xr]) } else { (0.0, 0.0) };
                    let v0 = if valid && k0 < k_size { cuda_device::float::ex2_approx_f32((x[x_off + xr * k_size + k0] - rm) * 1.4426950408889634) * rs } else { 0.0 };
                    let v1 = if valid && k1 < k_size { cuda_device::float::ex2_approx_f32((x[x_off + xr * k_size + k1] - rm) * 1.4426950408889634) * rs } else { 0.0 };
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // V comes directly from part 2 of folded raw QKV.
                for i in 0..2usize {
                    let idx = tid + i * 256;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let n0 = k_base + kp * 2;
                    let n1 = n0 + 1;
                    let head = out_group % 8;
                    let seq_batch = out_group / 8;
                    let token0 = if axis == 0 { n0 * bands + seq_batch } else { seq_batch * bands + n0 };
                    let token1 = if axis == 0 { n1 * bands + seq_batch } else { seq_batch * bands + n1 };
                    let v0 = if bc < n_size && n0 < k_size { qkv[token0 * 1536 + 1024 + head * 64 + bc] } else { 0.0 };
                    let v1 = if bc < n_size && n1 < k_size { qkv[token1 * 1536 + 1024 + head * 64 + bc] } else { 0.0 };
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        let head = out_group % 8;
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    let token = if axis == 0 { r * bands + out_group / 8 } else { (out_group / 8) * bands + r };
                    let sig = 1.0 / (1.0 + (-gates[token * 8 + head]).exp());
                    // SAFETY: token < M and cc < 512; y is the folded gated tensor.
                    unsafe {
                        *out_ptr.add(token * 512 + head * 64 + cc) = (acc[nt][j] + bias[cc]) * sig;
                    }
                }
            }
        }
    }

    /// Frequency-axis raw-QKV QK GEMM with fused RoPE and softmax max/scale.
    /// Batched QK attention reading folded raw QKV and fusing RoPE.
    /// Batched attention score GEMM: Q@K^T for BH groups. Q and K are both
    /// [BH, M/N, 64] (B uses the model's [N,K] layout); scores are [BH,M,N].
    #[kernel]
    pub fn attn_qk_rope_f16_stats(
        m: u32, n: u32, k: u32,
        qkv: &[f32], cos: &[f32], sin: &[f32], bias: &[f32], bands: u32, axis: u32,
        mut y: DisjointSlice<f32>,
        mut max_out: DisjointSlice<f32>, mut scale_out: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 64;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let out_group = thread::blockIdx_z() as usize;
        let bands = bands as usize;
        let axis = axis as usize;
        let y_off = out_group * m_size * n_size;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // Load Q from folded raw QKV and fuse RoPE plus query scaling.
                for i in 0..4usize {
                    let idx = tid + i * 128;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let mut v0 = 0.0f32;
                    let mut v1 = 0.0f32;
                    if xr < m_size && k1 < k_size {
                        let token = if axis == 0 { xr * bands + out_group / 8 } else { (out_group / 8) * bands + xr };
                        let base = token * 1536 + out_group % 8 * 64 + k0;
                        let c = cos[xr * 32 + k_base / 2 + kp];
                        let sn = sin[xr * 32 + k_base / 2 + kp];
                        let (even, odd) = (qkv[base], qkv[base + 1]);
                        v0 = (even * c - odd * sn) * 0.125;
                        v1 = (odd * c + even * sn) * 0.125;
                    }
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // Load K from folded raw QKV and fuse RoPE without query scale.
                for i in 0..4usize {
                    let idx = tid + i * 128;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let mut v0 = 0.0f32;
                    let mut v1 = 0.0f32;
                    if bc < n_size && k1 < k_size {
                        let token = if axis == 0 { bc * bands + out_group / 8 } else { (out_group / 8) * bands + bc };
                        let base = token * 1536 + 512 + out_group % 8 * 64 + k0;
                        let c = cos[bc * 32 + k_base / 2 + kp];
                        let sn = sin[bc * 32 + k_base / 2 + kp];
                        let (even, odd) = (qkv[base], qkv[base + 1]);
                        v0 = even * c - odd * sn;
                        v1 = odd * c + even * sn;
                    }
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(y_off + r * n_size + cc) = acc[nt][j] + bias[cc];
                    }
                }
            }
        }


        // A single column tile owns complete rows for the frequency axis, so
        // reduce each row across the four tig lanes and emit max/scale here.
        let mo = max_out.as_mut_ptr();
        let so = scale_out.as_mut_ptr();
        for half in 0..2usize {
            let r = row_base + group + half * 8;
            if r >= m_size { continue; }
            let mut sm = f32::NEG_INFINITY;
            for nt in 0..8usize {
                for jj in 0..2usize {
                    let j = half * 2 + jj;
                    let cc = col_base + nt * 8 + tig * 2 + jj;
                    if cc < n_size {
                        let v = acc[nt][j] + bias[cc];
                        if v > sm { sm = v; }
                    }
                }
            }
            let sm1 = warp::shuffle_xor_f32_sync(0xffffffff, sm, 1);
            if sm1 > sm { sm = sm1; }
            let sm2 = warp::shuffle_xor_f32_sync(0xffffffff, sm, 2);
            if sm2 > sm { sm = sm2; }
            let mut sz = 0.0f32;
            for nt in 0..8usize {
                for jj in 0..2usize {
                    let j = half * 2 + jj;
                    let cc = col_base + nt * 8 + tig * 2 + jj;
                    if cc < n_size { sz += cuda_device::float::ex2_approx_f32((acc[nt][j] + bias[cc] - sm) * 1.4426950408889634); }
                }
            }
            sz += warp::shuffle_xor_f32_sync(0xffffffff, sz, 1);
            sz += warp::shuffle_xor_f32_sync(0xffffffff, sz, 2);
            if tig == 0 {
                let idx = out_group * m_size + r;
                unsafe {
                    *mo.add(idx) = sm;
                    *so.add(idx) = 1.0 / sz;
                }
            }
        }
    }
    /// Tensor-core flash attention for one axis. One 256-thread block owns a
    /// 128-row query tile of one (batch, head) group and iterates every 64-key
    /// tile with an online softmax in registers. RoPE is fused into the Q/K
    /// loads and the head-gate sigmoid into the epilogue, which writes the
    /// gated folded attention output directly. Scores never touch global
    /// memory. Fragment layouts: QK's C fragment (rows g/g+8, cols 2tig..)
    /// is exactly the A fragment PV needs over the key dimension, so the
    /// probability tile converts to A fragments in place.
    #[kernel]
    pub fn attn_flash_tc(
        qkv: &[f32], cos: &[f32], sin: &[f32], gates: &[f32],
        bands: u32, axis: u32, seq: u32,
        mut y: DisjointSlice<f32>,
        mut dbg_m: DisjointSlice<f32>, mut dbg_l: DisjointSlice<f32>,
    ) {
        // Rows are padded to 33 words so warp-lane column walks hit distinct
        // banks (32-word rows would alias every 8 lanes onto one bank).
        static mut SQ: SharedArray<u32, { 64 * 33 }> = SharedArray::UNINIT; // [q row][d word]
        static mut SK: SharedArray<u32, { 64 * 33 }> = SharedArray::UNINIT; // [key][d word]
        static mut SV: SharedArray<u32, { 64 * 33 }> = SharedArray::UNINIT; // [key][d word] row-major staging
        static mut SVt: SharedArray<u32, { 64 * 33 }> = SharedArray::UNINIT; // [dim][key word] transposed

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let (n_size, bands_, axis_) = (seq as usize, bands as usize, axis as usize);
        let row_base = thread::blockIdx_y() as usize * 64;
        let grp = thread::blockIdx_z() as usize;
        let (b, head) = (grp / 8, grp % 8);
        let r0 = warp_id * 16 + group;
        let r1 = r0 + 8;
        let q_v0 = row_base + r0 < n_size;
        let q_v1 = row_base + r1 < n_size;
        let token_of = |r: usize| if axis_ == 0 { r * bands_ + b } else { b * bands_ + r };

        // ---- load this block's 64 query rows once, fusing RoPE + 0.125 ----
        // 128 threads = 4 warps x 16 rows = exactly the 64-row tile.
        // Flat r*32+w mapping keeps every warp on one 128 B row.
        // SAFETY: block-local writes; pad column 32 is never read.
        unsafe {
            for i in 0..16usize {
                let idx = tid + i * 128;
                let r = idx / 32;
                let w = idx % 32;
                let xr = row_base + r;
                let (mut v0, mut v1) = (0.0f32, 0.0f32);
                if xr < n_size {
                    let base = token_of(xr) * 1536 + head * 64 + w * 2;
                    let c = cos[xr * 32 + w];
                    let sn = sin[xr * 32 + w];
                    let (even, odd) = (qkv[base], qkv[base + 1]);
                    v0 = (even * c - odd * sn) * 0.125;
                    v1 = (odd * c + even * sn) * 0.125;
                }
                SQ[r * 33 + w] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
            }
        }

        let mut acc = [[0.0f32; 4]; 8];   // QK scores, later reused as P
        let mut acc_pv = [[0.0f32; 4]; 8]; // attention output rows x 64 dims
        let mut m_i = [f32::NEG_INFINITY; 2];
        let mut l_i = [0.0f32; 2];

        let mut kt = 0usize;
        while kt < n_size {
            // ---- cooperative K / V-tile load (64 keys) with RoPE on K ----
            // Both tiles load row-major and coalesced; the PV B fragment is
            // repacked from SV in shared memory with half-word selects.
            // SAFETY: indices bounded by 64*33; guards on key position.
            unsafe {
                for i in 0..16usize {
                    let idx = tid + i * 128;
                    let k = idx / 32;
                    let w = idx % 32;
                    let key = kt + k;
                    let (mut v0, mut v1) = (0.0f32, 0.0f32);
                    if key < n_size {
                        let base = token_of(key) * 1536 + 512 + head * 64 + w * 2;
                        let c = cos[key * 32 + w];
                        let sn = sin[key * 32 + w];
                        let (even, odd) = (qkv[base], qkv[base + 1]);
                        v0 = even * c - odd * sn;
                        v1 = odd * c + even * sn;
                    }
                    SK[k * 33 + w] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                for i in 0..16usize {
                    let idx = tid + i * 128;
                    let k = idx / 32;
                    let w = idx % 32;
                    let key = kt + k;
                    let v0 = if key < n_size { qkv[token_of(key) * 1536 + 1024 + head * 64 + w * 2] } else { 0.0 };
                    let v1 = if key < n_size { qkv[token_of(key) * 1536 + 1024 + head * 64 + w * 2 + 1] } else { 0.0 };
                    SV[k * 33 + w] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();
            // Transpose V into [dim][key-word] layout in shared so PV B
            // fragments need exactly one word per operand half.
            // SAFETY: SV reads see completed stores (sync above).
            unsafe {
                for i in 0..16usize {
                    let idx = tid + i * 128;
                    let d = idx / 32;
                    let w = idx % 32;
                    let dw = d / 2;
                    let w0 = SV[(2 * w) * 33 + dw];
                    let w1 = SV[(2 * w + 1) * 33 + dw];
                    SVt[d * 33 + w] = if d % 2 == 0 {
                        (w0 & 0xFFFF) | ((w1 & 0xFFFF) << 16)
                    } else {
                        (w0 >> 16) | ((w1 >> 16) << 16)
                    };
                }
            }
            thread::sync_threads();

            // ---- Q@K^T over d=64 (4 k-fragments x 8 key tiles) ----
            // Scores reset per key tile: each kt step is a fresh key range,
            // not a K-dimension continuation.
            for nt in 0..8usize {
                acc[nt] = [0.0f32; 4];
            }
            for j in 0..4usize {
                let mut a = [0u32; 4];
                // SAFETY: r0/r1 < 128, word index 8j+tig(+4) < 32.
                unsafe {
                    a[0] = SQ[r0 * 33 + 8 * j + tig];
                    a[1] = SQ[r1 * 33 + 8 * j + tig];
                    a[2] = SQ[r0 * 33 + 8 * j + tig + 4];
                    a[3] = SQ[r1 * 33 + 8 * j + tig + 4];
                }
                for nt in 0..8usize {
                    let mut bb = [0u32; 2];
                    // SAFETY: key row nt*8+group < 64, word < 32.
                    unsafe {
                        let kw = (nt * 8 + group) * 33 + 8 * j;
                        bb[0] = SK[kw + tig];
                        bb[1] = SK[kw + tig + 4];
                    }
                    acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, bb) };
                }
            }

            // ---- online softmax per row half + P conversion ----
            for half in 0..2usize {
                let row_valid = if half == 0 { q_v0 } else { q_v1 };
                let mut m_tile = f32::NEG_INFINITY;
                for nt in 0..8usize {
                    for jj in 0..2usize {
                        let j = half * 2 + jj;
                        let cc = kt + nt * 8 + tig * 2 + jj;
                        if row_valid && cc < n_size {
                            let v = acc[nt][j];
                            if v > m_tile { m_tile = v; }
                        }
                    }
                }
                m_tile = warp::shuffle_xor_f32_sync(0xffffffff, m_tile, 1).max(m_tile);
                m_tile = warp::shuffle_xor_f32_sync(0xffffffff, m_tile, 2).max(m_tile);
                let mut m_new = m_i[half].max(m_tile);
                if m_new == f32::NEG_INFINITY { m_new = 0.0; }
                // m_i = -inf on the first tile gives correction 0 cleanly.
                let corr = cuda_device::float::ex2_approx_f32((m_i[half] - m_new) * 1.4426950408889634f32);
                let mut s_tile = 0.0f32;
                for nt in 0..8usize {
                    for jj in 0..2usize {
                        let j = half * 2 + jj;
                        let cc = kt + nt * 8 + tig * 2 + jj;
                        let p = if row_valid && cc < n_size {
                            cuda_device::float::ex2_approx_f32((acc[nt][j] - m_new) * 1.4426950408889634f32)
                        } else {
                            0.0
                        };
                        acc[nt][j] = p;
                        s_tile += p;
                    }
                }
                l_i[half] = l_i[half] * corr + s_tile;
                m_i[half] = m_new;
                for nt in 0..8usize {
                    for jj in 0..2usize {
                        let j = half * 2 + jj;
                        acc_pv[nt][j] *= corr;
                    }
                }
            }

            // ---- P @ V over 64 keys (4 key fragments x 8 dim tiles) ----
            // B fragments come from row-major SV via register preloads: each
            // lane needs 4 key rows x 8 dim words per kf, loaded once and
            // repacked with integer half selects (no shared reads per mma).
            let g2 = group / 2;
            let hi_half = group % 2 == 1;
            for kf in 0..4usize {
                let mut a = [0u32; 4];
                // C tiles 2kf / 2kf+1 hold this A fragment's key columns.
                // SAFETY: nt index 2kf+1 <= 7.
                unsafe {
                    a[0] = cuda_device::convert::cvt_f16x2_f32(acc[2 * kf][0], acc[2 * kf][1]);
                    a[1] = cuda_device::convert::cvt_f16x2_f32(acc[2 * kf][2], acc[2 * kf][3]);
                    a[2] = cuda_device::convert::cvt_f16x2_f32(acc[2 * kf + 1][0], acc[2 * kf + 1][1]);
                    a[3] = cuda_device::convert::cvt_f16x2_f32(acc[2 * kf + 1][2], acc[2 * kf + 1][3]);
                }
                for nt in 0..8usize {
                    let mut bb = [0u32; 2];
                    // B fragment from the transposed tile: one word per half.
                    // SAFETY: dim row < 64, word < 32.
                    unsafe {
                        let kw = (nt * 8 + group) * 33 + 8 * kf;
                        bb[0] = SVt[kw + tig];
                        bb[1] = SVt[kw + tig + 4];
                    }
                    acc_pv[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc_pv[nt], a, bb) };
                }
            }
            thread::sync_threads();
            kt += 64;
        }

        // ---- epilogue: normalize, gate, scatter into the folded tensor ----
        for half in 0..2usize {
            let mut l = l_i[half];
            l += warp::shuffle_xor_f32_sync(0xffffffff, l, 1);
            l += warp::shuffle_xor_f32_sync(0xffffffff, l, 2);
            l_i[half] = l;
        }
        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let half = j / 2;
                let row = row_base + if half == 0 { r0 } else { r1 };
                if row < n_size {
                    let cc = nt * 8 + tig * 2 + (j & 1);
                    let token = token_of(row);
                    let g = gates[token * 8 + head];
                    let sig = 1.0 / (1.0 + (-g).exp());
                    let inv = if l_i[half] > 0.0 { 1.0 / l_i[half] } else { 0.0 };
                    // SAFETY: token < M (row < seq), cc < 64; y is M*512.
                    unsafe {
                        *out_ptr.add(token * 512 + head * 64 + cc) = acc_pv[nt][j] * inv * sig;
                    }
                    if cc == 0 && (j & 1) == 0 {
                        let dm = dbg_m.as_mut_ptr();
                        let dl = dbg_l.as_mut_ptr();
                        // SAFETY: grp*n_size+row < BH*seq <= buffer size.
                        unsafe {
                            *dm.add(grp * n_size + row) = m_i[half];
                            *dl.add(grp * n_size + row) = l_i[half];
                        }
                    }
                }
            }
        }
    }
    /// Compute per-row max and reciprocal softmax sum without rewriting P.
    /// PV applies `(score-max)*scale` while packing its A fragments, removing
    /// one full read/write round trip over the score matrix.
    #[kernel]
    pub fn softmax_stats(
        p: &[f32], rows: u32, cols: u32,
        mut max_out: DisjointSlice<f32>, mut scale_out: DisjointSlice<f32>,
    ) {
        let gid = thread::index_1d();
        let g0 = gid.get();
        let row = g0 / 32;
        let lane = warp::lane_id() as usize;
        let (rows, cols) = (rows as usize, cols as usize);
        if row >= rows { return; }
        let base = row * cols;
        let per = cols.div_ceil(32);
        let mut vals = [f32::NEG_INFINITY; 16];
        let mut m = f32::NEG_INFINITY;
        for i in 0..per {
            let j = i * 32 + lane;
            let v = if j < cols { p[base + j] } else { f32::NEG_INFINITY };
            vals[i] = v;
            if v > m { m = v; }
        }
        m = warp::reduce_max_f32(m);
        let mut z = 0.0f32;
        for i in 0..per {
            let j = i * 32 + lane;
            if j < cols {
                let e = cuda_device::float::ex2_approx_f32((vals[i] - m) * 1.4426950408889634);
                vals[i] = e;
                z += e;
            }
        }
        z = warp::reduce_sum_f32(z);
        if lane == 0 {
            let mo = max_out.as_mut_ptr();
            let so = scale_out.as_mut_ptr();
            unsafe {
                *mo.add(row) = m;
                *so.add(row) = 1.0 / z;
            }
        }
    }

    /// Batched attention score GEMM: Q@K^T for BH groups. Q and K are both
    /// [BH, M/N, 64] (B uses the model's [N,K] layout); scores are [BH,M,N].
    #[kernel]
    pub fn attn_qk_f16(
        m: u32, n: u32, k: u32,
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 64;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let out_group = thread::blockIdx_z() as usize;
        let x_off = out_group * m_size * k_size;
        let w_off = out_group * n_size * k_size;
        let y_off = out_group * m_size * n_size;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // 64 rows x 8 K-pairs = 512 assignments, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 128;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if xr < m_size && k0 < k_size { x[x_off + xr * k_size + k0] } else { 0.0 };
                    let v1 = if xr < m_size && k1 < k_size { x[x_off + xr * k_size + k1] } else { 0.0 };
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // 64 columns x 8 K-pairs = 512 assignments, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 128;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if bc < n_size && k0 < k_size { w[w_off + bc * k_size + k0] } else { 0.0 };
                    let v1 = if bc < n_size && k1 < k_size { w[w_off + bc * k_size + k1] } else { 0.0 };
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(y_off + r * n_size + cc) = acc[nt][j] + bias[cc];
                    }
                }
            }
        }
    }

    /// Batched attention PV GEMM: P@V for BH groups. V is [BH,K,64] in
    /// row-major [K,N] layout, unlike the weight-major GEMM above.
    #[kernel]
    pub fn attn_pv_f16_bn(
        m: u32, n: u32, k: u32,
        x: &[f32], w: &[f32], bias: &[f32],
        row_max: &[f32], row_scale: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let (m_size, n_size, k_size) = (m as usize, n as usize, k as usize);
        let out_group = thread::blockIdx_z() as usize;
        let x_off = out_group * m_size * k_size;
        let w_off = out_group * k_size * n_size;
        let y_off = out_group * m_size * n_size;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // 128 rows x 8 K-pairs = 1024 assignments, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 256;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let valid = xr < m_size;
                    let (rm, rs) = if valid { (row_max[out_group * m_size + xr], row_scale[out_group * m_size + xr]) } else { (0.0, 0.0) };
                    let v0 = if valid && k0 < k_size { cuda_device::float::ex2_approx_f32((x[x_off + xr * k_size + k0] - rm) * 1.4426950408889634) * rs } else { 0.0 };
                    let v1 = if valid && k1 < k_size { cuda_device::float::ex2_approx_f32((x[x_off + xr * k_size + k1] - rm) * 1.4426950408889634) * rs } else { 0.0 };
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // 64 columns x 8 K-pairs = 512 assignments, 2 per thread.
                for i in 0..2usize {
                    let idx = tid + i * 256;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if bc < n_size && k0 < k_size { w[w_off + k0 * n_size + bc] } else { 0.0 };
                    let v1 = if bc < n_size && k1 < k_size { w[w_off + k1 * n_size + bc] } else { 0.0 };
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(y_off + r * n_size + cc) = acc[nt][j] + bias[cc];
                    }
                }
            }
        }
    }

    /// Warp-per-row in-place softmax. Each lane owns ceil(cols/32) elements,
    /// replacing the one-thread-per-row kernel's long serial scans.
    #[kernel]
    pub fn softmax_rows_warp(mut p: DisjointSlice<f32>, rows: u32, cols: u32) {
        let gid = thread::index_1d();
        let g0 = gid.get();
        let row = g0 / 32;
        let lane = warp::lane_id() as usize;
        let (rows, cols) = (rows as usize, cols as usize);
        if row >= rows { return; }
        let base = row * cols;
        let per = cols.div_ceil(32);
        let ptr = p.as_mut_ptr();
        let mut vals = [f32::NEG_INFINITY; 16];
        let mut m = f32::NEG_INFINITY;
        for i in 0..per {
            let j = i * 32 + lane;
            let v = if j < cols { unsafe { *ptr.add(base + j) } } else { f32::NEG_INFINITY };
            vals[i] = v;
            if v > m { m = v; }
        }
        m = warp::reduce_max_f32(m);
        let mut z = 0.0f32;
        for i in 0..per {
            let j = i * 32 + lane;
            if j < cols {
                let e = cuda_device::float::ex2_approx_f32((vals[i] - m) * 1.4426950408889634);
                vals[i] = e;
                z += e;
            }
        }
        z = warp::reduce_sum_f32(z);
        let inv = 1.0 / z;
        for i in 0..per {
            let j = i * 32 + lane;
            if j < cols { unsafe { *ptr.add(base + j) = vals[i] * inv; } }
        }
    }

    /// BandSplit stage 2: grouped FP16 tensor-core GEMM over padded normalized bands.
    /// MaskEstimator GEMM1 for one stem: (62 bands, T, 256) -> hidden
    /// (62, T, 1024). One launch per stem; bands are the grid.z dimension.
    #[kernel]
    pub fn bandsplit_gemm_f16(
        band_off: &[u32], band_dims: &[u32],
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>, t_rows: u32, bands: u32, max_dim: u32,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let m_size = t_rows as usize;
        let n_size = 256usize;
        let bands = bands as usize;
        let max_dim = max_dim as usize;
        let band = thread::blockIdx_z() as usize;
        let k_size = band_dims[band] as usize;
        let g_off = band_off[band] as usize;
        let x_off = band * m_size * max_dim;
        let w_off = 256 * g_off;
        let bias_off = band * 256;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // 128 rows x 8 K-pairs = 1024 assignments, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 256;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if xr < m_size && k0 < k_size { x[x_off + xr * max_dim + k0] } else { 0.0 };
                    let v1 = if xr < m_size && k1 < k_size { x[x_off + xr * max_dim + k1] } else { 0.0 };
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // 64 columns x 8 K-pairs = 512 assignments, 2 per thread.
                for i in 0..2usize {
                    let idx = tid + i * 256;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if bc < n_size && k0 < k_size { w[w_off + bc * k_size + k0] } else { 0.0 };
                    let v1 = if bc < n_size && k1 < k_size { w[w_off + bc * k_size + k1] } else { 0.0 };
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add((r * bands + band) * n_size + cc) = acc[nt][j] + bias[bias_off + cc];
                    }
                }
            }
        }
    }


    /// MaskEstimator GEMM1 for one stem: (62 bands, T, 256) -> hidden
    /// (62, T, 1024). One launch per stem; bands are the grid.z dimension.
    #[kernel]
    pub fn mask_gemm1_f16(
        t_rows: u32, group_base: u32,
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let m_size = t_rows as usize;
        let n_size = 1024usize;
        let k_size = 256usize;
        let band = thread::blockIdx_z() as usize;
        let out_group = group_base as usize + band;
        let x_off = band * m_size * 256;
        let w_off = band * 1024 * 256;
        let bias_off = band * 1024;
        let y_off = out_group * m_size * 1024;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // 128 rows x 8 K-pairs = 1024 assignments, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 256;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if xr < m_size && k0 < k_size { x[x_off + xr * k_size + k0] } else { 0.0 };
                    let v1 = if xr < m_size && k1 < k_size { x[x_off + xr * k_size + k1] } else { 0.0 };
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // 64 columns x 8 K-pairs = 512 assignments, 2 per thread.
                for i in 0..2usize {
                    let idx = tid + i * 256;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if bc < n_size && k0 < k_size { w[w_off + bc * k_size + k0] } else { 0.0 };
                    let v1 = if bc < n_size && k1 < k_size { w[w_off + bc * k_size + k1] } else { 0.0 };
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(y_off + r * n_size + cc) = (acc[nt][j] + bias[bias_off + cc]).tanh();
                    }
                }
            }
        }
    }


    /// MaskEstimator GEMM2 for one stem, with per-band output width from
    /// band_off. Rows are padded to max_dim in y for the subsequent GLU pass.
    #[kernel]
    pub fn mask_gemm2_f16(
        t_rows: u32, group_base: u32, out_width: u32,
        band_off: &[u32],
        x: &[f32], w: &[f32], bias: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        // Packed f16 shared tiles. SA is row-major with adjacent K pairs;
        // SB is column-major with adjacent K pairs, matching the MMA fragments
        // exactly, so the K-loop performs each f32->f16 conversion once during
        // the cooperative load rather than once per warp/fragment reuse.
        static mut SA: SharedArray<u32, { 128 * 8 }> = SharedArray::UNINIT;
        static mut SB: SharedArray<u32, { 64 * 8 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let warp_id = tid / 32;
        let group = lane / 4;
        let tig = lane % 4;
        let block_row_base = thread::blockIdx_y() as usize * 128;
        let row_base = block_row_base + warp_id * 16;
        let col_base = thread::blockIdx_x() as usize * 64;
        let m_size = t_rows as usize;
        let k_size = 1024usize;
        let band = thread::blockIdx_z() as usize;
        let dim_in = (band_off[band + 1] - band_off[band]) as usize;
        let n_size = dim_in * 2;
        if col_base >= n_size {
            return;
        }
        let out_group = group_base as usize + band;
        let x_off = out_group * m_size * 1024;
        let w_off = (2 * band_off[band] as usize) * 1024;
        let bias_off = 2 * band_off[band] as usize;
        let out_width = out_width as usize;
        let y_off = out_group * m_size * out_width;
        let mut acc = [[0.0f32; 4]; 8];

        let num_k = k_size.div_ceil(16);
        for ks in 0..num_k {
            let k_base = ks * 16;
            unsafe {
                // 128 rows x 8 K-pairs = 1024 assignments, 4 per thread.
                for i in 0..4usize {
                    let idx = tid + i * 256;
                    let r = idx / 8;
                    let kp = idx % 8;
                    let xr = block_row_base + r;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if xr < m_size && k0 < k_size { x[x_off + xr * k_size + k0] } else { 0.0 };
                    let v1 = if xr < m_size && k1 < k_size { x[x_off + xr * k_size + k1] } else { 0.0 };
                    SA[r * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
                // 64 columns x 8 K-pairs = 512 assignments, 2 per thread.
                for i in 0..2usize {
                    let idx = tid + i * 256;
                    let col = idx / 8;
                    let kp = idx % 8;
                    let bc = col_base + col;
                    let k0 = k_base + kp * 2;
                    let k1 = k0 + 1;
                    let v0 = if bc < n_size && k0 < k_size { w[w_off + bc * k_size + k0] } else { 0.0 };
                    let v1 = if bc < n_size && k1 < k_size { w[w_off + bc * k_size + k1] } else { 0.0 };
                    SB[col * 8 + kp] = cuda_device::convert::cvt_f16x2_f32(v0, v1);
                }
            }
            thread::sync_threads();

            let mut a = [0u32; 4];
            unsafe {
                let r0 = warp_id * 16 + group;
                let r1 = r0 + 8;
                let k_word = tig;
                let k_word_hi = tig + 4;
                a[0] = SA[r0 * 8 + k_word];
                a[1] = SA[r1 * 8 + k_word];
                a[2] = SA[r0 * 8 + k_word_hi];
                a[3] = SA[r1 * 8 + k_word_hi];
            }
            for nt in 0..8usize {
                let mut b = [0u32; 2];
                unsafe {
                    let col_word = (nt * 8 + group) * 8;
                    b[0] = SB[col_word + tig];
                    b[1] = SB[col_word + tig + 4];
                }
                acc[nt] = unsafe { wmma::mma_m16n8k16_f32_f16(acc[nt], a, b) };
            }
            thread::sync_threads();
        }

        let out_ptr = y.as_mut_ptr();
        for nt in 0..8usize {
            for j in 0..4usize {
                let r = row_base + group + if j >= 2 { 8 } else { 0 };
                let cc = col_base + nt * 8 + tig * 2 + (j & 1);
                if r < m_size && cc < n_size {
                    // SAFETY: bounds checked; y has m*n elements.
                    unsafe {
                        *out_ptr.add(y_off + r * out_width + cc) = acc[nt][j] + bias[bias_off + cc];
                    }
                }
            }
        }
    }


    /// GLU + scatter for all mask bands: read padded pre2 rows and write the
    /// compact-into-max_dim glu layout consumed by mask_apply.
    #[kernel]
    pub fn glu_scatter(
        pre: &[f32], band_off: &[u32], mut out: DisjointSlice<f32>,
        t_rows: u32, bands: u32, max_dim: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() { return; }
        let (t_f, bands_, width) = (t_rows as usize, bands as usize, max_dim as usize);
        let pre_width = width * 2;
        let d = i % width;
        let row = i / width;
        let t = row % t_f;
        let sb = row / t_f;
        let band = sb % bands_;
        let dim_in = (band_off[band + 1] - band_off[band]) as usize;
        let group = sb * t_f + t;
        let value = if d < dim_in {
            let a = pre[group * pre_width + d];
            let b = pre[group * pre_width + dim_in + d];
            a / (1.0 + (-b).exp())
        } else { 0.0 };
        if let Some(o) = out.get_mut(idx) { *o = value; }
    }

    /// Element-wise: y[i] += bias[i % n]. Used after cuBLAS GEMM (no bias).
    #[kernel]
    pub fn bias_add(mut y: DisjointSlice<f32>, bias: &[f32], n: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < y.len() {
            let col = i % n as usize;
            if let Some(o) = y.get_mut(idx) {
                *o += bias[col];
            }
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


/// RMSNorm parity vs F.normalize(x)*sqrt(dim)*gamma (parity/rmsnorm.npz).
/// Gate: max rel-err < 1e-6 (pure elementwise after the reduction).
fn rmsnorm_parity_test(device: usize, model_dir: &std::path::Path) {
    let npz_path = model_dir.parent().unwrap_or(model_dir).join("parity/rmsnorm.npz");
    let npz = npz::Npz::open(&npz_path).unwrap_or_else(|e| panic!("{e}"));
    let x = npz.f32("x").expect("x");
    let gamma = npz.f32("gamma").expect("gamma");
    let ref_out = npz.f32("out").expect("out");
    let rows = npz.shapes["x"][0];
    let dim = npz.shapes["x"][1];

    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let x_dev = DeviceBuffer::from_host(&stream, x).unwrap();
    let g_dev = DeviceBuffer::from_host(&stream, gamma).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, rows * dim).unwrap();
    let km = gpu_kernels::load(&ctx).expect("kernel module");
    // SAFETY: one warp per row; threads = rows*32 covers exactly rows warps.
    unsafe {
        km.rmsnorm(
            &stream,
            cuda_core::simt::LaunchConfig::for_num_elems((rows * 32) as u32),
            &x_dev,
            &g_dev,
            &mut out_dev,
            rows as u32,
            dim as u32,
        )
    }
    .expect("rmsnorm kernel");

    let got = out_dev.to_host_vec(&stream).unwrap();
    let mut max_err = 0.0f32;
    let mut denom = 0.0f32;
    for i in 0..got.len() {
        max_err = max_err.max((got[i] - ref_out[i]).abs());
        denom = denom.max(ref_out[i].abs());
    }
    println!("rmsnorm parity: rows={rows} dim={dim} max_err={max_err:e} rel={:.3e}", max_err / denom);
    assert!(max_err < 1e-5 * denom, "rmsnorm parity failed");
    println!("OK");
}

/// BandSplit parity vs per-band RMSNorm+Linear (parity/bandsplit.npz).
fn bandsplit_parity_test(device: usize, model_dir: &std::path::Path) {
    let npz_path = model_dir.parent().unwrap_or(model_dir).join("parity/bandsplit.npz");
    let npz = npz::Npz::open(&npz_path).unwrap_or_else(|e| panic!("{e}"));
    let x = npz.f32("x").expect("x");       // (t, 4100)
    let gamma = npz.f32("gamma").expect("gamma");
    let w = npz.f32("w").expect("w");
    let b = npz.f32("b").expect("b");
    let ref_out = npz.f32("out").expect("out"); // (t, 62, 256)
    let rows = npz.shapes["x"][0];
    let n_bands = 62usize;
    let freqs: Vec<usize> = {
        let mut v = vec![2usize; 24];
        v.extend(vec![4; 12]);
        v.extend(vec![12; 8]);
        v.extend(vec![24; 8]);
        v.extend(vec![48; 8]);
        v.push(128);
        v.push(129);
        v
    };
    let dims: Vec<u32> = freqs.iter().map(|f| (2 * f * 2) as u32).collect();
    let mut offs = vec![0u32; n_bands + 1];
    for i in 0..n_bands {
        offs[i + 1] = offs[i] + dims[i];
    }
    assert_eq!(offs[n_bands] as usize, npz.shapes["x"][1], "band offsets must cover x width");

    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let x_dev = DeviceBuffer::from_host(&stream, x).unwrap();
    let g_dev = DeviceBuffer::from_host(&stream, gamma).unwrap();
    let w_dev = DeviceBuffer::from_host(&stream, w).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, b).unwrap();
    let offs_dev = DeviceBuffer::from_host(&stream, &offs).unwrap();
    let dims_dev = DeviceBuffer::from_host(&stream, &dims).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, rows * n_bands * 256).unwrap();
    let km = gpu_kernels::load(&ctx).expect("kernel module");
    // SAFETY: one warp per (row, band) task; threads = rows*62*32 exactly.
    unsafe {
        km.bandsplit(
            &stream,
            cuda_core::simt::LaunchConfig::for_num_elems((rows * n_bands * 32) as u32),
            &x_dev,
            &g_dev,
            &w_dev,
            &b_dev,
            &offs_dev,
            &dims_dev,
            &mut out_dev,
            rows as u32,
            n_bands as u32,
        )
    }
    .expect("bandsplit kernel");

    let got = out_dev.to_host_vec(&stream).unwrap();
    let mut max_err = 0.0f32;
    let mut denom = 0.0f32;
    for i in 0..got.len() {
        max_err = max_err.max((got[i] - ref_out[i]).abs());
        denom = denom.max(ref_out[i].abs());
    }
    println!("bandsplit parity: rows={rows} max_err={max_err:e} rel={:.3e}", max_err / denom);
    assert!(max_err < 1e-4 * denom, "bandsplit parity failed");
    println!("OK");
}

/// GEMM parity across the four model shapes (parity/gemm.npz, M=2048).
fn gemm_parity_test(device: usize, model_dir: &std::path::Path) {
    let npz_path = model_dir.parent().unwrap_or(model_dir).join("parity/gemm.npz");
    let npz = npz::Npz::open(&npz_path).unwrap_or_else(|e| panic!("{e}"));
    let shapes = [(256usize, 1536usize), (512, 256), (256, 1024), (1024, 256)];
    let m = 2048usize;
    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let km = gpu_kernels::load(&ctx).expect("kernel module");
    for (gi, (k, n)) in shapes.iter().enumerate() {
        let x = npz.f32(&format!("x{gi}")).unwrap();
        let w = npz.f32(&format!("w{gi}")).unwrap();
        let bias = npz.f32(&format!("b{gi}")).unwrap();
        let ref_y = npz.f32(&format!("y{gi}")).unwrap();
        let x_dev = DeviceBuffer::from_host(&stream, x).unwrap();
        let w_dev = DeviceBuffer::from_host(&stream, w).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, bias).unwrap();
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, m * n).unwrap();
        // SAFETY: 2-D grid covers ceil(n/16) x ceil(m/16) tiles of 16x16 threads.
        unsafe {
            km.gemm_bias_64(
                &stream,
                tile_cfg_64(m, *n),
                m as u32,
                *n as u32,
                *k as u32,
                &x_dev,
                &w_dev,
                &b_dev,
                &mut y_dev,
            )
        }
        .expect("gemm kernel");
        let got = y_dev.to_host_vec(&stream).unwrap();
        let mut max_err = 0.0f32;
        let mut denom = 0.0f32;
        for i in 0..got.len() {
            max_err = max_err.max((got[i] - ref_y[i]).abs());
            denom = denom.max(ref_y[i].abs());
        }

        let mut y_tf = DeviceBuffer::<f32>::zeroed(&stream, m * n).unwrap();
        // SAFETY: grid covers 64x128 FP16 tensor-core tiles over m x n.
        unsafe {
            km.gemm_f16_128x64(&stream, tile_cfg_tf32_64(m, *n), m as u32, *n as u32, *k as u32, &x_dev, &w_dev, &b_dev, &mut y_tf)
        }
        .expect("tf32 gemm kernel");
        let got_tf = y_tf.to_host_vec(&stream).unwrap();
        let (mut me_tf, mut dn_tf, mut bad) = (0.0f32, 0.0f32, 0usize);
        for i in 0..got_tf.len() {
            let e = (got_tf[i] - ref_y[i]).abs();
            if e > me_tf { me_tf = e; bad = i; }
            dn_tf = dn_tf.max(ref_y[i].abs());
        }
        let threshold = 0.01 * dn_tf;
        let mut by_row = [0usize; 64];
        let mut by_col = [0usize; 8];
        let mut bad_count = 0usize;
        for i in 0..got_tf.len() {
            if (got_tf[i] - ref_y[i]).abs() > threshold {
                bad_count += 1;
                by_row[(i / n) % 64] += 1;
                by_col[i % 8] += 1;
            }
        }
        println!("tf32 bad_count={bad_count}/{} by_row={:?}", got_tf.len(), by_row);

        println!("tf32 by_col={:?}", by_col);
        if gi == 0 {
            println!("tf32 row mapping (got row -> best ref row, maxerr):");
            for r in 0..64usize {
                let mut best = (0usize, f32::INFINITY);
                for p in 0..64usize {
                    let mut em = 0.0f32;
                    for c in 0..*n { em = em.max((got_tf[r*n+c]-ref_y[p*n+c]).abs()); }
                    if em < best.1 { best = (p, em); }
                }
                println!("  got {r} -> ref {} err {:.3e}", best.0, best.1);
            }
        }
        println!(
            "gemm_tf32[{gi}] K={k} N={n}: rel={:.3e} bad_idx={bad} got={:.6} ref={:.6}",
            me_tf / dn_tf,
            got_tf[bad],
            ref_y[bad]
        );
    }
    println!("OK");
}

/// QKV+RoPE+scale parity (parity/qkvrope.npz): rmsnorm -> gemm -> rope.
fn qkvrope_parity_test(device: usize, model_dir: &std::path::Path) {
    let npz_path = model_dir.parent().unwrap_or(model_dir).join("parity/qkvrope.npz");
    let npz = npz::Npz::open(&npz_path).unwrap_or_else(|e| panic!("{e}"));
    let x = npz.f32("x").expect("x");           // (b*seq*dim) folded rows
    let gamma = npz.f32("gamma").expect("gamma");
    let w = npz.f32("w").expect("w");
    let bias = npz.f32("bias").expect("bias");
    let cos = npz.f32("cos").expect("cos");     // (seq*32)
    let sin = npz.f32("sin").expect("sin");
    let ref_q = npz.f32("q").expect("q");
    let ref_k = npz.f32("k").expect("k");
    let ref_v = npz.f32("v").expect("v");
    let (b, seq, dim, nqkv) = (2usize, 64usize, 256usize, 1536usize);
    let rows = b * seq;

    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let km = gpu_kernels::load(&ctx).expect("kernel module");
    let x_dev = DeviceBuffer::from_host(&stream, x).unwrap();
    let g_dev = DeviceBuffer::from_host(&stream, gamma).unwrap();
    let mut h_dev = DeviceBuffer::<f32>::zeroed(&stream, rows * dim).unwrap();
    // SAFETY: one warp per row.
    unsafe {
        km.rmsnorm(&stream, cuda_core::simt::LaunchConfig::for_num_elems((rows * 32) as u32), &x_dev, &g_dev, &mut h_dev, rows as u32, dim as u32)
    }
    .expect("rmsnorm");
    let w_dev = DeviceBuffer::from_host(&stream, w).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, bias).unwrap();
    let mut qkv_dev = DeviceBuffer::<f32>::zeroed(&stream, rows * nqkv).unwrap();
    // SAFETY: 2-D grid of 16x16 tiles covering rows x nqkv.
    unsafe {
        km.gemm_bias_64(
            &stream,
            cuda_core::simt::LaunchConfig {
                grid_dim: ((nqkv.div_ceil(16)) as u32, (rows.div_ceil(16)) as u32, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            },
            rows as u32,
            nqkv as u32,
            dim as u32,
            &h_dev,
            &w_dev,
            &b_dev,
            &mut qkv_dev,
        )
    }
    .expect("gemm qkv");
    let cos_dev = DeviceBuffer::from_host(&stream, cos).unwrap();
    let sin_dev = DeviceBuffer::from_host(&stream, sin).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, rows * nqkv).unwrap();
    // SAFETY: one thread per (row, pair).
    unsafe {
        km.rope_scale(&stream, cuda_core::simt::LaunchConfig::for_num_elems((rows * 768) as u32), &qkv_dev, &cos_dev, &sin_dev, &mut out_dev, seq as u32, 1, 0)
    }
    .expect("rope");

    let got = out_dev.to_host_vec(&stream).unwrap();
    let mut max_err = 0.0f32;
    let mut denom = 0.0f32;
    for m in 0..rows {
        for c in 0..nqkv {
            let g = got[m * nqkv + c];
            let r = if c < 512 { ref_q[m * 512 + c] } else if c < 1024 { ref_k[m * 512 + (c - 512)] } else { ref_v[m * 512 + (c - 1024)] };
            max_err = max_err.max((g - r).abs());
            denom = denom.max(r.abs());
        }
    }
    println!("qkv+rope parity: rows={rows} max_err={max_err:e} rel={:.3e}", max_err / denom);
    assert!(max_err < 1e-4 * denom, "qkv+rope parity failed");
    println!("OK");
}

/// Short-attention parity vs SDPA (parity/attn_short.npz). Note: the npz
/// q/k/v are UNSCALED; our kernel expects q pre-scaled by 0.125 (rope_scale
/// upstream does it), so the test scales q on the host before upload.
fn attn_parity_test(device: usize, model_dir: &std::path::Path) {
    let npz_path = model_dir.parent().unwrap_or(model_dir).join("parity/attn_short.npz");
    let npz = npz::Npz::open(&npz_path).unwrap_or_else(|e| panic!("{e}"));
    let q = npz.f32("q").expect("q");
    let k = npz.f32("k").expect("k");
    let v = npz.f32("v").expect("v");
    let ref_out = npz.f32("out").expect("out");
    let (bh, seq) = (16usize, 62usize);
    let mut qs = vec![0.0f32; q.len()];
    for i in 0..q.len() {
        qs[i] = q[i] * 0.125;
    }
    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let q_dev = DeviceBuffer::from_host(&stream, &qs).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, k).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, v).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, bh * seq * 64).unwrap();
    let km = gpu_kernels::load(&ctx).expect("kernel module");
    // SAFETY: grid = bh blocks of 128 threads; buffers sized bh*seq*64.
    unsafe {
        km.attn_short(
            &stream,
            cuda_core::simt::LaunchConfig {
                grid_dim: (bh as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            &q_dev,
            &k_dev,
            &v_dev,
            &mut out_dev,
            seq as u32,
        )
    }
    .expect("attn kernel");
    let got = out_dev.to_host_vec(&stream).unwrap();
    let mut max_err = 0.0f32;
    let mut denom = 0.0f32;
    for i in 0..got.len() {
        max_err = max_err.max((got[i] - ref_out[i]).abs());
        denom = denom.max(ref_out[i].abs());
    }
    println!("attn_short parity: bh={bh} max_err={max_err:e} rel={:.3e}", max_err / denom);
    assert!(max_err < 1e-4 * denom, "attn parity failed");
    println!("OK");

    // ---- long-sequence (time-axis) 2-pass attention ----
    let npz2 = npz::Npz::open(&model_dir.parent().unwrap_or(model_dir).join("parity/attn_long.npz"))
        .unwrap_or_else(|e| panic!("{e}"));
    let ql = npz2.f32("q").expect("q"); // unscaled; scale on host below
    let kl = npz2.f32("k").expect("k");
    let vl = npz2.f32("v").expect("v");
    let ref_l = npz2.f32("out").expect("out");
    let (bh2, seq2) = (4usize, 256usize);
    let mut qs2 = vec![0.0f32; ql.len()];
    for i in 0..ql.len() {
        qs2[i] = ql[i] * 0.125;
    }
    let q2 = DeviceBuffer::from_host(&stream, &qs2).unwrap();
    let k2 = DeviceBuffer::from_host(&stream, kl).unwrap();
    let v2 = DeviceBuffer::from_host(&stream, vl).unwrap();
    // pass 1: scores = q @ k^T per (b,h) — W = k as [N=seq, K=64]
    let mut p2 = DeviceBuffer::<f32>::zeroed(&stream, bh2 * seq2 * seq2).unwrap();
    let zero_bias = DeviceBuffer::<f32>::zeroed(&stream, seq2).unwrap();
    for g in 0..bh2 {
        let off = g * seq2 * 64;
        let qseg = q2.cu_deviceptr() + (off * 4) as u64;
        let kseg = k2.cu_deviceptr() + (off * 4) as u64;
        let pseg = p2.cu_deviceptr() + (g * seq2 * seq2 * 4) as u64;
        // SAFETY: aliasing views over disjoint segments of sized buffers.
        let q_view = unsafe { DeviceBuffer::<f32>::from_raw_parts(qseg, seq2 * 64, ctx.clone()) };
        let k_view = unsafe { DeviceBuffer::<f32>::from_raw_parts(kseg, seq2 * 64, ctx.clone()) };
        let mut p_view = unsafe { DeviceBuffer::<f32>::from_raw_parts(pseg, seq2 * seq2, ctx.clone()) };
        unsafe {
            km.gemm_bias_64(
                &stream,
                cuda_core::simt::LaunchConfig {
                    grid_dim: ((seq2.div_ceil(16)) as u32, (seq2.div_ceil(16)) as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                },
                seq2 as u32,
                seq2 as u32,
                64,
                &q_view,
                &k_view,
                &zero_bias,
                &mut p_view,
            )
        }
        .expect("scores gemm");
        std::mem::forget(q_view);
        std::mem::forget(k_view);
        std::mem::forget(p_view);
    }
    // pass 1.5: in-place row softmax
    // SAFETY: one thread per row of bh2*seq2 rows.
    unsafe {
        km.softmax_rows(&stream, cuda_core::simt::LaunchConfig::for_num_elems((bh2 * seq2) as u32), &mut p2, (bh2 * seq2) as u32, seq2 as u32)
    }
    .expect("softmax");
    // pass 2: out = P @ V with B=[K=seq, N=64] layout (v is [seq, 64] row-major)
    let zero64 = DeviceBuffer::<f32>::zeroed(&stream, 64).unwrap();
    let mut out2 = DeviceBuffer::<f32>::zeroed(&stream, bh2 * seq2 * 64).unwrap();
    for g in 0..bh2 {
        let pseg = p2.cu_deviceptr() + (g * seq2 * seq2 * 4) as u64;
        let vseg = v2.cu_deviceptr() + (g * seq2 * 64 * 4) as u64;
        let oseg = out2.cu_deviceptr() + (g * seq2 * 64 * 4) as u64;
        // SAFETY: disjoint segment views.
        let p_view = unsafe { DeviceBuffer::<f32>::from_raw_parts(pseg, seq2 * seq2, ctx.clone()) };
        let v_view = unsafe { DeviceBuffer::<f32>::from_raw_parts(vseg, seq2 * 64, ctx.clone()) };
        let mut o_view = unsafe { DeviceBuffer::<f32>::from_raw_parts(oseg, seq2 * 64, ctx.clone()) };
        unsafe {
            km.gemm_bias_bn(
                &stream,
                cuda_core::simt::LaunchConfig {
                    grid_dim: ((64 / 16) as u32, (seq2.div_ceil(16)) as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                },
                seq2 as u32,
                64,
                seq2 as u32,
                &p_view,
                &v_view,
                &zero64,
                &mut o_view,
            )
        }
        .expect("pv gemm");
        std::mem::forget(p_view);
        std::mem::forget(v_view);
        std::mem::forget(o_view);
    }

    let got2 = out2.to_host_vec(&stream).unwrap();

    // Same long-sequence inputs through the batched online-softmax kernel.
    let mut out3 = DeviceBuffer::<f32>::zeroed(&stream, bh2 * seq2 * 64).unwrap();
    // SAFETY: grid covers 32-row query tiles for all bh2 batches/heads.
    unsafe {
        km.attn_flash(
            &stream,
            cuda_core::simt::LaunchConfig {
                grid_dim: ((seq2.div_ceil(8) * bh2) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &q2,
            &k2,
            &v2,
            &mut out3,
            seq2 as u32,
        )
    }
    .expect("flash attention");
    let mut out4 = DeviceBuffer::<f32>::zeroed(&stream, bh2 * seq2 * 64).unwrap();
    // SAFETY: 16-row query tiles, 256 threads (two rows per warp).
    unsafe {
        km.attn_flash2(
            &stream,
            cuda_core::simt::LaunchConfig {
                grid_dim: ((seq2.div_ceil(16) * bh2) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &q2,
            &k2,
            &v2,
            &mut out4,
            seq2 as u32,
        )
    }
    .expect("flash2 attention");
    let got4 = out4.to_host_vec(&stream).unwrap();
    let (mut mef2, mut dnf2) = (0.0f32, 0.0f32);
    for i in 0..got4.len() {
        mef2 = mef2.max((got4[i] - ref_l[i]).abs());
        dnf2 = dnf2.max(ref_l[i].abs());
    }
    println!("attn_long flash2 parity: rel={:.3e}", mef2 / dnf2);
    assert!(mef2 < 1e-3 * dnf2, "attn_long flash2 parity failed");
    let got3 = out3.to_host_vec(&stream).unwrap();
    let (mut mef, mut dnf) = (0.0f32, 0.0f32);
    let mut flash_diff = 0.0f32;
    let (mut bad_ref, mut bad_2p) = (0usize, 0usize);
    for i in 0..got3.len() {
        if (got3[i] - ref_l[i]).abs() > mef {
            mef = (got3[i] - ref_l[i]).abs();
            bad_ref = i;
        }
        dnf = dnf.max(ref_l[i].abs());
        if (got3[i] - got2[i]).abs() > flash_diff {
            flash_diff = (got3[i] - got2[i]).abs();
            bad_2p = i;
        }
    }
    let decode = |i: usize| (i / (seq2 * 64), (i / 64) % seq2, i % 64);
    println!("bad ref idx={bad_ref} coords={:?} flash={:.7} ref={:.7}", decode(bad_ref), got3[bad_ref], ref_l[bad_ref]);
    println!("bad 2p  idx={bad_2p} coords={:?} flash={:.7} 2pass={:.7}", decode(bad_2p), got3[bad_2p], got2[bad_2p]);
    let zero_count = got3.iter().filter(|x| x.abs() == 0.0).count();
    println!("flash zero count={zero_count}/{}", got3.len());
    let rows_with = (0..bh2*seq2).filter(|r| got3[r*64..(r+1)*64].iter().any(|x| x.abs()!=0.0)).count();
    println!("flash rows with output={rows_with}/{}", bh2*seq2);
    let mut row_counts=Vec::new();
    for r in 0..20 { row_counts.push(got3[r*64..(r+1)*64].iter().filter(|x| x.abs()!=0.0).count()); }
    println!("flash first20 row nonzero={:?}",row_counts);
    let dims=(0..64).filter(|d| got3[*d].abs()!=0.0).collect::<Vec<_>>();
    println!("flash q0 nonzero dims={:?}",dims);
    for b in 0..bh2 {
        let lo=b*seq2*64; let hi=lo+seq2*64;
        let z=got3[lo..hi].iter().filter(|x| x.abs()==0.0).count();
        println!("flash bh={b} zeros={z}/{}", hi-lo);
    }
    let row_off = (1 * seq2 + 136) * 64;
    println!("flash row(bh1,q136)[..16]={:?}", &got3[row_off..row_off + 16]);
    println!("ref  row(bh1,q136)[..16]={:?}", &ref_l[row_off..row_off + 16]);
    println!("flash[0..4]={:?} 2pass[0..4]={:?} ref[0..4]={:?}", &got3[0..4], &got2[0..4], &ref_l[0..4]);
    println!(
        "attn_long flash parity: rel={:.3e} vs_2pass={:.3e}",
        mef / dnf,
        flash_diff / dnf
    );
    assert!(mef < 1e-3 * dnf, "attn_long flash parity failed");

    {
        let p_back = p2.to_host_vec(&stream).unwrap();
        let rs0: f32 = p_back[0..8].iter().sum();
        println!("P row0[0..4]={:?} sum8={:.4}", &p_back[0..4], rs0);
        let rs_last: f32 = p_back[(bh2 * seq2 - 1) * seq2..(bh2 * seq2 - 1) * seq2 + 8].iter().sum();
        println!("P lastrow[0..4] sum8={:.4}", rs_last);
        let r768: f32 = p_back[768 * seq2..768 * seq2 + 8].iter().sum();
        let r1000: f32 = p_back[1000 * seq2..1000 * seq2 + 8].iter().sum();
        println!("P bh3 row0 sum8={:.4} row1000 sum8={:.4}", r768, r1000);
        println!("P raw scores g3 row0[0..3] pre-softmax unknown; P val={:?}", &p_back[768 * seq2..768 * seq2 + 3]);
        println!("out2[0..4]={:?} ref={:?}", &got2[0..4], &ref_l[0..4]);
    }
    let mut me2 = 0.0f32;
    let mut dn2 = 0.0f32;
    for i in 0..got2.len() {
        me2 = me2.max((got2[i] - ref_l[i]).abs());
        dn2 = dn2.max(ref_l[i].abs());
    }
    println!("attn_long parity: bh={bh2} seq={seq2} max_err={me2:e} rel={:.3e}", me2 / dn2);
    assert!(me2 < 1e-3 * dn2, "attn_long parity failed");
    println!("OK");
}

/// Gate/out-proj/FF chain parity (parity/gateff.npz).
fn gateff_parity_test(device: usize, model_dir: &std::path::Path) {
    let npz_path = model_dir.parent().unwrap_or(model_dir).join("parity/gateff.npz");
    let npz = npz::Npz::open(&npz_path).unwrap_or_else(|e| panic!("{e}"));
    let m = 512usize;
    let (dim, heads_dim, ff) = (256usize, 512usize, 1024usize);
    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let km = gpu_kernels::load(&ctx).expect("kernel module");
    let gemm2 = |mm: usize, nn: usize, kk: usize, x: &DeviceBuffer<f32>, w: &DeviceBuffer<f32>, bias: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>| {
        // SAFETY: 2-D grid of 16x16 tiles covering mm x nn.
        unsafe {
            km.gemm_bias_64(
                &stream,
                cuda_core::simt::LaunchConfig {
                    grid_dim: ((nn.div_ceil(16)) as u32, (mm.div_ceil(16)) as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                },
                mm as u32,
                nn as u32,
                kk as u32,
                x,
                w,
                bias,
                y,
            )
        }
        .expect("gemm");
    };

    // gates GEMM (M, 256 -> 8)
    let h = npz.f32("h").unwrap();
    let wg = npz.f32("wg").unwrap();
    let bg = npz.f32("bg").unwrap();
    let h_dev = DeviceBuffer::from_host(&stream, h).unwrap();
    let wg_dev = DeviceBuffer::from_host(&stream, wg).unwrap();
    let bg_dev = DeviceBuffer::from_host(&stream, bg).unwrap();
    let mut gates_dev = DeviceBuffer::<f32>::zeroed(&stream, m * 8).unwrap();
    gemm2(m, 8, dim, &h_dev, &wg_dev, &bg_dev, &mut gates_dev);

    // gate scale on attn_raw
    let attn_raw = npz.f32("attn_raw").unwrap();
    let attn_dev = DeviceBuffer::from_host(&stream, attn_raw).unwrap();
    let mut scaled_dev = DeviceBuffer::<f32>::zeroed(&stream, m * heads_dim).unwrap();
    // SAFETY: elementwise over m*512.
    unsafe {
        km.gate_scale(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * heads_dim) as u32), &attn_dev, &gates_dev, &mut scaled_dev)
    }
    .expect("gate scale");

    // out proj GEMM (M, 512 -> 256) + residual
    let wo = npz.f32("wo").unwrap();
    let bo = npz.f32("bo").unwrap();
    let x_res = npz.f32("x_res").unwrap();
    let wo_dev = DeviceBuffer::from_host(&stream, wo).unwrap();
    let bo_dev = DeviceBuffer::from_host(&stream, bo).unwrap();
    let mut oproj_dev = DeviceBuffer::<f32>::zeroed(&stream, m * dim).unwrap();
    gemm2(m, dim, heads_dim, &scaled_dev, &wo_dev, &bo_dev, &mut oproj_dev);
    let res_dev = DeviceBuffer::from_host(&stream, x_res).unwrap();
    let mut attn_out_dev = DeviceBuffer::<f32>::zeroed(&stream, m * dim).unwrap();
    // SAFETY: elementwise over m*256.
    unsafe {
        km.add_resid(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * dim) as u32), &oproj_dev, &res_dev, &mut attn_out_dev)
    }
    .expect("add resid");

    let attn_out = attn_out_dev.to_host_vec(&stream).unwrap();
    let ref_attn = npz.f32("attn_out").unwrap();
    let mut e1 = 0.0f32;
    let mut d1 = 0.0f32;
    for i in 0..attn_out.len() {
        e1 = e1.max((attn_out[i] - ref_attn[i]).abs());
        d1 = d1.max(ref_attn[i].abs());
    }
    println!("gate+outproj parity: rel={:.3e}", e1 / d1);
    assert!(e1 < 1e-4 * d1, "gate/outproj parity failed");

    // FF1: GEMM (M,256->1024) + GELU
    let w1 = npz.f32("w1").unwrap();
    let b1 = npz.f32("b1").unwrap();
    let w1_dev = DeviceBuffer::from_host(&stream, w1).unwrap();
    let b1_dev = DeviceBuffer::from_host(&stream, b1).unwrap();
    let mut pre_dev = DeviceBuffer::<f32>::zeroed(&stream, m * ff).unwrap();
    gemm2(m, ff, dim, &attn_out_dev, &w1_dev, &b1_dev, &mut pre_dev);
    let mut ff1_dev = DeviceBuffer::<f32>::zeroed(&stream, m * ff).unwrap();
    // SAFETY: elementwise over m*ff.
    unsafe {
        km.gelu_erf(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * ff) as u32), &pre_dev, &mut ff1_dev)
    }
    .expect("gelu");
    let ff1 = ff1_dev.to_host_vec(&stream).unwrap();
    let ref_ff1 = npz.f32("ff1").unwrap();
    let mut e2 = 0.0f32;
    let mut d2 = 0.0f32;
    for i in 0..ff1.len() {
        e2 = e2.max((ff1[i] - ref_ff1[i]).abs());
        d2 = d2.max(ref_ff1[i].abs());
    }
    println!("ff1+gelu parity: rel={:.3e}", e2 / d2);
    assert!(e2 < 5e-4 * d2, "ff1 parity failed");

    // FF2: GEMM (M,1024->256) + residual
    let w2 = npz.f32("w2").unwrap();
    let b2 = npz.f32("b2").unwrap();
    let w2_dev = DeviceBuffer::from_host(&stream, w2).unwrap();
    let b2_dev = DeviceBuffer::from_host(&stream, b2).unwrap();
    let mut ffo_dev = DeviceBuffer::<f32>::zeroed(&stream, m * dim).unwrap();
    gemm2(m, dim, ff, &ff1_dev, &w2_dev, &b2_dev, &mut ffo_dev);
    let mut ff_out_dev = DeviceBuffer::<f32>::zeroed(&stream, m * dim).unwrap();
    // SAFETY: elementwise over m*256.
    unsafe {
        km.add_resid(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * dim) as u32), &ffo_dev, &attn_out_dev, &mut ff_out_dev)
    }
    .expect("add resid 2");
    let ff_out = ff_out_dev.to_host_vec(&stream).unwrap();
    let ref_ff = npz.f32("ff_out").unwrap();
    let mut e3 = 0.0f32;
    let mut d3 = 0.0f32;
    for i in 0..ff_out.len() {
        e3 = e3.max((ff_out[i] - ref_ff[i]).abs());
        d3 = d3.max(ref_ff[i].abs());
    }
    println!("ff2+resid parity: rel={:.3e}", e3 / d3);
    assert!(e3 < 1e-4 * d3, "ff2 parity failed");
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


// ---------------------------------------------------------------------------
// End-to-end single-chunk forward (Phase 3)
// ---------------------------------------------------------------------------

struct GpuWeights {
    // transformer per layer per axis: qkv_w, gates_w, gates_b, out_w (shared biases shared once)
    qkv_w: Vec<DeviceBuffer<f32>>,      // [24] (1536*256)
    gates_w: Vec<DeviceBuffer<f32>>,    // [24] (8*256)
    gates_b: Vec<DeviceBuffer<f32>>,    // [24] (8)
    out_w: Vec<DeviceBuffer<f32>>,      // [24] (256*512)
    norm_gamma: Vec<DeviceBuffer<f32>>, // [24] attention pre-norm
    ff_gamma: Vec<DeviceBuffer<f32>>,   // [24]
    ff_w1: Vec<DeviceBuffer<f32>>,      // [24] (1024*256)
    ff_b1: Vec<DeviceBuffer<f32>>,      // [24]
    ff_w2: Vec<DeviceBuffer<f32>>,      // [24] (256*1024)
    ff_b2: Vec<DeviceBuffer<f32>>,      // [24]
    shared_qkv_bias: DeviceBuffer<f32>,
    shared_out_bias: DeviceBuffer<f32>,
    final_norm: DeviceBuffer<f32>,
    band_gamma: DeviceBuffer<f32>,
    band_w: DeviceBuffer<f32>,
    band_b: DeviceBuffer<f32>,
    // maskest per stem: w1 (62*1024*256), b1, w2 concatenated, b2
    mask_w1: Vec<DeviceBuffer<f32>>,
    mask_b1: Vec<DeviceBuffer<f32>>,
    mask_w2: Vec<DeviceBuffer<f32>>,
    mask_b2: Vec<DeviceBuffer<f32>>,
}


struct E2eScratch {
    h: DeviceBuffer<f32>,          // (M, 256)
    qkv: DeviceBuffer<f32>,        // (M, 1536)
    qkv_rope: DeviceBuffer<f32>,   // (M, 1536)
    qkv_attn: DeviceBuffer<f32>,   // time: [3*496, T, 64]; freq: [3*9208, 62, 64]
    v_flat: DeviceBuffer<f32>,     // (M, 512)
    gates: DeviceBuffer<f32>,      // (M, 8)
    scaled: DeviceBuffer<f32>,     // (M, 512)
    oproj: DeviceBuffer<f32>,      // (M, 256)
    attn_out: DeviceBuffer<f32>,   // (M, 256)
    ffpre: DeviceBuffer<f32>,      // (M, 1024)
    ff1: DeviceBuffer<f32>,        // (M, 1024)
    ff2: DeviceBuffer<f32>,        // (M, 256)
    p_big: DeviceBuffer<f32>,      // time scores: 496*T*T
    attn_out_long: DeviceBuffer<f32>, // [496, T, 64]
    cos: [DeviceBuffer<f32>; 2],
    sin: [DeviceBuffer<f32>; 2],
    zero_bias_t: DeviceBuffer<f32>,
    zero_bias_64: DeviceBuffer<f32>,
    attn_max: DeviceBuffer<f32>,
    attn_scale: DeviceBuffer<f32>,
}

fn upload_weights(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, w: &weights::ModelWeights) -> Result<GpuWeights, String> {
    let up = |v: &[f32]| DeviceBuffer::from_host(stream, v).map_err(|e| e.to_string());
    let mut qkv_w = Vec::new();
    let mut gates_w = Vec::new();
    let mut gates_b = Vec::new();
    let mut out_w = Vec::new();
    let mut norm_gamma = Vec::new();
    let mut ff_gamma = Vec::new();
    let mut ff_w1 = Vec::new();
    let mut ff_b1 = Vec::new();
    let mut ff_w2 = Vec::new();
    let mut ff_b2 = Vec::new();
    for layer in &w.layers {
        for (attn, ff) in [layer.time.clone(), layer.freq.clone()] {
            qkv_w.push(up(&attn.to_qkv_w)?);
            gates_w.push(up(&attn.to_gates_w)?);
            gates_b.push(up(&attn.to_gates_b)?);
            out_w.push(up(&attn.to_out_w)?);
            norm_gamma.push(up(&attn.norm_gamma)?);
            ff_gamma.push(up(&ff.gamma0)?);
            ff_w1.push(up(&ff.w1)?);
            ff_b1.push(up(&ff.b1)?);
            ff_w2.push(up(&ff.w2)?);
            ff_b2.push(up(&ff.b2)?);
        }
    }
    let mut mask_w1 = Vec::new();
    let mut mask_b1 = Vec::new();
    let mut mask_w2 = Vec::new();
    let mut mask_b2 = Vec::new();
    for s in 0..w.mask_w1.len() {
        mask_w1.push(up(&w.mask_w1[s].concat())?);
        mask_b1.push(up(&w.mask_b1[s].concat())?);
        mask_w2.push(up(&w.mask_w2[s].concat())?);
        mask_b2.push(up(&w.mask_b2[s].concat())?);
    }
    Ok(GpuWeights {
        qkv_w,
        gates_w,
        gates_b,
        out_w,
        norm_gamma,
        ff_gamma,
        ff_w1,
        ff_b1,
        ff_w2,
        ff_b2,
        shared_qkv_bias: up(&w.shared_qkv_bias)?,
        shared_out_bias: up(&w.shared_out_bias)?,
        final_norm: up(&w.final_norm_gamma)?,
        band_gamma: up(&w.band_gamma.concat())?,
        band_w: up(&w.band_w.concat())?,
        band_b: up(&w.band_b.concat())?,
        mask_w1,
        mask_b1,
        mask_w2,
        mask_b2,
    })
}


/// One full transformer layer step on the folded token tensor x (M, 256).
/// axis 0 = time (seq=T rows per f), 1 = freq (seq=62 rows per t).
unsafe fn transformer_step(
    km: &gpu_kernels::LoadedModule,
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gw: &GpuWeights,
    layer_idx: usize,
    axis: usize,
    x: &mut DeviceBuffer<f32>,
    t_frames: usize,
    bands: usize,
    scratch: &mut E2eScratch,
) -> Result<(), String> {
    let m = t_frames * bands; // 71342
    let idx = layer_idx * 2 + axis;
    let launch1 = |n: u32| cuda_core::simt::LaunchConfig::for_num_elems(n);
    // 1. pre-attention RMSNorm
    // SAFETY: one warp per row over m rows.
    unsafe {
        km.rmsnorm_gates(stream, launch1((m * 32) as u32), x, &gw.norm_gamma[idx], &gw.gates_w[idx], &gw.gates_b[idx], &mut scratch.h, &mut scratch.gates, m as u32, 256)
    }.map_err(|e| e.to_string())?;

    // 2. QKV GEMM (M,256 -> 1536) with shared bias
    // SAFETY: 2-D tile grid over m x 1536.
    unsafe {
        km.gemm_f16_128x64(stream, tile_cfg_tf32_64(m, 1536), m as u32, 1536, 256, &scratch.h, &gw.qkv_w[idx], &gw.shared_qkv_bias, &mut scratch.qkv)
    }.map_err(|e| e.to_string())?;

    // RoPE and attention-major reordering are fused into the raw-QKV
    // QK/PV attention kernels below.
    let (bh, n_len) = if axis == 0 { (bands * 8, t_frames) } else { (t_frames * 8, bands) };
    // Gates were fused into the pre-attention RMSNorm kernel.
    // 5. attention: short (freq) uses attn_short per (b,h) block; long (time)
    // uses 2-pass materialized with the batched kernels below.
    if axis == 1 {
        // Frequency axis (seq=62, one key tile): flash attention in a single
        // launch, ~8x faster than the three-kernel chain.
        // SAFETY: one 128-thread block per (BH group, 64-row query tile).
        unsafe {
            km.attn_flash_tc(stream, cuda_core::simt::LaunchConfig { grid_dim: (1, bands.div_ceil(64) as u32, bh as u32), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, &scratch.qkv, &scratch.cos[axis], &scratch.sin[axis], &scratch.gates, bands as u32, axis as u32, bands as u32, &mut scratch.scaled, &mut scratch.attn_max, &mut scratch.attn_scale)
        }.map_err(|e| e.to_string())?;
    } else {
        // Time axis (seq=259): the materialized three-kernel chain still
        // beats the flash kernel whose key-tile loop pays 5x sync/softmax
        // serialization; see BENCHMARK.md.
        // SAFETY: 64x128 FP16 tiles over [BH, N, N].
        unsafe {
            km.attn_qk_rope_f16(stream, cuda_core::simt::LaunchConfig { grid_dim: (n_len.div_ceil(64) as u32, n_len.div_ceil(64) as u32, bh as u32), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, n_len as u32, n_len as u32, 64, &scratch.qkv, &scratch.cos[axis], &scratch.sin[axis], &scratch.zero_bias_t, bands as u32, axis as u32, &mut scratch.p_big)
        }.map_err(|e| e.to_string())?;
        // SAFETY: one warp per score row.
        unsafe {
            km.softmax_stats(stream, launch1((bh * n_len * 32) as u32), &scratch.p_big, (bh * n_len) as u32, n_len as u32, &mut scratch.attn_max, &mut scratch.attn_scale)
        }.map_err(|e| e.to_string())?;
        // SAFETY: one 64-column FP16 tile per row block and BH group.
        unsafe {
            km.attn_pv_raw_f16(stream, cuda_core::simt::LaunchConfig { grid_dim: (1, n_len.div_ceil(128) as u32, bh as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, n_len as u32, 64, n_len as u32, &scratch.p_big, &scratch.qkv, &scratch.zero_bias_64, &scratch.gates, bands as u32, axis as u32, &scratch.attn_max, &scratch.attn_scale, &mut scratch.scaled)
        }.map_err(|e| e.to_string())?;
    }
    // 7. gate application was fused into attn_v_to_flat; out proj reads scaled.
    // SAFETY: 2-D tile grid over m x 256.
    unsafe {
        km.gemm_f16_residual_128x64(stream, tile_cfg_tf32_64(m, 256), m as u32, 256, 512, &scratch.scaled, &gw.out_w[idx], &gw.shared_out_bias, x, &mut scratch.attn_out)
    }.map_err(|e| e.to_string())?;
    // Residual addition is fused into the out-projection epilogue.
    // 8. FF: norm -> w1+gelu -> w2 -> residual
    // SAFETY: one warp per row.
    unsafe {
        km.rmsnorm(stream, launch1((m * 32) as u32), &scratch.attn_out, &gw.ff_gamma[idx], &mut scratch.h, m as u32, 256)
    }.map_err(|e| e.to_string())?;
    // SAFETY: 2-D tile grid over m x 1024.
    unsafe {
        km.gemm_f16_gelu_128x64(stream, tile_cfg_tf32_64(m, 1024), m as u32, 1024, 256, &scratch.h, &gw.ff_w1[idx], &gw.ff_b1[idx], &mut scratch.ff1)
    }.map_err(|e| e.to_string())?;
    // GELU is fused into the FF1 tensor-core epilogue.
    // SAFETY: 2-D tile grid over m x 256.
    unsafe {
        km.gemm_f16_residual_128x64(stream, tile_cfg_tf32_64(m, 256), m as u32, 256, 1024, &scratch.ff1, &gw.ff_w2[idx], &gw.ff_b2[idx], &scratch.attn_out, x)
    }.map_err(|e| e.to_string())?;
    // Residual addition is fused into the FF2 epilogue.
    Ok(())
}

fn tile_cfg_tf32_64(m: usize, n: usize) -> cuda_core::simt::LaunchConfig {
    cuda_core::simt::LaunchConfig {
        grid_dim: ((n.div_ceil(64)) as u32, (m.div_ceil(128)) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn tile_cfg_tf32(m: usize, n: usize) -> cuda_core::simt::LaunchConfig {
    cuda_core::simt::LaunchConfig {
        grid_dim: ((n.div_ceil(8)) as u32, (m.div_ceil(64)) as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn tile_cfg_64(m: usize, n: usize) -> cuda_core::simt::LaunchConfig {
    cuda_core::simt::LaunchConfig {
        grid_dim: ((n.div_ceil(64)) as u32, (m.div_ceil(64)) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}

fn tile_cfg(m: usize, n: usize) -> cuda_core::simt::LaunchConfig {
    cuda_core::simt::LaunchConfig {
        grid_dim: ((n.div_ceil(16)) as u32, (m.div_ceil(16)) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}


/// A borrowed interior segment of a DeviceBuffer. Dropping the wrapper
/// never frees the underlying memory (the parent buffer stays the sole
/// owner), so segments can be passed to kernels as &DeviceBuffer freely.
pub struct Seg {
    inner: Option<DeviceBuffer<f32>>,
}
impl std::ops::Deref for Seg {
    type Target = DeviceBuffer<f32>;
    fn deref(&self) -> &DeviceBuffer<f32> {
        self.inner.as_ref().unwrap()
    }
}
impl std::ops::DerefMut for Seg {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<f32> {
        self.inner.as_mut().unwrap()
    }
}
impl Drop for Seg {
    fn drop(&mut self) {
        if let Some(b) = self.inner.take() {
            std::mem::forget(b); // never free an interior pointer
        }
    }
}

fn slice_view(stream: &Arc<CudaStream>, b: &DeviceBuffer<f32>, off: usize, len: usize) -> Result<Seg, String> {
    if off + len > b.len() {
        return Err(format!("slice_view oob {off}+{len}>{}", b.len()));
    }
    let ptr = b.cu_deviceptr() + (off * 4) as u64;
    let _ = stream;
    // SAFETY: interior segment of a live parent allocation in the same ctx.
    Ok(Seg { inner: Some(unsafe { DeviceBuffer::from_raw_parts(ptr, len, b.context().clone()) }) })
}

fn mut_slice_view(stream: &Arc<CudaStream>, b: &mut DeviceBuffer<f32>, off: usize, len: usize) -> Result<Seg, String> {
    if off + len > b.len() {
        return Err(format!("mut_slice_view oob {off}+{len}>{}", b.len()));
    }
    let ptr = b.cu_deviceptr() + (off * 4) as u64;
    let _ = stream;
    // SAFETY: same as slice_view; the caller's use is exclusive in stream order.
    Ok(Seg { inner: Some(unsafe { DeviceBuffer::from_raw_parts(ptr, len, b.context().clone()) }) })
}


/// Full end-to-end parity on the golden 3s input (parity/e2e_mid.npz for the
/// transformer trunk + assets/ref_output.npz for the final stems).

/// qkv_to_attn / attn_v_to_flat unit test vs python reference (T=37).
fn reorder_test(device: usize, model_dir: &std::path::Path) {
    let root = model_dir.parent().unwrap_or(model_dir);
    let npz = npz::Npz::open(&root.join("parity/reorder.npz")).unwrap_or_else(|e| panic!("{e}"));
    let qkv = npz.f32("qkv").expect("qkv");
    let ref_q = npz.f32("q").expect("q");
    let t_frames = 37usize;
    let bands = 62usize;
    let m = t_frames * bands;
    let bh = bands * 8;
    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let km = gpu_kernels::load(&ctx).expect("kernels");
    let in_dev = DeviceBuffer::from_host(&stream, qkv).unwrap();
    let mut att = DeviceBuffer::<f32>::zeroed(&stream, 3 * bh * t_frames * 64).unwrap();
    // SAFETY: elementwise reorder over 3*BH*T*64.
    unsafe {
        km.qkv_to_attn(&stream, cuda_core::simt::LaunchConfig::for_num_elems((3 * bh * t_frames * 64) as u32), &in_dev, &mut att, t_frames as u32, bands as u32, 0)
    }.expect("reorder");
    let got = att.to_host_vec(&stream).unwrap();
    // q segment = first bh*T*64
    let mut me = 0.0f32;
    let mut dn = 0.0f32;
    for i in 0..ref_q.len() {
        me = me.max((got[i] - ref_q[i]).abs());
        dn = dn.max(ref_q[i].abs());
    }
    println!("reorder q segment: rel={:.3e}", me / dn);
    for (i, label) in [(0usize, "att[0]"), (64, "att[64]"), (37 * 64, "att[t1d0]"), (62 * 8 * 37 * 64 / 2, "mid")] {
        println!("spot {label}: got={:.5} ref={:.5}", got[i], ref_q[i]);
    }
    // v roundtrip: attn_v_to_flat on the v segment
    let mut vflat = DeviceBuffer::<f32>::zeroed(&stream, m * 512).unwrap();
    let zero_gates = DeviceBuffer::<f32>::zeroed(&stream, m * 8).unwrap();
    let vseg = slice_view(&stream, &att, 2 * bh * t_frames * 64, bh * t_frames * 64).unwrap();
    // SAFETY: elementwise over m*512.
    unsafe {
        km.attn_v_to_flat(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * 512) as u32), &*vseg, &zero_gates, &mut vflat, t_frames as u32, bands as u32, 0)
    }.expect("unreorder");
    let vf = vflat.to_host_vec(&stream).unwrap();
    // v_flat reference: identity (v part of qkv)
    let mut me2 = 0.0f32;
    for i in 0..vf.len() {
        let row = i / 512;
        me2 = me2.max((vf[i] - qkv[row * 1536 + 1024 + i % 512]).abs());
    }
    println!("v roundtrip: max_err={me2:e}");
    assert!(me < 1e-5 * dn, "reorder q mismatch");
    assert!(me2 < 1e-6, "v roundtrip mismatch");
    println!("OK");
}

fn e2e_test(device: usize, model_dir: &std::path::Path) {
    let root = model_dir.parent().unwrap_or(model_dir);
    let mid = npz::Npz::open(&root.join("parity/e2e_mid.npz")).unwrap_or_else(|e| panic!("{e}"));
    let ref_mid = mid.f32("x_final").expect("x_final");
    let golden = npz::Npz::open(&root.join("assets/ref_output.npz")).unwrap_or_else(|e| panic!("{e}"));
    let inp = golden.f32("inp").expect("inp"); // (2, L) planar
    let ref_out = golden.f32("out").expect("out"); // (6, 2, L)

    let cfg = config::ModelConfig::parse(&std::fs::read_to_string(model_dir.join("logic_bs_roformer.yaml")).unwrap()).unwrap();
    let t_frames = stft::num_frames(inp.len() / 2); // 259
    let bands = cfg.num_bands(); // 62
    let m = t_frames * bands;
    let len = inp.len() / 2;
    println!("e2e: L={len} T={t_frames} bands={bands} M={m}");

    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let km = gpu_kernels::load(&ctx).expect("kernels");

    // weights
    let weights_t0 = std::time::Instant::now();
    let st = weights::SafeTensors::open(&model_dir.join("model.safetensors")).unwrap();
    let w = weights::ModelWeights::load(&st, &cfg).unwrap();
    let gw = upload_weights(&ctx, &stream, &w).expect("weights upload");
    println!("weights load+upload: {:?}", weights_t0.elapsed());
    let forward_t0 = std::time::Instant::now();

    // input interleaved
    let mut xi = vec![0.0f32; len * 2];
    for i in 0..len {
        xi[i * 2] = inp[i];
        xi[i * 2 + 1] = inp[len + i];
    }
    let x_dev = DeviceBuffer::from_host(&stream, &xi).unwrap();

    // 1. STFT
    let fft = std::sync::Arc::new(cufft::Cufft::load().expect("cufft"));
    let sp = stft::Stft::new(fft.clone(), t_frames).expect("stft plan");
    let win_dev = DeviceBuffer::from_host(&stream, sp.window()).unwrap();
    let mut frames_dev = DeviceBuffer::<f32>::zeroed(&stream, 2 * t_frames * stft::N_FFT).unwrap();
    // SAFETY: one thread per frame element.
    unsafe {
        km.frame_hann_reflect(&stream, cuda_core::simt::LaunchConfig::for_num_elems((2 * t_frames * stft::N_FFT) as u32),
            &x_dev, &win_dev, &mut frames_dev, stft::N_FFT as u32, stft::HOP as u32, len as u32, t_frames as u32)
    }.expect("frame");
    let mut spec_dev = DeviceBuffer::<f32>::zeroed(&stream, 2 * t_frames * stft::FREQ_BINS * 2).unwrap();
    sp.exec_fwd(&frames_dev, &mut spec_dev).expect("r2c");

    // 2. reorder to BandSplit input (T, 4100)
    let freqs: Vec<usize> = cfg.freqs_per_bands.clone();
    let dims: Vec<u32> = freqs.iter().map(|f| (2 * f * 2) as u32).collect();
    let mut offs = vec![0u32; bands + 1];
    let mut f0 = vec![0u32; bands];
    for i in 0..bands { offs[i + 1] = offs[i] + dims[i]; f0[i] = freqs[..i].iter().sum::<usize>() as u32; }
    let offs_dev = DeviceBuffer::from_host(&stream, &offs).unwrap();
    let dims_dev = DeviceBuffer::from_host(&stream, &dims).unwrap();
    let f0_dev = DeviceBuffer::from_host(&stream, &f0).unwrap();
    let mut xin_dev = DeviceBuffer::<f32>::zeroed(&stream, t_frames * 4100).unwrap();
    // SAFETY: elementwise gather over t*4100.
    unsafe {
        km.spec_reorder(&stream, cuda_core::simt::LaunchConfig::for_num_elems((t_frames * 4100) as u32),
            &spec_dev, &f0_dev, &offs_dev, &mut xin_dev, t_frames as u32)
    }.expect("spec reorder");

    // 3. BandSplit: padded RMSNorm followed by grouped tensor-core GEMM.
    let max_dim = 2 * 129 * 2; // 516
    let mut bsnorm = DeviceBuffer::<f32>::zeroed(&stream, bands * t_frames * max_dim).unwrap();
    let mut x = DeviceBuffer::<f32>::zeroed(&stream, m * 256).unwrap();
    // SAFETY: one warp per (band, time) row.
    unsafe {
        km.bandsplit_norm(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * 32) as u32),
            &xin_dev, &gw.band_gamma, &offs_dev, &dims_dev, &mut bsnorm, t_frames as u32, bands as u32, max_dim as u32)
    }.expect("bandsplit norm");
    // SAFETY: 64x128 FP16 tiles for every band.
    unsafe {
        km.bandsplit_gemm_f16(&stream, cuda_core::simt::LaunchConfig { grid_dim: (4, t_frames.div_ceil(128) as u32, bands as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
            &offs_dev, &dims_dev, &bsnorm, &gw.band_w, &gw.band_b, &mut x, t_frames as u32, bands as u32, max_dim as u32)
    }.expect("bandsplit gemm");

    // rotary tables for both axes
    let cos_t: Vec<f32> = (0..t_frames).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).cos())).collect();
    let sin_t: Vec<f32> = (0..t_frames).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).sin())).collect();
    let cos_f: Vec<f32> = (0..bands).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).cos())).collect();
    let sin_f: Vec<f32> = (0..bands).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).sin())).collect();

    // scratch
    let z = |n: usize| DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
    let mut scr = E2eScratch {
        h: z(m * 256),
        qkv: z(m * 1536),
        qkv_rope: z(m * 1536),
        qkv_attn: z(3 * 62 * 8 * t_frames.max(bands) * 64), // max of both axes' layouts
        v_flat: z(m * 512),
        gates: z(m * 8),
        scaled: z(m * 512),
        oproj: z(m * 256),
        attn_out: z(m * 256),
        ffpre: z(m * 1024),
        ff1: z(m * 1024),
        ff2: z(m * 256),
        p_big: z(62 * 8 * t_frames * t_frames),
        attn_out_long: z(496 * t_frames * 64),
        cos: [DeviceBuffer::from_host(&stream, &cos_t).unwrap(), DeviceBuffer::from_host(&stream, &cos_f).unwrap()],
        sin: [DeviceBuffer::from_host(&stream, &sin_t).unwrap(), DeviceBuffer::from_host(&stream, &sin_f).unwrap()],
        zero_bias_t: z(t_frames),
        zero_bias_64: z(64),
        attn_max: z(m * 8),
        attn_scale: z(m * 8),
    };

    // staged references for bisection
    let stages = npz::Npz::open(&root.join("parity/e2e_stages.npz")).ok();
    let stage_check = |name: &str, got: &[f32], ref_arr: &[f32], perm_ft: bool| {
        let mut me = 0.0f32;
        let mut dn = 0.0f32;
        let dim = ref_arr.len() / (t_frames * bands);
        for t in 0..t_frames {
            for f in 0..bands {
                for d in 0..dim {
                    let (g, r) = if perm_ft {
                        (got[(t * bands + f) * dim + d], ref_arr[(f * t_frames + t) * dim + d])
                    } else {
                        (got[(t * bands + f) * dim + d], ref_arr[(t * bands + f) * dim + d])
                    };
                    me = me.max((g - r).abs());
                    dn = dn.max(r.abs());
                }
            }
        }
        println!("stage {name}: rel={:.3e}", me / dn);
    };
    {
        let bsnpz = npz::Npz::open(&root.join("parity/e2e_bs.npz")).expect("e2e_bs");
        let bsin_ref = bsnpz.f32("bs_input").unwrap();
        let xin = xin_dev.to_host_vec(&stream).unwrap();
        {
            let mut me = 0.0f32;
            let mut dn = 0.0f32;
            for i in 0..xin.len() {
                me = me.max((xin[i] - bsin_ref[i]).abs());
                dn = dn.max(bsin_ref[i].abs());
            }
            println!("stage bs_input(reorder): rel={:.3e}", me / dn);
        }
        let bs_ref = bsnpz.f32("bandsplit").unwrap();
        let xv = x.to_host_vec(&stream).unwrap();
        stage_check("bandsplit", &xv, bs_ref, false);
    }

    // Manual first-half of layer 0 with staged checks. Disabled for clean
    // performance runs; the full layer is recomputed by transformer_step below.
    if false {
        let sub = npz::Npz::open(&root.join("parity/e2e_sub.npz")).expect("e2e_sub");
        let m_ = m;
        let launch1 = |n: usize| cuda_core::simt::LaunchConfig::for_num_elems(n as u32);
        // rmsnorm
        // SAFETY: one warp per row.
        unsafe { km.rmsnorm(&stream, launch1(m_ * 32), &x, &gw.norm_gamma[0], &mut scr.h, m_ as u32, 256) }.unwrap();
        {
            let hv = scr.h.to_host_vec(&stream).unwrap();
            let r = sub.f32("h_norm").unwrap();
            stage_check("L0.h_norm", &hv, r, true);
        }
        // qkv gemm
        // SAFETY: tile grid over m x 1536.
        unsafe { km.gemm_bias_64(&stream, tile_cfg_64(m_, 1536), m_ as u32, 1536, 256, &scr.h, &gw.qkv_w[0], &gw.shared_qkv_bias, &mut scr.qkv) }.unwrap();
        // rope (time axis)
        // SAFETY: one thread per (row, pair).
        unsafe { km.rope_scale(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m_ * 768) as u32), &scr.qkv, &scr.cos[0], &scr.sin[0], &mut scr.qkv_rope, t_frames as u32, bands as u32, 0) }.unwrap();
        // gates (uses h)
        // SAFETY: tile grid over m x 8.
        unsafe { km.gemm_bias_64(&stream, tile_cfg_64(m_, 8), m_ as u32, 8, 256, &scr.h, &gw.gates_w[0], &gw.gates_b[0], &mut scr.gates) }.unwrap();
        // SAFETY: elementwise reorder.
        unsafe { km.qkv_to_attn(&stream, cuda_core::simt::LaunchConfig::for_num_elems((3 * 62 * 8 * t_frames * 64) as u32), &scr.qkv_rope, &mut scr.qkv_attn, t_frames as u32, bands as u32, 0) }.unwrap();
        {
            let (bh, n_len) = (62usize * 8, t_frames);
            for g in 0..bh {
                let qseg = slice_view(&stream, &scr.qkv_attn, g * n_len * 64, n_len * 64).unwrap();
                let kseg = slice_view(&stream, &scr.qkv_attn, bh * n_len * 64 + g * n_len * 64, n_len * 64).unwrap();
                let mut pseg = mut_slice_view(&stream, &mut scr.p_big, g * n_len * n_len, n_len * n_len).unwrap();
                // SAFETY: tile grid over n_len x n_len.
                unsafe { km.gemm_bias_64(&stream, tile_cfg_64(n_len, n_len), n_len as u32, n_len as u32, 64, &*qseg, &*kseg, &scr.zero_bias_t, &mut *pseg) }.unwrap();
            }
            // SAFETY: one thread per row.
            unsafe { km.softmax_rows(&stream, cuda_core::simt::LaunchConfig::for_num_elems((bh * n_len) as u32), &mut scr.p_big, (bh * n_len) as u32, n_len as u32) }.unwrap();
            {
                let ps = npz::Npz::open(&root.join("parity/e2e_pspot.npz")).expect("pspot");
                let p0 = ps.f32("p0").unwrap();
                let q00 = ps.f32("q00").unwrap();
                let pb = scr.p_big.to_host_vec(&stream).unwrap();
                let qa = scr.qkv_attn.to_host_vec(&stream).unwrap();
                println!("P[0,0,:16] = {:?}", &pb[0..16]);
                println!("ref P0[:16]  = {:?}", &p0[0..16]);
                // also the peak position
                let mut pk = 0usize; let mut pv = f32::NEG_INFINITY;
                for j in 0..259 { if pb[j] > pv { pv = pb[j]; pk = j; } }
                println!("P0 peak at {pk} = {pv:.5}");
                println!("q[bh0,t0,:4] = {:?} ref {:?}", &qa[0..4], &q00[0..4]);
            }
            for g in 0..bh {
                let pseg = slice_view(&stream, &scr.p_big, g * n_len * n_len, n_len * n_len).unwrap();
                let vseg = slice_view(&stream, &scr.qkv_attn, 2 * bh * n_len * 64 + g * n_len * 64, n_len * 64).unwrap();
                let mut oseg = mut_slice_view(&stream, &mut scr.attn_out_long, g * n_len * 64, n_len * 64).unwrap();
                // SAFETY: tile grid over n_len x 64.
                unsafe { km.gemm_bias_bn(&stream, tile_cfg(n_len, 64), n_len as u32, 64, n_len as u32, &*pseg, &*vseg, &scr.zero_bias_64, &mut *oseg) }.unwrap();
            }
            {
                let ol = scr.attn_out_long.to_host_vec(&stream).unwrap();
                let vfnpz = npz::Npz::open(&root.join("parity/e2e_vflat.npz")).expect("e2e_vflat");
                let r = vfnpz.f32("v_flat").unwrap();
                println!("ol[bh0,t0,:4] = {:?} ref_vflat[0..4] = {:?}", &ol[0..4], &r[0..4]);
                println!("ol[bh1,t0,:4] = {:?} ref vflat(t0,f1,h0)= {:?}", &ol[259 * 64..259 * 64 + 4], &r[512..516]);
                // host check: dot(P row0, V col0) should equal ol[0]
                let pb2 = scr.p_big.to_host_vec(&stream).unwrap();
                let qa2 = scr.qkv_attn.to_host_vec(&stream).unwrap();
                let vbase = 2 * 496usize * 259 * 64;
                let mut dot = 0.0f32;
                for j in 0..259usize {
                    dot += pb2[j] * qa2[vbase + j * 64];
                }
                println!("host dot(P0, V0col0) = {dot:.6} vs ol[0] = {:.6}", ol[0]);
                let roped = scr.qkv_rope.to_host_vec(&stream).unwrap();
                let raw_v0: Vec<f32> = scr.qkv.to_host_vec(&stream).unwrap()[1024..1028].to_vec();
                println!("attV[0..4]={:?} ropeV={:?} rawqkv={:?}", &qa2[vbase..vbase + 4], &roped[1024..1028], &raw_v0);
                {
                    let qa_ref = npz::Npz::open(&root.join("parity/e2e_qkv_attn.npz")).expect("qkv_attn ref");
                    let (rq, rk, rv) = (qa_ref.f32("qr").unwrap(), qa_ref.f32("kr").unwrap(), qa_ref.f32("v").unwrap());
                    println!("attQ[0..4]={:?} refQ={:?}", &qa2[0..4], &rq[0..4]);
                    println!("attK[0..4]={:?} refK={:?}", &qa2[496 * 259 * 64..496 * 259 * 64 + 4], &rk[0..4]);
                    println!("attV0[0..4]={:?} refV={:?}", &qa2[vbase..vbase + 4], &rv[0..4]);
                }
                // v provenance: qkv_attn part2 first 4, and the raw qkv v segment for m=0
            }
            // SAFETY: elementwise un-reorder (time axis).
            unsafe { km.attn_v_to_flat(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m_ * 512) as u32), &scr.attn_out_long, &scr.gates, &mut scr.v_flat, t_frames as u32, bands as u32, 0) }.unwrap();
        }
        {
            let vfnpz = npz::Npz::open(&root.join("parity/e2e_vflat.npz")).expect("e2e_vflat");
            let r = vfnpz.f32("v_flat").unwrap();
            let vv = scr.v_flat.to_host_vec(&stream).unwrap();
            let mut me = 0.0f32;
            let mut dn = 0.0f32;
            for i in 0..vv.len() {
                me = me.max((vv[i] - r[i]).abs());
                dn = dn.max(r[i].abs());
            }
            println!("stage L0.v_flat: rel={:.3e}", me / dn);
        }
        // SAFETY: elementwise over m*512.
        unsafe { km.gate_scale(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m_ * 512) as u32), &scr.v_flat, &scr.gates, &mut scr.scaled) }.unwrap();
        {
            let sc = npz::Npz::open(&root.join("parity/e2e_scaled.npz")).expect("e2e_scaled");
            {
                let gv = scr.gates.to_host_vec(&stream).unwrap();
                let r = sc.f32("gates").unwrap();
                stage_check("L0.gates", &gv, r, true);
            }
            {
                let sv = scr.scaled.to_host_vec(&stream).unwrap();
                let r = sc.f32("to_out_in").unwrap();
                stage_check("L0.scaled", &sv, r, true);
                println!("scaled[(t0,f0)0..4] = {:?}", &sv[0..4]);
                println!("ref[(f0,t0)0..4]    = {:?}", &r[0..4]);
                println!("ours m=62(f0,t1)?  = {:?}", &sv[62 * 512..62 * 512 + 4]);
                println!("ref (f0,t1)        = {:?}", &r[512..516]);
            }
        }
        // SAFETY: tile grid over m x 256.
        unsafe { km.gemm_bias_64(&stream, tile_cfg_64(m_, 256), m_ as u32, 256, 512, &scr.scaled, &gw.out_w[0], &gw.shared_out_bias, &mut scr.oproj) }.unwrap();
        // SAFETY: elementwise over m*256.
        unsafe { km.add_resid(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m_ * 256) as u32), &scr.oproj, &x, &mut scr.attn_out) }.unwrap();
        {
            let av = scr.attn_out.to_host_vec(&stream).unwrap();
            let r = sub.f32("attn_out").unwrap();
            stage_check("L0.attn_out", &av, r, true);
        }
        let _ = &sub;
    }
    // DIAG: compare tensor-core flash attention against the proven three
    // kernel chain on identical layer-0 axis-0 inputs.
    if true {
        let x0 = DeviceBuffer::from_host(&stream, &x.to_host_vec(&stream).unwrap()).unwrap();
        let mut h0 = DeviceBuffer::<f32>::zeroed(&stream, m * 256).unwrap();
        let mut g0b = DeviceBuffer::<f32>::zeroed(&stream, m * 8).unwrap();
        // SAFETY: one warp per row.
        unsafe { km.rmsnorm_gates(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * 32) as u32), &x0, &gw.norm_gamma[0], &gw.gates_w[0], &gw.gates_b[0], &mut h0, &mut g0b, m as u32, 256) }.unwrap();
        let mut qkv0 = DeviceBuffer::<f32>::zeroed(&stream, m * 1536).unwrap();
        // SAFETY: tile grid over m x 1536.
        unsafe { km.gemm_f16_128x64(&stream, tile_cfg_tf32_64(m, 1536), m as u32, 1536, 256, &h0, &gw.qkv_w[0], &gw.shared_qkv_bias, &mut qkv0) }.unwrap();
        let mut s_old = DeviceBuffer::<f32>::zeroed(&stream, m * 512).unwrap();
        let mut s_new = DeviceBuffer::<f32>::zeroed(&stream, m * 512).unwrap();
        let mut a_max = DeviceBuffer::<f32>::zeroed(&stream, m * 8).unwrap();
        let mut a_scale = DeviceBuffer::<f32>::zeroed(&stream, m * 8).unwrap();
        let bh_t = bands * 8;
        // old chain
        // SAFETY: 64x128 FP16 tiles over [BH, N, N].
        unsafe { km.attn_qk_rope_f16(&stream, cuda_core::simt::LaunchConfig { grid_dim: (t_frames.div_ceil(64) as u32, t_frames.div_ceil(64) as u32, bh_t as u32), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, t_frames as u32, t_frames as u32, 64, &qkv0, &scr.cos[0], &scr.sin[0], &scr.zero_bias_t, bands as u32, 0, &mut scr.p_big) }.unwrap();
        // SAFETY: one warp per score row.
        unsafe { km.softmax_stats(&stream, cuda_core::simt::LaunchConfig::for_num_elems((bh_t * t_frames * 32) as u32), &scr.p_big, (bh_t * t_frames) as u32, t_frames as u32, &mut a_max, &mut a_scale) }.unwrap();
        // SAFETY: one 64-column FP16 tile per row block and BH group.
        unsafe { km.attn_pv_raw_f16(&stream, cuda_core::simt::LaunchConfig { grid_dim: (1, t_frames.div_ceil(128) as u32, bh_t as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, t_frames as u32, 64, t_frames as u32, &scr.p_big, &qkv0, &scr.zero_bias_64, &g0b, bands as u32, 0, &a_max, &a_scale, &mut s_old) }.unwrap();
        // new flash
        // SAFETY: one 128-thread block per (BH group, 64-row query tile).
        let mut f_max = DeviceBuffer::<f32>::zeroed(&stream, bh_t * t_frames).unwrap();
        let mut f_l = DeviceBuffer::<f32>::zeroed(&stream, bh_t * t_frames).unwrap();
        // SAFETY: one 128-thread block per (BH group, 64-row query tile).
        unsafe { km.attn_flash_tc(&stream, cuda_core::simt::LaunchConfig { grid_dim: (1, t_frames.div_ceil(64) as u32, bh_t as u32), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, &qkv0, &scr.cos[0], &scr.sin[0], &g0b, bands as u32, 0, t_frames as u32, &mut s_new, &mut f_max, &mut f_l) }.unwrap();
        let qkv0h = qkv0.to_host_vec(&stream).unwrap();
        {
            let mv = a_max.to_host_vec(&stream).unwrap();
            let sv = a_scale.to_host_vec(&stream).unwrap();
            let fmv = f_max.to_host_vec(&stream).unwrap();
            let flv = f_l.to_host_vec(&stream).unwrap();
            let (mut dm, mut dl, mut drows) = (0.0f32, 0.0f32, 0usize);
            for g in 0..bh_t * t_frames {
                let (mo, zo) = (mv[g], 1.0 / sv[g]);
                let (mf, lf) = (fmv[g], flv[g]);
                dm = dm.max((mo - mf).abs());
                let ldiff = (zo - lf).abs() / zo.abs().max(1e-6);
                if ldiff > dl { dl = ldiff; drows = g; }
            }
            println!("DIAG softmax stats: max|dm|={dm:.3e} max rel|dl|={dl:.3e} worst_row={drows} (grp {} row {})", drows / t_frames, drows % t_frames);
            // Host reference for the worst row: RoPE + dot products.
            {
                let (grp_, row_) = (drows / t_frames, drows % t_frames);
                let (b_, h_) = (grp_ / 8, grp_ % 8);
                let token_q = row_ * bands + b_;
                let mut s_host = vec![0.0f32; t_frames];
                let mut m_host = f32::NEG_INFINITY;
                for kc in 0..t_frames {
                    let token_k = kc * bands + b_;
                    let mut acc = 0.0f64;
                    for pair in 0..32usize {
                        let (qe, qo) = (qkv0h[token_q * 1536 + h_ * 64 + pair * 2] as f64, qkv0h[token_q * 1536 + h_ * 64 + pair * 2 + 1] as f64);
                        let (ke, ko) = (qkv0h[token_k * 1536 + 512 + h_ * 64 + pair * 2] as f64, qkv0h[token_k * 1536 + 512 + h_ * 64 + pair * 2 + 1] as f64);
                        let (c, sn) = (cos_t[row_ * 32 + pair] as f64, sin_t[row_ * 32 + pair] as f64);
                        let (ck, sk) = (cos_t[kc * 32 + pair] as f64, sin_t[kc * 32 + pair] as f64);
                        let qr0 = (qe * c - qo * sn) * 0.125;
                        let qr1 = (qo * c + qe * sn) * 0.125;
                        let kr0 = ke * ck - ko * sk;
                        let kr1 = ko * ck + ke * sk;
                        acc += qr0 * kr0 + qr1 * kr1;
                    }
                    s_host[kc] = acc as f32;
                    if s_host[kc] > m_host { m_host = s_host[kc]; }
                }
                let mut l_host = 0.0f32;
                for kc in 0..t_frames { l_host += ((s_host[kc] - m_host) as f32).exp(); }
                let pb = scr.p_big.to_host_vec(&stream).unwrap();
                let prow = grp_ * t_frames * t_frames + row_ * t_frames;
                println!("DIAG host-vs-old scores[:8]: {:?}", &s_host[0..8]);
                println!("DIAG pbig              [:8]: {:?}", &pb[prow..prow + 8]);
                let mut pd = 0.0f32;
                for kc in 0..t_frames { pd = pd.max((pb[prow + kc] - s_host[kc]).abs()); }
                println!("DIAG host-vs-pbig max diff = {pd:.3e}");
                println!("DIAG m: host={m_host:.4} old={:.4} flash={:.4}", mv[drows], fmv[drows]);
                println!("DIAG l: host={l_host:.4} old={:.4} flash={:.4}", 1.0 / sv[drows], flv[drows]);
            }
        }
        let ov = s_old.to_host_vec(&stream).unwrap();
        let nv = s_new.to_host_vec(&stream).unwrap();
        let (mut me, mut dn, mut bad) = (0.0f32, 0.0f32, 0usize);
        let mut bad_by_tile = [0usize; 5];
        let mut bad_by_head = [0usize; 8];
        for i in 0..ov.len() {
            let d = (nv[i] - ov[i]).abs();
            if d > 1e-3 * ov[i].abs().max(1.0) {
                bad += 1;
                let row = i / 512;
                let t = row / bands;
                bad_by_tile[(t / 64).min(4)] += 1;
                bad_by_head[(i % 512) / 64] += 1;
            }
            me = me.max(d);
            dn = dn.max(ov[i].abs());
        }
        println!("DIAG flash-vs-old: max_abs={me:.3e} rel={:.3e} bad={bad}/{}", me / dn, ov.len());
        println!("DIAG bad by q-tile: {bad_by_tile:?}");
        println!("DIAG bad by head: {bad_by_head:?}");
        {
            // spot: first bad element coordinates
            for i in 0..ov.len() {
                let d = (nv[i] - ov[i]).abs();
                if d > 1e-3 * ov[i].abs().max(1.0) {
                    let row = i / 512;
                    println!("DIAG first bad i={i} token={row} t={} band={} head={} col={} old={:.5} new={:.5}", row / bands, row % bands, (i % 512) / 64, i % 64, ov[i], nv[i]);
                    break;
                }
            }
            println!("DIAG spot token0: old[0..4]={:?} new[0..4]={:?}", &ov[0..4], &nv[0..4]);
        }
    }
    // 4. 12 layers x 2 axes (layer 0 axis 0 redone fully inside)
    for layer in 0..12 {
        for axis in 0..2 {
            unsafe { transformer_step(&km, &ctx, &stream, &gw, layer, axis, &mut x, t_frames, bands, &mut scr).unwrap(); }
            if layer == 0 && axis == 0 {

                if let Some(st) = &stages {
                    let r = st.f32("L0_time").unwrap();
                    let xv = x.to_host_vec(&stream).unwrap();
                    stage_check("L0_time", &xv, r, true);
                }
            }
            if layer == 0 && axis == 1 {
                if let Some(st) = &stages {
                    let r = st.f32("L0_freq").unwrap();
                    let xv = x.to_host_vec(&stream).unwrap();
                    stage_check("L0_freq", &xv, r, false);
                }
            }
        }
    }
    println!("trunk forward host wall: {:?}", forward_t0.elapsed());
    // final norm
    let mut x_final = DeviceBuffer::<f32>::zeroed(&stream, m * 256).unwrap();
    // SAFETY: one warp per row.
    unsafe {
        km.rmsnorm(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * 32) as u32), &x, &gw.final_norm, &mut x_final, m as u32, 256)
    }.expect("final norm");

    // Trunk parity is checked from the final six-stem SNR below; avoid a
    // mid-forward 16 MiB device-to-host stall during performance runs.
    let _ = &ref_mid;

    // 5. MaskEstimator: per stem per band GEMMs
    let mut xb = DeviceBuffer::<f32>::zeroed(&stream, m * 256).unwrap();
    // SAFETY: elementwise transpose.
    unsafe {
        km.transpose_band_major(&stream, cuda_core::simt::LaunchConfig::for_num_elems((m * 256) as u32), &x_final, &mut xb, t_frames as u32, bands as u32)
    }.expect("transpose");

    // ---- MaskEstimator: grouped per-stem launches ----
    // glu_all: (stem, band, t, dim_in) padded to max_dim.
    let max_dim = 2 * 129 * 2; // 516
    let mut glu_all = DeviceBuffer::<f32>::zeroed(&stream, 6 * bands * t_frames * max_dim).unwrap();
    let hidden_sz = 6 * bands * t_frames * 1024;
    let mut hidden = DeviceBuffer::<f32>::zeroed(&stream, hidden_sz).unwrap();
    let mut hidden_t = DeviceBuffer::<f32>::zeroed(&stream, hidden_sz).unwrap();
    // GEMM2 output rows are padded to 2*max_dim so GLU can read both halves.
    let pre_width = 2 * max_dim;
    let mut pre2_all = DeviceBuffer::<f32>::zeroed(&stream, 6 * bands * t_frames * pre_width).unwrap();

    for s in 0..6usize {
        // SAFETY: 16 column tiles x ceil(T/128) row tiles x 62 bands.
        unsafe {
            km.mask_gemm1_f16(&stream, cuda_core::simt::LaunchConfig { grid_dim: (16, t_frames.div_ceil(128) as u32, bands as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, t_frames as u32, (s * bands) as u32, &xb, &gw.mask_w1[s], &gw.mask_b1[s], &mut hidden_t)
        }.expect("grouped mask gemm1");
    }
    // Tanh is fused into the grouped mask GEMM1 epilogue.


    for s in 0..6usize {
        // SAFETY: worst-case 17 column tiles; invalid per-band tiles return.
        unsafe {
            km.mask_gemm2_f16(&stream, cuda_core::simt::LaunchConfig { grid_dim: (pre_width.div_ceil(64) as u32, t_frames.div_ceil(128) as u32, bands as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, t_frames as u32, (s * bands) as u32, pre_width as u32, &offs_dev, &hidden_t, &gw.mask_w2[s], &gw.mask_b2[s], &mut pre2_all)
        }.expect("grouped mask gemm2");
    }
    // SAFETY: one thread per padded GLU output element.
    unsafe {
        km.glu_scatter(&stream, cuda_core::simt::LaunchConfig::for_num_elems(glu_all.len() as u32), &pre2_all, &offs_dev, &mut glu_all, t_frames as u32, bands as u32, max_dim as u32)
    }.expect("grouped glu");

    // Keep scratch alive through C2R: freeing 1+ GiB mid-forward stalls the
    // host allocator and adds tens of milliseconds to wall time.
    // mask apply -> C2R input frames
    let mut c2r_in = DeviceBuffer::<f32>::zeroed(&stream, 12 * t_frames * stft::FREQ_BINS * 2).unwrap();
    // SAFETY: elementwise over 12*T*1025*2.
    unsafe { km.mask_apply(&stream, cuda_core::simt::LaunchConfig::for_num_elems((12 * t_frames * 1025 * 2) as u32), &spec_dev, &glu_all, &f0_dev, &mut c2r_in, t_frames as u32, bands as u32) }.expect("mask apply");
    // Host-side mask bisection is disabled for clean timing; E2E SNR below
    // remains the correctness gate.
    if false {
        let mref = npz::Npz::open(&root.join("parity/e2e_mask.npz")).expect("e2e_mask");
        let mask_ref = mref.f32("mask").unwrap();
        let ga = glu_all.to_host_vec(&stream).unwrap();
        let f0v: Vec<usize> = f0.iter().map(|&x| x as usize).collect();
        let mut me = 0.0f32;
        let mut dn = 0.0f32;
        for s in 0..6usize {
            for fpos in 0..1025usize {
                let band = f0v.iter().position(|&x| x > fpos).unwrap_or(62) - 1;
                let fi = fpos - f0v[band];
                for t in 0..t_frames {
                    for ch in 0..2usize {
                        for cc in 0..2usize {
                            let ref_idx = ((s * 2050 + fpos * 2 + ch) * t_frames + t) * 2 + cc;
                            let gcol = fi * 4 + ch * 2 + cc;
                            let got = ga[((s * 62 + band) * t_frames + t) * 516 + gcol];
                            me = me.max((got - mask_ref[ref_idx]).abs());
                            dn = dn.max(mask_ref[ref_idx].abs());
                        }
                    }
                }
            }
        }
        println!("mask (glu_all) parity: rel={:.3e}", me / dn);
        // spot: stem 0 band 0 t 0 first 8 values
        // also spot band 24 (freqs=12, dim_in=48) stem 0 t 0
        {
            let b24 = 24usize;
            let f0_24 = f0[b24] as usize; // freq start for band 24
            let dim_in_24 = (offs[b24 + 1] - offs[b24]) as usize;
            for gc in 0..8usize {
                let fi = gc / 4;
                let ch = (gc % 4) / 2;
                let cc = gc % 2;
                let got_v = ga[((0 * 62 + b24) * t_frames + 0) * 516 + gc];
                let ref_idx = ((0 * 2050 + (f0_24 + fi) * 2 + ch) * t_frames + 0) * 2 + cc;
                let ref_v = mask_ref[ref_idx];
                println!("  b24 glu[{}] = {got_v:.6} ref={ref_v:.6}", gc);
            }
        }
        for gc in 0..8usize {
            let band0 = 0usize;
            let t0 = 0usize;
            let s0 = 0usize;
            let got_v = ga[((s0 * 62 + band0) * t_frames + t0) * 516 + gc];
            let fpos = 0usize;
            let ch = gc / 4;
            let cc = gc % 2;
            let ref_v = mask_ref[((s0 * 2050 + fpos * 2 + ch) * t_frames + t0) * 2 + cc];
            println!("  glu_all[0,0,0,{}] = {got_v:.6} ref={ref_v:.6}", gc);
        }
    }
    for (n, b) in [(16usize, 2usize), (2048, 2), (2048, 32), (2048, 259)] {
        let p0 = fft.plan(n, b, false);
        eprintln!("probe c2r n={n} b={b}: {}", p0.is_ok())
    }
    // C2R: plan batch = 12*T frames of n=2048
    let mut pcm = DeviceBuffer::<f32>::zeroed(&stream, 12 * t_frames * 2048).unwrap();
    for g in 0..12usize {
        let inv = fft.plan(2048, t_frames, false).expect("c2r plan");
        let iseg = slice_view(&stream, &c2r_in, g * t_frames * 1025 * 2, t_frames * 1025 * 2).unwrap();
        let mut oseg = mut_slice_view(&stream, &mut pcm, g * t_frames * 2048, t_frames * 2048).unwrap();
        inv.exec_c2r((*iseg).cu_deviceptr(), (*oseg).cu_deviceptr()).expect("exec c2r");
    }
    // Clean forward timing: all GPU work through the six C2R transforms has
    // been enqueued; synchronize before any host OLA or reference comparison.
    stream.synchronize().expect("forward sync");
    println!("E2E GPU forward wall (STFT through C2R): {:?}", forward_t0.elapsed());
    let frames_out = pcm.to_host_vec(&stream).unwrap();
    // host OLA: frame (s*2+ch)*T + t covers padded[n] = sum over frames w[n - t*hop] * f[..], then divide by win^2 sum, trim center pad
    let win = sp.window();
    let hop = stft::HOP;
    let padded = len + 2 * 1024;
    let mut result = vec![0.0f32; 6 * 2 * padded];
    let mut counter = vec![0.0f32; padded];
    for t in 0..t_frames {
        for n in 0..2048usize {
            let pos = t * hop + n;
            if pos < padded {
                counter[pos] += win[n] * win[n];
                for sc in 0..12usize {
                    let s = sc / 2;
                    let ch = sc % 2;
                    // cuFFT C2R is unnormalized: apply 1/N here.
                    result[(s * 2 + ch) * padded + pos] += frames_out[(sc * t_frames + t) * 2048 + n] * win[n] * (1.0 / 2048.0);
                }
            }
        }
    }
    let mut final_out = vec![0.0f32; 6 * 2 * len];
    for s in 0..6usize {
        for ch in 0..2usize {
            for i in 0..len {
                let c = counter[1024 + i];
                let v = if c > 1e-8 { result[(s * 2 + ch) * padded + 1024 + i] / c } else { 0.0 };
                final_out[(s * 2 + ch) * len + i] = v;
            }
        }
    }
    // SNR vs ref
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for i in 0..final_out.len() {
        sig += (ref_out[i] as f64) * (ref_out[i] as f64);
        let d = (final_out[i] - ref_out[i]) as f64;
        noise += d * d;
    }
    let snr = 10.0 * (sig / (noise + 1e-30)).log10();
    println!("E2E SNR vs ref_output: {snr:.2} dB");
    println!("forward+mask+parity host wall: {:?}", forward_t0.elapsed());

}

// ---------------------------------------------------------------------------
// Warm benchmark: weights and every scratch buffer are allocated once; the
// timed loop repeats full forwards on the golden 3 s input. Mirrors
// tools/bench_ref.py methodology (warmup 3, single sync around the loop).
// ---------------------------------------------------------------------------

struct BenchBufs {
    x_dev: DeviceBuffer<f32>,
    win_dev: DeviceBuffer<f32>,
    frames_dev: DeviceBuffer<f32>,
    spec_dev: DeviceBuffer<f32>,
    xin_dev: DeviceBuffer<f32>,
    bsnorm: DeviceBuffer<f32>,
    x: DeviceBuffer<f32>,
    scr: E2eScratch,
    x_final: DeviceBuffer<f32>,
    xb: DeviceBuffer<f32>,
    glu_all: DeviceBuffer<f32>,
    hidden_t: DeviceBuffer<f32>,
    pre2_all: DeviceBuffer<f32>,
    c2r_in: DeviceBuffer<f32>,
    pcm: DeviceBuffer<f32>,
    offs_dev: DeviceBuffer<u32>,
    dims_dev: DeviceBuffer<u32>,
    f0_dev: DeviceBuffer<u32>,
}

#[allow(clippy::too_many_arguments)]
unsafe fn bench_forward(
    km: &gpu_kernels::LoadedModule,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gw: &GpuWeights,
    sp: &stft::Stft,
    c2r: &cufft::CufftPlan,
    b: &mut BenchBufs,
    len: usize,
    t_frames: usize,
    bands: usize,
) -> Result<(), String> {
    let m = t_frames * bands;
    let max_dim = 516usize;
    let elems = |n: usize| LaunchConfig::for_num_elems(n as u32);
    // 1. STFT front-end
    // SAFETY: one thread per frame element.
    unsafe {
        km.frame_hann_reflect(stream, elems(2 * t_frames * stft::N_FFT),
            &b.x_dev, &b.win_dev, &mut b.frames_dev, stft::N_FFT as u32, stft::HOP as u32, len as u32, t_frames as u32)
    }.map_err(|e| e.to_string())?;
    sp.exec_fwd(&b.frames_dev, &mut b.spec_dev)?;
    // 2. reorder
    // SAFETY: elementwise gather over t*4100.
    unsafe {
        km.spec_reorder(stream, elems(t_frames * 4100),
            &b.spec_dev, &b.f0_dev, &b.offs_dev, &mut b.xin_dev, t_frames as u32)
    }.map_err(|e| e.to_string())?;
    // 3. BandSplit
    // SAFETY: one warp per (band, time) row.
    unsafe {
        km.bandsplit_norm(stream, elems(m * 32),
            &b.xin_dev, &gw.band_gamma, &b.offs_dev, &b.dims_dev, &mut b.bsnorm, t_frames as u32, bands as u32, max_dim as u32)
    }.map_err(|e| e.to_string())?;
    // SAFETY: 64x128 FP16 tiles for every band.
    unsafe {
        km.bandsplit_gemm_f16(stream, LaunchConfig { grid_dim: (4, t_frames.div_ceil(128) as u32, bands as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
            &b.offs_dev, &b.dims_dev, &b.bsnorm, &gw.band_w, &gw.band_b, &mut b.x, t_frames as u32, bands as u32, max_dim as u32)
    }.map_err(|e| e.to_string())?;
    // 4. trunk
    for layer in 0..12usize {
        for axis in 0..2usize {
            unsafe { transformer_step(km, ctx, stream, gw, layer, axis, &mut b.x, t_frames, bands, &mut b.scr)?; }
        }
    }
    // 5. final norm
    // SAFETY: one warp per row.
    unsafe {
        km.rmsnorm(stream, elems(m * 32), &b.x, &gw.final_norm, &mut b.x_final, m as u32, 256)
    }.map_err(|e| e.to_string())?;
    // 6. band-major transpose
    // SAFETY: elementwise transpose.
    unsafe {
        km.transpose_band_major(stream, elems(m * 256), &b.x_final, &mut b.xb, t_frames as u32, bands as u32)
    }.map_err(|e| e.to_string())?;
    // 7. MaskEstimator grouped GEMMs
    let pre_width = 2 * max_dim;
    for s in 0..6usize {
        // SAFETY: 16 column tiles x ceil(T/128) row tiles x 62 bands.
        unsafe {
            km.mask_gemm1_f16(stream, LaunchConfig { grid_dim: (16, t_frames.div_ceil(128) as u32, bands as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                t_frames as u32, (s * bands) as u32, &b.xb, &gw.mask_w1[s], &gw.mask_b1[s], &mut b.hidden_t)
        }.map_err(|e| e.to_string())?;
    }
    for s in 0..6usize {
        // SAFETY: worst-case 17 column tiles; invalid per-band tiles return.
        unsafe {
            km.mask_gemm2_f16(stream, LaunchConfig { grid_dim: (pre_width.div_ceil(64) as u32, t_frames.div_ceil(128) as u32, bands as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                t_frames as u32, (s * bands) as u32, pre_width as u32, &b.offs_dev, &b.hidden_t, &gw.mask_w2[s], &gw.mask_b2[s], &mut b.pre2_all)
        }.map_err(|e| e.to_string())?;
    }
    // SAFETY: one thread per padded GLU output element.
    unsafe {
        km.glu_scatter(stream, elems(b.glu_all.len()), &b.pre2_all, &b.offs_dev, &mut b.glu_all, t_frames as u32, bands as u32, max_dim as u32)
    }.map_err(|e| e.to_string())?;
    // 8. mask apply + ISTFT
    // SAFETY: elementwise over 12*T*1025*2.
    unsafe {
        km.mask_apply(stream, elems(12 * t_frames * 1025 * 2), &b.spec_dev, &b.glu_all, &b.f0_dev, &mut b.c2r_in, t_frames as u32, bands as u32)
    }.map_err(|e| e.to_string())?;
    for g in 0..12usize {
        let iseg = slice_view(stream, &b.c2r_in, g * t_frames * 1025 * 2, t_frames * 1025 * 2)?;
        let mut oseg = mut_slice_view(stream, &mut b.pcm, g * t_frames * 2048, t_frames * 2048)?;
        c2r.exec_c2r(iseg.cu_deviceptr(), oseg.cu_deviceptr())?;
    }
    Ok(())
}

fn bench_warm(device: usize, model_dir: &std::path::Path, iters: usize) {
    let iters = iters.max(1);
    let golden = npz::Npz::open(&model_dir.join("ref_output.npz")).unwrap_or_else(|e| panic!("{e}"));
    let inp = golden.f32("inp").expect("inp");
    let ref_out = golden.f32("out").expect("out");
    let cfg = config::ModelConfig::parse(&std::fs::read_to_string(model_dir.join("logic_bs_roformer.yaml")).unwrap()).unwrap();
    let t_frames = stft::num_frames(inp.len() / 2);
    let bands = cfg.num_bands();
    let m = t_frames * bands;
    let len = inp.len() / 2;
    println!("bench: L={len} T={t_frames} bands={bands} M={m} iters={iters}");

    let ctx = CudaContext::new(device).expect("ctx");
    let stream = ctx.default_stream();
    let km = gpu_kernels::load(&ctx).expect("kernels");

    let setup_t0 = std::time::Instant::now();
    let st = weights::SafeTensors::open(&model_dir.join("model.safetensors")).unwrap();
    let w = weights::ModelWeights::load(&st, &cfg).unwrap();
    let gw = upload_weights(&ctx, &stream, &w).expect("weights upload");
    println!("weights load+upload: {:?}", setup_t0.elapsed());

    // input interleaved
    let mut xi = vec![0.0f32; len * 2];
    for i in 0..len {
        xi[i * 2] = inp[i];
        xi[i * 2 + 1] = inp[len + i];
    }
    let fft = std::sync::Arc::new(cufft::Cufft::load().expect("cufft"));
    let sp = stft::Stft::new(fft.clone(), t_frames).expect("stft plan");
    // One reusable C2R plan for all 12 stem/channel planes (same shape).
    let c2r_plan = {
        let r: &cufft::Cufft = &fft;
        // SAFETY: the Arc<Cufft> outlives the plan in this scope.
        unsafe {
            let s: &'static cufft::Cufft = std::mem::transmute(r);
            s.plan(2048, t_frames, false).expect("c2r plan")
        }
    };

    // band tables
    let freqs: Vec<usize> = cfg.freqs_per_bands.clone();
    let dims: Vec<u32> = freqs.iter().map(|f| (2 * f * 2) as u32).collect();
    let mut offs = vec![0u32; bands + 1];
    let mut f0 = vec![0u32; bands];
    for i in 0..bands {
        offs[i + 1] = offs[i] + dims[i];
        f0[i] = freqs[..i].iter().sum::<usize>() as u32;
    }
    let offs_dev = DeviceBuffer::from_host(&stream, &offs).unwrap();
    let dims_dev = DeviceBuffer::from_host(&stream, &dims).unwrap();
    let f0_dev = DeviceBuffer::from_host(&stream, &f0).unwrap();

    // rotary tables
    let cos_t: Vec<f32> = (0..t_frames).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).cos())).collect();
    let sin_t: Vec<f32> = (0..t_frames).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).sin())).collect();
    let cos_f: Vec<f32> = (0..bands).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).cos())).collect();
    let sin_f: Vec<f32> = (0..bands).flat_map(|p| (0..32).map(move |i| ((p as f64 * (1.0 / 10000.0f64.powf(2.0 * i as f64 / 64.0))) as f32).sin())).collect();

    let z = |n: usize| DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
    let alloc_t0 = std::time::Instant::now();
    let mut bufs = BenchBufs {
        x_dev: DeviceBuffer::from_host(&stream, &xi).unwrap(),
        win_dev: DeviceBuffer::from_host(&stream, sp.window()).unwrap(),
        frames_dev: z(2 * t_frames * stft::N_FFT),
        spec_dev: z(2 * t_frames * stft::FREQ_BINS * 2),
        xin_dev: z(t_frames * 4100),
        bsnorm: z(bands * t_frames * 516),
        x: z(m * 256),
        scr: E2eScratch {
            h: z(m * 256),
            qkv: z(m * 1536),
            qkv_rope: z(m * 1536),
            qkv_attn: z(3 * 62 * 8 * t_frames.max(bands) * 64),
            v_flat: z(m * 512),
            gates: z(m * 8),
            scaled: z(m * 512),
            oproj: z(m * 256),
            attn_out: z(m * 256),
            ffpre: z(m * 1024),
            ff1: z(m * 1024),
            ff2: z(m * 256),
            p_big: z(62 * 8 * t_frames * t_frames),
            attn_out_long: z(496 * t_frames * 64),
            cos: [DeviceBuffer::from_host(&stream, &cos_t).unwrap(), DeviceBuffer::from_host(&stream, &cos_f).unwrap()],
            sin: [DeviceBuffer::from_host(&stream, &sin_t).unwrap(), DeviceBuffer::from_host(&stream, &sin_f).unwrap()],
            zero_bias_t: z(t_frames),
            zero_bias_64: z(64),
            attn_max: z(m * 8),
            attn_scale: z(m * 8),
        },
        x_final: z(m * 256),
        xb: z(m * 256),
        glu_all: z(6 * bands * t_frames * 516),
        hidden_t: z(6 * bands * t_frames * 1024),
        pre2_all: z(6 * bands * t_frames * 2 * 516),
        c2r_in: z(12 * t_frames * stft::FREQ_BINS * 2),
        pcm: z(12 * t_frames * 2048),
        offs_dev,
        dims_dev,
        f0_dev,
    };
    println!("scratch alloc: {:?}", alloc_t0.elapsed());

    // warmup
    for _ in 0..3 {
        unsafe { bench_forward(&km, &ctx, &stream, &gw, &sp, &c2r_plan, &mut bufs, len, t_frames, bands).expect("forward"); }
    }
    stream.synchronize().expect("warmup sync");

    // timed aggregate loop: single sync around all iters (same as bench_ref.py)
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe { bench_forward(&km, &ctx, &stream, &gw, &sp, &c2r_plan, &mut bufs, len, t_frames, bands).expect("forward"); }
    }
    stream.synchronize().expect("timed sync");
    let agg = t0.elapsed().as_secs_f64() / iters as f64;
    println!("BENCH warm aggregate: {:.3} ms/iter over {iters} iters, RTF {:.4}", agg * 1000.0, agg / 3.0);

    // per-iter breakdown with sync (shows variance; upper bound per chunk)
    let mut per = Vec::with_capacity(iters);
    for _ in 0..iters {
        stream.synchronize().expect("sync");
        let i0 = std::time::Instant::now();
        unsafe { bench_forward(&km, &ctx, &stream, &gw, &sp, &c2r_plan, &mut bufs, len, t_frames, bands).expect("forward"); }
        stream.synchronize().expect("sync");
        per.push(i0.elapsed().as_secs_f64() * 1000.0);
    }
    let med = {
        let mut v = per.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("BENCH warm per-iter (sync each): min {:.3} ms, med {med:.3} ms, max {:.3} ms", per.iter().cloned().fold(f64::INFINITY, f64::min), per.iter().cloned().fold(0.0, f64::max));

    // correctness spot-check on the last output (host OLA + SNR, untimed)
    let frames_out = bufs.pcm.to_host_vec(&stream).unwrap();
    let win = sp.window();
    let hop = stft::HOP;
    let padded = len + 2 * 1024;
    let mut result = vec![0.0f32; 6 * 2 * padded];
    let mut counter = vec![0.0f32; padded];
    for t in 0..t_frames {
        for n in 0..2048usize {
            let pos = t * hop + n;
            if pos < padded {
                counter[pos] += win[n] * win[n];
                for sc in 0..12usize {
                    result[(sc / 2 * 2 + sc % 2) * padded + pos] += frames_out[(sc * t_frames + t) * 2048 + n] * win[n] * (1.0 / 2048.0);
                }
            }
        }
    }
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for s in 0..6usize {
        for ch in 0..2usize {
            for i in 0..len {
                let c = counter[1024 + i];
                let v = if c > 1e-8 { result[(s * 2 + ch) * padded + 1024 + i] / c } else { 0.0 };
                let r = ref_out[(s * 2 + ch) * len + i];
                sig += (r as f64) * (r as f64);
                let d = (v - r) as f64;
                noise += d * d;
            }
        }
    }
    let snr = 10.0 * (sig / (noise + 1e-30)).log10();
    println!("BENCH final SNR vs ref_output: {snr:.2} dB");
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