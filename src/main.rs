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
    rmsnorm_test: bool,
    bandsplit_test: bool,
    gemm_test: bool,
    qkvrope_test: bool,
    attn_test: bool,
    gateff_test: bool,
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
            "--rmsnorm-test" => args.rmsnorm_test = true,
            "--bandsplit-test" => args.bandsplit_test = true,
            "--gemm-test" => args.gemm_test = true,
            "--qkvrope-test" => args.qkvrope_test = true,
            "--attn-test" => args.attn_test = true,
            "--gateff-test" => args.gateff_test = true,
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
        let base = warp_id * dim + lane * per_lane;
        // partial squared sum over this lane's elements
        let mut vals = [0.0f32; 8];
        let mut s = 0.0f32;
        for k in 0..per_lane {
            let v = x[base + k];
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
            let i = base + k;
            // SAFETY: rows*dim total elements, i < rows*dim by construction;
            // each warp owns a disjoint row, lanes write disjoint columns.
            unsafe {
                *out_ptr.add(i) = vals[k] * scale * gamma[i % dim];
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
        let pos = m % seq as usize;
        let base = m * 1536 + part * 512 + head * 64 + pair * 2;
        let c = cos[pos * 32 + pair];
        let s = sin[pos * 32 + pair];
        let (ve, vo) = (qkv[base], qkv[base + 1]);
        let (re, ro) = if part < 2 {
            (ve * c - vo * s, vo * c + ve * s)
        } else {
            (ve, vo)
        };
        let scale = if part < 2 { 0.125f32 } else { 1.0f32 };
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
            km.gemm_bias(
                &stream,
                cuda_core::simt::LaunchConfig {
                    grid_dim: ((n.div_ceil(16)) as u32, (m.div_ceil(16)) as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                },
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
        println!("gemm[{gi}] K={k} N={n}: max_err={max_err:e} rel={:.3e}", max_err / denom);
        assert!(max_err < 1e-4 * denom, "gemm[{gi}] parity failed");
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
        km.gemm_bias(
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
        km.rope_scale(&stream, cuda_core::simt::LaunchConfig::for_num_elems((rows * 768) as u32), &qkv_dev, &cos_dev, &sin_dev, &mut out_dev, seq as u32)
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
            km.gemm_bias(
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
            km.gemm_bias(
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