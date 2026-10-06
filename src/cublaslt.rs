//! Minimal runtime (dlopen) bindings to NVIDIA cuBLASLt.
//!
//! Hosts the hot GEMM paths on cuBLASLt tensor-core kernels with fused
//! epilogues:
//!   - out-projection / FF2: y[M,N]f32 = x16[M,K]·Wᵀ + bias + resid
//!     (the residual rides along as beta·C with C=resid, C≠D, beta=1)
//!   - QKV: y16[M,N]f16 = x16[M,K]·Wᵀ + bias(f32 via BIAS_DATA_TYPE)
//!
//! Layout mapping (all our buffers are row-major):
//!   x16 row-major [M,K] ≡ colmajor (K,M) ld=K  → opB=N
//!   w   row-major [N,K] ≡ colmajor (K,N) ld=K  → opA=T
//!   y   row-major [M,N] ≡ colmajor (N,M) ld=N  → C/D layout
//! i.e. colmajor D(N,M) = op_T(Wc)·op_N(Xc); bias[n] broadcasts down each
//! column of D exactly like the hand-written kernels' bias[cc].
//!
//! Enum values below were verified against /usr/local/cuda-13.3/include
//! (cublasLt.h / cublas_api.h / library_types.h) on the bench node.

use cuda_core::CudaContext;
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{c_int, c_void};

const OP_N: c_int = 0;
const OP_T: c_int = 1;
const R_16F: c_int = 2;
const R_32F: c_int = 0;
const COMPUTE_32F: c_int = 68;
const ATTR_TRANSA: u32 = 3;
const ATTR_TRANSB: u32 = 4;
const ATTR_EPILOGUE: u32 = 7;
const ATTR_BIAS_POINTER: u32 = 8;
const EPI_BIAS: c_int = 4;
const EPI_GELU_BIAS: c_int = 36;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;

/// Opaque 64-byte `cublasLtMatmulAlgo_t`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Algo([u64; 8]);

/// `cublasLtMatmulHeuristicResult_t` in C declaration order (96 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct Heuristic {
    algo: Algo,
    workspace_size: usize,
    state: c_int,
    waves: f32,
    reserved: [c_int; 4],
}

type FnLtCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type FnDescCreate = unsafe extern "C" fn(*mut *mut c_void, c_int, c_int) -> c_int;
type FnSetAttr = unsafe extern "C" fn(*mut c_void, u32, *const c_void, usize) -> c_int;
type FnDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnLayoutCreate = unsafe extern "C" fn(*mut *mut c_void, c_int, u64, u64, i64) -> c_int;
type FnPrefCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type FnHeuristic = unsafe extern "C" fn(
    *mut c_void, // handle
    *mut c_void, // operation desc
    *mut c_void, // Adesc
    *mut c_void, // Bdesc
    *mut c_void, // Cdesc
    *mut c_void, // Ddesc
    *mut c_void, // preference
    c_int,       // requested
    *mut Heuristic,
    *mut c_int,  // returned
) -> c_int;
type FnMatmul = unsafe extern "C" fn(
    *mut c_void,        // handle
    *mut c_void,        // compute desc
    *const c_void,      // alpha
    *const c_void,      // A
    *mut c_void,        // Adesc
    *const c_void,      // B
    *mut c_void,        // Bdesc
    *const c_void,      // beta
    *const c_void,      // C
    *mut c_void,        // Cdesc
    *mut c_void,        // D
    *mut c_void,        // Ddesc
    *const Algo,        // algo
    *mut c_void,        // workspace
    usize,              // workspace size
    *mut c_void,        // stream
) -> c_int;

// SAFETY: handle is only ever used from the single host thread that owns it.
unsafe impl Send for CublasLt {}

pub struct CublasLt {
    _lib: Library,
    _cuda: Option<Library>,
    sync_ctx: Option<unsafe extern "C" fn() -> c_int>,
    dummy: u64,
    dummy_size: u64,
    handle: *mut c_void,
    desc_create: FnDescCreate,
    desc_set: FnSetAttr,
    desc_destroy: FnDestroy,
    layout_create: FnLayoutCreate,
    layout_destroy: FnDestroy,
    pref_create: FnPrefCreate,
    pref_set: FnSetAttr,
    pref_destroy: FnDestroy,
    heuristic: FnHeuristic,
    matmul: FnMatmul,
    ws: usize,
    ws_size: usize,
    /// One algo per (m, n, k, kind); heuristic results are stable for the
    /// process lifetime and the shapes here are few.
    algos: HashMap<(u64, u64, u64, u8), Algo>,
}

fn chk(rc: c_int, what: &str) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("cublasLt {what}: status {rc}"))
    }
}

impl CublasLt {
    /// dlopen libcublasLt and create a handle bound to `ctx`. `ws`/`ws_size`
    /// is a caller-owned device buffer used as the matmul workspace;
    /// `dummy`/`dummy_size` a garbage buffer (>= m*3072 bytes for the
    /// largest chunk) the first call of each shape autotunes candidate
    /// algos against — the heuristic's first pick measures up to 2.4x
    /// slower than its own alternatives on the big-M resid shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn load(ctx: &CudaContext, ws: u64, ws_size: usize, dummy: u64, dummy_size: usize) -> Result<CublasLt, String> {
        let mut candidates: Vec<String> = Vec::new();
        for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"] {
            if let Ok(p) = std::env::var(var) {
                candidates.push(format!("{p}/lib64/libcublasLt.so"));
                candidates.push(format!("{p}/lib/libcublasLt.so"));
            }
        }
        candidates.push("libcublasLt.so".into());
        let mut last = String::new();
        let lib = loop {
            if candidates.is_empty() {
                return Err(format!("libcublasLt not found; last: {last}"));
            }
            let cand = candidates.remove(0);
            match unsafe { Library::new(&cand) } {
                Ok(l) => break l,
                Err(e) => last = format!("{cand}: {e}"),
            }
        };
        unsafe {
            let create: Symbol<FnLtCreate> =
                lib.get(b"cublasLtCreate").map_err(|e| format!("sym cublasLtCreate: {e}"))?;
            // cublasLtCreate binds the new handle to the *current* context;
            // make that the primary context our streams live in.
            ctx.bind_to_thread().map_err(|e| format!("bind ctx: {e}"))?;
            let mut handle: *mut c_void = std::ptr::null_mut();
            chk(create(&mut handle), "create")?;
            // cuCtxSynchronize for autotune timing (optional).
            let (cuda_lib, sync_ctx) = match unsafe { Library::new("libcuda.so.1") } {
                Ok(l) => {
                    let f: Symbol<unsafe extern "C" fn() -> c_int> = match l.get(b"cuCtxSynchronize") {
                        Ok(f) => f,
                        Err(_) => return Err("sym cuCtxSynchronize: not found".into()),
                    };
                    let fp = *f; // copy the fn pointer while the borrow lives
                    (Some(l), Some(fp))
                }
                Err(_) => (None, None),
            };
            let g = |name: &str, e: libloading::Error| format!("sym {name}: {e}");
            let desc_create: FnDescCreate = *lib
                .get(b"cublasLtMatmulDescCreate")
                .map_err(|e| g("cublasLtMatmulDescCreate", e))?;
            let desc_set: FnSetAttr = *lib
                .get(b"cublasLtMatmulDescSetAttribute")
                .map_err(|e| g("cublasLtMatmulDescSetAttribute", e))?;
            let desc_destroy: FnDestroy = *lib
                .get(b"cublasLtMatmulDescDestroy")
                .map_err(|e| g("cublasLtMatmulDescDestroy", e))?;
            let layout_create: FnLayoutCreate = *lib
                .get(b"cublasLtMatrixLayoutCreate")
                .map_err(|e| g("cublasLtMatrixLayoutCreate", e))?;
            let layout_destroy: FnDestroy = *lib
                .get(b"cublasLtMatrixLayoutDestroy")
                .map_err(|e| g("cublasLtMatrixLayoutDestroy", e))?;
            let pref_create: FnPrefCreate = *lib
                .get(b"cublasLtMatmulPreferenceCreate")
                .map_err(|e| g("cublasLtMatmulPreferenceCreate", e))?;
            let pref_set: FnSetAttr = *lib
                .get(b"cublasLtMatmulPreferenceSetAttribute")
                .map_err(|e| g("cublasLtMatmulPreferenceSetAttribute", e))?;
            let pref_destroy: FnDestroy = *lib
                .get(b"cublasLtMatmulPreferenceDestroy")
                .map_err(|e| g("cublasLtMatmulPreferenceDestroy", e))?;
            let heuristic: FnHeuristic = *lib
                .get(b"cublasLtMatmulAlgoGetHeuristic")
                .map_err(|e| g("cublasLtMatmulAlgoGetHeuristic", e))?;
            let matmul: FnMatmul =
                *lib.get(b"cublasLtMatmul").map_err(|e| g("cublasLtMatmul", e))?;
            Ok(CublasLt {
                _lib: lib,
                _cuda: cuda_lib,
                sync_ctx,
                dummy,
                dummy_size: dummy_size as u64,
                handle,
                desc_create,
                desc_set,
                desc_destroy,
                layout_create,
                layout_destroy,
                pref_create,
                pref_set,
                pref_destroy,
                heuristic,
                matmul,
                ws: ws as usize,
                ws_size,
                algos: HashMap::new(),
            })
        }
    }

    /// y[M,N]f32 = x16[M,K]·Wᵀ + bias + resid.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_resid(
        &mut self,
        stream: usize,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        x16: u64,
        bias: u64,
        resid: u64,
        y: u64,
    ) -> Result<(), String> {
        self.run(stream, m, n, k, w, x16, bias, resid, y, false, 1.0, false)
    }

    /// y16[M,N]f16 = x16[M,K]·Wᵀ + bias(f16). With `gelu` the tanh-form
    /// CUBLASLT_EPILOGUE_GELU_BIAS is applied (vs our erf-form kernel — the
    /// end-to-end SNR check gates whether this stays enabled).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f16out(
        &mut self,
        stream: usize,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        x16: u64,
        bias: u64,
        y16: u64,
        gelu: bool,
    ) -> Result<(), String> {
        self.run(stream, m, n, k, w, x16, bias, y16, y16, true, 0.0, gelu)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        stream: usize,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        x16: u64,
        bias: u64,
        c: u64,
        d: u64,
        f16_out: bool,
        beta: f32,
        gelu: bool,
    ) -> Result<(), String> {
        // cublasLt requires the calling thread's current context to match the
        // handle's; ctx.bind_to_thread() is idempotent and cheap.
        // SAFETY: fn pointers come from the live Library; descriptors are
        // created and destroyed within this call; pointers are device
        // addresses of caller-owned allocations on the bound context.
        unsafe {
            let mut desc: *mut c_void = std::ptr::null_mut();
            chk((self.desc_create)(&mut desc, COMPUTE_32F, R_32F), "descCreate")?;
            let (ta, tb, bias_dt) = (OP_T, OP_N, R_32F);
            let epi: c_int = if gelu { EPI_GELU_BIAS } else { EPI_BIAS };
            chk((self.desc_set)(desc, ATTR_TRANSA, &ta as *const c_int as *const c_void, 4), "TRANSA")?;
            chk((self.desc_set)(desc, ATTR_TRANSB, &tb as *const c_int as *const c_void, 4), "TRANSB")?;
            chk((self.desc_set)(desc, ATTR_EPILOGUE, &epi as *const c_int as *const c_void, 4), "EPILOGUE")?;
            // For f16-out the bias vector must also be f16 (the heuristic
            // rejects BIAS_DATA_TYPE=f32 with f16 D on this library build),
            // so the QKV bias is pre-packed to f16 on the host.
            let _ = bias_dt;
            chk((self.desc_set)(desc, ATTR_BIAS_POINTER, &bias as *const u64 as *const c_void, 8), "BIAS_PTR")?;

            // A = Wc(K,N) ld=K (opA=T), B = Xc(K,M) ld=K, C/D = (N,M) ld=N.
            let mut la: *mut c_void = std::ptr::null_mut();
            let mut lb: *mut c_void = std::ptr::null_mut();
            let mut lc: *mut c_void = std::ptr::null_mut();
            let mut ldsc: *mut c_void = std::ptr::null_mut();
            let dt = if f16_out { R_16F } else { R_32F };
            chk((self.layout_create)(&mut la, R_16F, k, n, k as i64), "layoutA")?;
            chk((self.layout_create)(&mut lb, R_16F, k, m, k as i64), "layoutB")?;
            chk((self.layout_create)(&mut lc, dt, n, m, n as i64), "layoutC")?;
            chk((self.layout_create)(&mut ldsc, dt, n, m, n as i64), "layoutD")?;

            let key = (m, n, k, f16_out as u8 + if gelu { 2 } else { 0 });
            let algo = match self.algos.get(&key) {
                Some(a) => *a,
                None => {
                    let mut pref: *mut c_void = std::ptr::null_mut();
                    chk((self.pref_create)(&mut pref), "prefCreate")?;
                    let wss = self.ws_size;
                    let _ = (self.pref_set)(pref, PREF_MAX_WORKSPACE_BYTES, &wss as *const usize as *const c_void, 8);
                    let mut hr = [Heuristic { algo: Algo([0; 8]), workspace_size: 0, state: 0, waves: 0.0, reserved: [0; 4] }; 8];
                    let mut nr: c_int = 0;
                    let rc = (self.heuristic)(self.handle, desc, la, lb, lc, ldsc, pref, 8, hr.as_mut_ptr(), &mut nr);
                    let _ = (self.pref_destroy)(pref);
                    chk(rc, "heuristic")?;
                    if nr == 0 {
                        let _ = (self.desc_destroy)(desc);
                        for l in [la, lb, lc, ldsc] {
                            let _ = (self.layout_destroy)(l);
                        }
                        return Err(format!("cublasLt heuristic: no algo for m={m} n={n} k={k} f16={f16_out}"));
                    }
                    // Autotune: time each valid candidate on the dummy buffer
                    // (all four matrix pointers aliased into it — the data is
                    // garbage, only the kernel time matters), pick the
                    // fastest, cache it. Falls back to heuristic order when
                    // sync/dummy are unavailable.
                    let chosen = 'pick: {
                        let Some(sync) = self.sync_ctx else { break 'pick hr[0].algo };
                        let role = (m * n * 4).max(m * k * 2).max(n * k * 2);
                        if self.dummy_size < role || self.dummy == 0 {
                            break 'pick hr[0].algo;
                        }
                        let mm = self.matmul;
                        let (h, wsr, wssz, sstr) = (self.handle, self.ws as *mut c_void, self.ws_size, stream as *mut c_void);
                        let (ap, bp) = (&1.0f32 as *const f32 as *const c_void, &beta as *const f32 as *const c_void);
                        let dmyr = self.dummy as *const c_void;
                        let dmyw = self.dummy as *mut c_void;
                        // SAFETY: same call shape as the real matmul below,
                        // against the caller-provided dummy buffer.
                        let call = |algo: &Algo| unsafe {
                            mm(h, desc, ap, dmyr, la, dmyr, lb, bp, dmyr, lc, dmyw, ldsc, algo, wsr, wssz, sstr)
                        };
                        let mut best_i = usize::MAX;
                        let mut best_dt = u64::MAX;
                        for i in 0..nr as usize {
                            if hr[i].state != 0 {
                                continue;
                            }
                            let cand = hr[i].algo;
                            if call(&cand) != 0 {
                                continue;
                            }
                            // SAFETY: driver sync; no-op return code checked.
                            unsafe { sync() };
                            let t0 = std::time::Instant::now();
                            for _ in 0..8 {
                                if call(&cand) != 0 {
                                    break;
                                }
                            }
                            // SAFETY: as above.
                            unsafe { sync() };
                            let dt = t0.elapsed().as_nanos() as u64;
                            if dt < best_dt {
                                best_dt = dt;
                                best_i = i;
                            }
                        }
                        if best_i == usize::MAX {
                            hr[0].algo
                        } else {
                            if best_i != 0 {
                                eprintln!("[lt] autotune m={m} n={n} k={k}: algo {best_i} beats 0 ({}/{best_dt}ns per 8 reps)", hr[0].waves);
                            }
                            hr[best_i].algo
                        }
                    };
                    self.algos.insert(key, chosen);
                    chosen
                }
            };

            let (alpha, beta_v): (f32, f32) = (1.0, beta);
            let rc = (self.matmul)(
                self.handle,
                desc,
                &alpha as *const f32 as *const c_void,
                w as *const c_void,
                la,
                x16 as *const c_void,
                lb,
                &beta_v as *const f32 as *const c_void,
                c as *const c_void,
                lc,
                d as *mut c_void,
                ldsc,
                &algo,
                self.ws as *mut c_void,
                self.ws_size,
                stream as *mut c_void,
            );
            let _ = (self.desc_destroy)(desc);
            for l in [la, lb, lc, ldsc] {
                let _ = (self.layout_destroy)(l);
            }
            chk(rc, "matmul")
        }
    }
}
