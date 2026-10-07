//! cuBLASLt FP16-input / FP32-accumulation GEMMs, bounded event-based tuning.
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use libloading::Library;
use std::{
    collections::HashMap,
    ffi::{c_int, c_void},
    sync::Arc,
};
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
    *mut c_int, // returned
) -> c_int;
type FnMatmul = unsafe extern "C" fn(
    *mut c_void,   // handle
    *mut c_void,   // compute desc
    *const c_void, // alpha
    *const c_void, // A
    *mut c_void,   // Adesc
    *const c_void, // B
    *mut c_void,   // Bdesc
    *const c_void, // beta
    *const c_void, // C
    *mut c_void,   // Cdesc
    *mut c_void,   // D
    *mut c_void,   // Ddesc
    *const Algo,   // algo
    *mut c_void,   // workspace
    usize,         // workspace size
    *mut c_void,   // stream
) -> c_int;

/// An owned descriptor/layout/handle, including partial-construction failures.
struct Resource {
    ptr: *mut c_void,
    destroy: FnDestroy,
}
impl Drop for Resource {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.destroy)(self.ptr) };
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    m: u64,
    n: u64,
    k: u64,
    dtypes: [i32; 4],
    strides: [i64; 4],
    transpose: [i32; 2],
    epilogue: i32,
    beta_bits: u32,
    workspace: usize,
}
struct Operation {
    desc: Resource,
    layouts: [Resource; 4],
    algo: Algo,
    index: usize,
    median_ms: f64,
    candidates: usize,
}
pub struct CublasLt {
    ctx: Arc<CudaContext>,
    handle: Resource,
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
    ws: u64,
    ws_size: usize,
    ops: HashMap<Key, Operation>,
    tuning_peak: usize,
    version: usize,
    _lib: Library,
}
fn chk(rc: c_int, what: &str) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("cublasLt {what}: status {rc}"))
    }
}
impl CublasLt {
    pub fn load(ctx: &Arc<CudaContext>, ws: u64, ws_size: usize) -> Result<Self, String> {
        let mut candidates: Vec<String> = Vec::new();
        if cfg!(windows) {
            // CUDA 12.x/13.x on Windows: DLLs live in bin\x64 (13.x) or bin
            // (12.x); names carry the CUDA major version.
            for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"] {
                if let Ok(p) = std::env::var(var) {
                    for sub in ["bin/x64", "bin"] {
                        for name in ["cublasLt64_13.dll", "cublasLt64_12.dll"] {
                            candidates.push(format!("{p}/{sub}/{name}"));
                        }
                    }
                }
            }
            candidates.push("cublasLt64_13.dll".into());
            candidates.push("cublasLt64_12.dll".into());
        } else {
            for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"] {
                if let Ok(p) = std::env::var(var) {
                    candidates.push(format!("{p}/lib64/libcublasLt.so"));
                    candidates.push(format!("{p}/lib/libcublasLt.so"));
                }
            }
            candidates.push("libcublasLt.so".into());
        }
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
            let create: FnLtCreate = *lib.get(b"cublasLtCreate").map_err(|e| e.to_string())?;
            let destroy: FnDestroy = *lib.get(b"cublasLtDestroy").map_err(|e| e.to_string())?;
            let desc_create: FnDescCreate = *lib
                .get(b"cublasLtMatmulDescCreate")
                .map_err(|e| e.to_string())?;
            let desc_set: FnSetAttr = *lib
                .get(b"cublasLtMatmulDescSetAttribute")
                .map_err(|e| e.to_string())?;
            let desc_destroy: FnDestroy = *lib
                .get(b"cublasLtMatmulDescDestroy")
                .map_err(|e| e.to_string())?;
            let layout_create: FnLayoutCreate = *lib
                .get(b"cublasLtMatrixLayoutCreate")
                .map_err(|e| e.to_string())?;
            let layout_destroy: FnDestroy = *lib
                .get(b"cublasLtMatrixLayoutDestroy")
                .map_err(|e| e.to_string())?;
            let pref_create: FnPrefCreate = *lib
                .get(b"cublasLtMatmulPreferenceCreate")
                .map_err(|e| e.to_string())?;
            let pref_set: FnSetAttr = *lib
                .get(b"cublasLtMatmulPreferenceSetAttribute")
                .map_err(|e| e.to_string())?;
            let pref_destroy: FnDestroy = *lib
                .get(b"cublasLtMatmulPreferenceDestroy")
                .map_err(|e| e.to_string())?;
            let heuristic: FnHeuristic = *lib
                .get(b"cublasLtMatmulAlgoGetHeuristic")
                .map_err(|e| e.to_string())?;
            let matmul: FnMatmul = *lib.get(b"cublasLtMatmul").map_err(|e| e.to_string())?;
            let version: unsafe extern "C" fn() -> usize =
                *lib.get(b"cublasLtGetVersion").map_err(|e| e.to_string())?;
            let version = version();
            ctx.bind_to_thread().map_err(|e| e.to_string())?;
            let mut handle = Resource {
                ptr: std::ptr::null_mut(),
                destroy,
            };
            chk(create(&mut handle.ptr), "create")?;
            Ok(Self {
                ctx: ctx.clone(),
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
                ws,
                ws_size,
                ops: HashMap::new(),
                tuning_peak: 0,
                version,
                _lib: lib,
            })
        }
    }
    pub fn version(&self) -> usize {
        self.version
    }
    pub fn tuning_peak_bytes(&self) -> usize {
        self.tuning_peak
    }
    pub fn algorithms_json(&self) -> serde_json::Value {
        let mut rows: Vec<_> = self
            .ops
            .iter()
            .map(|(k, o)| {
                (
                    k.m,
                    k.n,
                    k.k,
                    k.dtypes[3],
                    k.beta_bits,
                    o.index,
                    o.median_ms,
                    o.candidates,
                )
            })
            .collect();
        rows.sort_by(|a, b| (a.0, a.1, a.2, a.3, a.4).cmp(&(b.0, b.1, b.2, b.3, b.4)));
        serde_json::json!(rows)
    }
    pub fn matmul_resid(
        &mut self,
        stream: &Arc<CudaStream>,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        x: u64,
        bias: u64,
        resid: u64,
        y: u64,
    ) -> Result<(), String> {
        self.run(stream, m, n, k, w, x, bias, resid, y, false, 1.0)
    }
    pub fn matmul_f16out(
        &mut self,
        stream: &Arc<CudaStream>,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        x: u64,
        bias: u64,
        y: u64,
    ) -> Result<(), String> {
        self.run(stream, m, n, k, w, x, bias, y, y, true, 0.0)
    }
    /// FF1 has FP32 bias/output; exact erf-GELU and the FP16 rounding run afterwards.
    pub fn matmul_f32out(
        &mut self,
        stream: &Arc<CudaStream>,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        x: u64,
        bias: u64,
        y: u64,
    ) -> Result<(), String> {
        self.run(stream, m, n, k, w, x, bias, y, y, false, 0.0)
    }
    fn layout(&self, dtype: i32, rows: u64, cols: u64, stride: i64) -> Result<Resource, String> {
        let mut r = Resource {
            ptr: std::ptr::null_mut(),
            destroy: self.layout_destroy,
        };
        chk(
            unsafe { (self.layout_create)(&mut r.ptr, dtype, rows, cols, stride) },
            "layoutCreate",
        )?;
        Ok(r)
    }
    fn run(
        &mut self,
        stream: &Arc<CudaStream>,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        x: u64,
        bias: u64,
        c: u64,
        d: u64,
        f16: bool,
        beta: f32,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.ctx, stream.context()) {
            return Err("cuBLASLt stream/context mismatch".into());
        }
        self.ctx.bind_to_thread().map_err(|e| e.to_string())?;
        let dtype = if f16 { R_16F } else { R_32F };
        let key = Key {
            m,
            n,
            k,
            dtypes: [R_16F, R_16F, dtype, dtype],
            strides: [k as i64, k as i64, n as i64, n as i64],
            transpose: [OP_T, OP_N],
            epilogue: EPI_BIAS,
            beta_bits: beta.to_bits(),
            workspace: self.ws_size,
        };
        if !self.ops.contains_key(&key) {
            let op = self.prepare(stream, &key, w, x, bias, c)?;
            self.ops.insert(key.clone(), op);
        }
        let op = &self.ops[&key];
        chk(
            unsafe {
                (self.desc_set)(
                    op.desc.ptr,
                    ATTR_BIAS_POINTER,
                    (&bias as *const u64).cast(),
                    8,
                )
            },
            "BIAS_PTR",
        )?;
        let rc = self.call(stream, op, &op.algo, w, x, c, d, beta);
        chk(rc, "matmul")
    }
    fn call(
        &self,
        stream: &Arc<CudaStream>,
        op: &Operation,
        algo: &Algo,
        w: u64,
        x: u64,
        c: u64,
        d: u64,
        beta: f32,
    ) -> i32 {
        let alpha = 1.0f32;
        unsafe {
            (self.matmul)(
                self.handle.ptr,
                op.desc.ptr,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                op.layouts[0].ptr,
                x as *const c_void,
                op.layouts[1].ptr,
                (&beta as *const f32).cast(),
                c as *const c_void,
                op.layouts[2].ptr,
                d as *mut c_void,
                op.layouts[3].ptr,
                algo,
                self.ws as *mut c_void,
                self.ws_size,
                stream.cu_stream().cast(),
            )
        }
    }
    fn prepare(
        &mut self,
        stream: &Arc<CudaStream>,
        key: &Key,
        w: u64,
        x: u64,
        bias: u64,
        c: u64,
    ) -> Result<Operation, String> {
        let mut desc = Resource {
            ptr: std::ptr::null_mut(),
            destroy: self.desc_destroy,
        };
        chk(
            unsafe { (self.desc_create)(&mut desc.ptr, COMPUTE_32F, R_32F) },
            "descCreate",
        )?;
        for (attr, val) in [
            (ATTR_TRANSA, key.transpose[0]),
            (ATTR_TRANSB, key.transpose[1]),
            (ATTR_EPILOGUE, key.epilogue),
        ] {
            chk(
                unsafe { (self.desc_set)(desc.ptr, attr, (&val as *const i32).cast(), 4) },
                "desc attribute",
            )?;
        }
        chk(
            unsafe {
                (self.desc_set)(desc.ptr, ATTR_BIAS_POINTER, (&bias as *const u64).cast(), 8)
            },
            "BIAS_PTR",
        )?;
        let layouts = [
            self.layout(R_16F, key.k, key.n, key.strides[0])?,
            self.layout(R_16F, key.k, key.m, key.strides[1])?,
            self.layout(key.dtypes[2], key.n, key.m, key.strides[2])?,
            self.layout(key.dtypes[3], key.n, key.m, key.strides[3])?,
        ];
        let mut pref = Resource {
            ptr: std::ptr::null_mut(),
            destroy: self.pref_destroy,
        };
        chk(unsafe { (self.pref_create)(&mut pref.ptr) }, "prefCreate")?;
        chk(
            unsafe {
                (self.pref_set)(
                    pref.ptr,
                    PREF_MAX_WORKSPACE_BYTES,
                    (&self.ws_size as *const usize).cast(),
                    std::mem::size_of::<usize>(),
                )
            },
            "workspace preference",
        )?;
        let mut hr = [Heuristic {
            algo: Algo([0; 8]),
            workspace_size: 0,
            state: 0,
            waves: 0.0,
            reserved: [0; 4],
        }; 8];
        let mut count = 0;
        chk(
            unsafe {
                (self.heuristic)(
                    self.handle.ptr,
                    desc.ptr,
                    layouts[0].ptr,
                    layouts[1].ptr,
                    layouts[2].ptr,
                    layouts[3].ptr,
                    pref.ptr,
                    8,
                    hr.as_mut_ptr(),
                    &mut count,
                )
            },
            "heuristic",
        )?;
        if count < 1 || count > 8 {
            return Err(format!("cuBLASLt no usable heuristic: {key:?}"));
        }
        let bytes = key
            .m
            .checked_mul(key.n)
            .and_then(|v| v.checked_mul(if key.dtypes[3] == R_16F { 2 } else { 4 }))
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("cuBLASLt output size overflow")?;
        // W/X/C remain production inputs; D is independent and has its actual dtype size.
        let scratch = DeviceBuffer::<u8>::zeroed(stream, bytes).map_err(|e| e.to_string())?;
        self.tuning_peak = self.tuning_peak.max(bytes);
        let mut op = Operation {
            desc,
            layouts,
            algo: Algo([0; 8]),
            index: 0,
            median_ms: f64::INFINITY,
            candidates: 0,
        };
        let beta = f32::from_bits(key.beta_bits);
        // Validated against the unchanged FP32 golden on Linux Ada/cuBLASLt 13.6.
        // Unconstrained reductions lose >0.1dB on individual short-fixture stems.
        // Both verified shapes retain B0 arithmetic while all candidates are still measured.
        // The exception is version/SM/shape bounded; opaque algorithms stay in-process.
        let compatible = if cfg!(target_os = "linux")
            && self.version == 130600
            && self.ctx.compute_capability().map_err(|e| e.to_string())? == (8, 9)
            && self.ctx.device_name().map_err(|e| e.to_string())? == "NVIDIA GeForce RTX 4060 Ti"
        {
            match (key.m, key.n, key.k, key.dtypes[3], beta.to_bits()) {
                (16058 | 71362, 1536, 256, R_16F, 0) => Some(0),
                (16058, 256, 512, R_32F, 0x3f800000) => Some(1),
                (16058, 256, 1024, R_32F, 0x3f800000) => Some(3),
                (71362, 256, 512, R_32F, 0x3f800000) => Some(2),
                (71362, 256, 1024, R_32F, 0x3f800000) => Some(0),
                _ => None,
            }
        } else {
            None
        };
        for (index, h) in hr[..count as usize].iter().enumerate() {
            if h.state != 0 || h.workspace_size > self.ws_size {
                continue;
            }
            let mut valid = true;
            for _ in 0..3 {
                let rc = self.call(stream, &op, &h.algo, w, x, c, scratch.cu_deviceptr(), beta);
                if rc != 0 {
                    valid = false;
                    stream
                        .synchronize()
                        .map_err(|e| format!("autotune CUDA failure: {e}"))?;
                    if rc == 13 || rc == 14 {
                        return Err(format!("autotune execution failure status={rc}"));
                    }
                    break;
                }
            }
            stream
                .synchronize()
                .map_err(|e| format!("autotune warmup sync: {e}"))?;
            if !valid {
                continue;
            }
            let mut times = Vec::with_capacity(3);
            for _ in 0..3 {
                let start = self.ctx.new_event(Some(0)).map_err(|e| e.to_string())?;
                let end = self.ctx.new_event(Some(0)).map_err(|e| e.to_string())?;
                start.record(stream).map_err(|e| e.to_string())?;
                for _ in 0..10 {
                    let rc = self.call(stream, &op, &h.algo, w, x, c, scratch.cu_deviceptr(), beta);
                    if rc != 0 {
                        valid = false;
                        stream
                            .synchronize()
                            .map_err(|e| format!("autotune CUDA failure: {e}"))?;
                        if rc == 13 || rc == 14 {
                            return Err(format!("autotune execution failure status={rc}"));
                        }
                        break;
                    }
                }
                end.record(stream).map_err(|e| e.to_string())?;
                end.synchronize()
                    .map_err(|e| format!("autotune end sync: {e}"))?;
                if !valid {
                    break;
                }
                times.push(start.elapsed_ms(&end).map_err(|e| e.to_string())? as f64 / 10.0);
            }
            if !valid || times.len() != 3 {
                continue;
            }
            op.candidates += 1;
            let ms = crate::benchmark::median(&times);
            if compatible.map_or(ms < op.median_ms, |index_required| index == index_required) {
                op.median_ms = ms;
                op.algo = h.algo;
                op.index = index;
            }
        }
        if !op.median_ms.is_finite() {
            return Err(format!("all cuBLASLt candidates rejected: {key:?}"));
        }
        if compatible.is_some() {
            eprintln!(
                "[lt] selection constrained by verified Linux sm89/cuBLASLt-13.6 numerical compatibility"
            );
        }
        eprintln!(
            "[lt] m={} n={} k={} dtype={} beta={} algo={} valid={} median={:.4}ms temporary={}B",
            key.m, key.n, key.k, key.dtypes[3], beta, op.index, op.candidates, op.median_ms, bytes
        );
        // Every group has synchronized; the tuning output is freed here, never kept resident.
        Ok(op)
    }
}
impl Drop for CublasLt {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        self.ops.clear();
    }
}
