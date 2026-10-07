use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::sync::Arc;

const ATTN_SCALE: f32 = 0.125;

type FnInit = unsafe extern "C" fn(*const c_char) -> c_int;
type FnCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type FnSetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type FnBuild = unsafe extern "C" fn(
    *mut c_void, // handle
    i64,
    i64,
    i64,
    i64, // b, heads, s, d
    *const i64,
    *const i64,
    *const i64,
    *const i64, // q/k/v/o strides (elements)
    f32,        // attn scale
    c_int,
    c_int,            // heur mode, io dtype
    *mut *mut c_void, // graph out
    *mut i64,         // workspace size out
) -> c_int;
type FnExec = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
) -> c_int;
type FnFreeGraph = unsafe extern "C" fn(*mut c_void);
type FnDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnVersion = unsafe extern "C" fn() -> c_int;
type FnErr = unsafe extern "C" fn() -> *const c_char;

unsafe impl Send for CudnnSdpa {}

/// Complete graph identity. Equal B/S does not imply equal folded-axis strides.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanKey {
    axis: usize,
    b: i64,
    h: i64,
    s: i64,
    d: i64,
    dtype: i32,
    q: [i64; 4],
    k: [i64; 4],
    v: [i64; 4],
    o: [i64; 4],
    scale_bits: u32,
}
impl PlanKey {
    pub fn new(axis: usize, bands: i64, frames: i64) -> Result<Self, String> {
        if axis > 1 || bands < 1 || frames < 1 {
            return Err("invalid SDPA axis/shape".into());
        }
        let (b, s) = if axis == 0 {
            (bands, frames)
        } else {
            (frames, bands)
        };
        let (q, k, o) = if axis == 0 {
            (
                [1536, 64, bands * 1536, 1],
                [512, 64, bands * 512, 1],
                [512, 64, bands * 512, 1],
            )
        } else {
            (
                [bands * 1536, 64, 1536, 1],
                [bands * 512, 64, 512, 1],
                [bands * 512, 64, 512, 1],
            )
        };
        Ok(Self {
            axis,
            b,
            h: 8,
            s,
            d: 64,
            dtype: 0,
            q,
            k,
            v: q,
            o,
            scale_bits: ATTN_SCALE.to_bits(),
        })
    }
}
struct Plan {
    graph: *mut c_void,
    workspace: usize,
    heuristic: i32,
}
pub struct CudnnSdpa {
    ctx: Arc<CudaContext>,
    handle: *mut c_void,
    set_stream: FnSetStream,
    build: FnBuild,
    exec: FnExec,
    free_graph: FnFreeGraph,
    destroy: FnDestroy,
    err: FnErr,
    version: i32,
    plans: HashMap<PlanKey, Plan>,
    workspace: Option<DeviceBuffer<u8>>,
    stream: Option<usize>,
    #[cfg(test)]
    fail_workspace_allocation: bool,
    // Keep libraries alive until graphs, handle and workspace have been freed.
    _lib: Library,
    _nvrtc: Vec<Library>,
}

fn first_existing(paths: &[String]) -> Option<String> {
    paths
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .cloned()
}

fn cstr_to_string(p: *const c_char) -> String {
    if p.is_null() {
        return "(null)".into();
    }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
}
impl CudnnSdpa {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<CudnnSdpa, String> {
        let cudnn_dir = std::env::var("LBRR_CUDNN_DIR").unwrap_or_else(|_| {
            if cfg!(windows) {
                "D:/Projects/lbrr-win-libs/cudnn/bin".into()
            } else {
                "/data/dsh/lbrr-venv/lib/python3.12/site-packages/nvidia/cudnn/lib".into()
            }
        });
        let nvrtc_dir = std::env::var("LBRR_NVRTC_DIR").unwrap_or_else(|_| {
            if cfg!(windows) {
                // CUDA 13.4 ships nvrtc64_130_0.dll + builtins in bin/x64.
                std::env::var("CUDA_HOME")
                    .map(|h| format!("{h}/bin/x64"))
                    .unwrap_or_default()
            } else {
                "/data/dsh/lbrr-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib".into()
            }
        });
        let wrap_env = if cfg!(windows) {
            std::env::var("LBRR_SDPA_WRAP")
                .unwrap_or_else(|_| "D:/Projects/lbrr-win-libs/cudnn_sdpa_wrap.dll".into())
        } else {
            std::env::var("LBRR_SDPA_WRAP")
                .unwrap_or_else(|_| "/data/dsh/libcudnn_sdpa_wrap.so".into())
        };
        let wrap_fallback = if cfg!(windows) {
            "cudnn_sdpa_wrap.dll"
        } else {
            "libcudnn_sdpa_wrap.so"
        };
        let wrap = first_existing(&[wrap_env.clone(), wrap_fallback.into()])
            .ok_or_else(|| format!("sdpa wrapper not found: {}", wrap_env))?;
        if cfg!(windows) {
            // LoadLibrary 解析依赖 DLL 时只搜应用目录/System32/PATH，不含
            // 目标 DLL 所在目录（Linux 由 RPATH+RTLD_GLOBAL 处理）。把 cudnn
            // bin 目录注入进程 PATH，子库（ops→graph 等）才能互相解析。
            let cur = std::env::var("PATH").unwrap_or_default();
            let mut extra: Vec<&str> = Vec::new();
            if !cur
                .split(';')
                .any(|p| p.trim_end_matches('\\').eq_ignore_ascii_case(&cudnn_dir))
            {
                extra.push(&cudnn_dir);
            }
            // cudart/nvrtc 裸名解析也需要 toolkit bin/x64 在 PATH 上。
            if !nvrtc_dir.is_empty()
                && !cur
                    .split(';')
                    .any(|p| p.trim_end_matches('\\').eq_ignore_ascii_case(&nvrtc_dir))
            {
                extra.push(&nvrtc_dir);
            }
            if !extra.is_empty() {
                let mut full = extra.join(";");
                full.push(';');
                full.push_str(&cur);
                // SAFETY: 单线程初始化路径（首次 CudnnSdpa::load），无并发读者。
                unsafe { std::env::set_var("PATH", full) };
            }
        }
        // The sm86 sdpa engine runtime-compiles kernels via libnvrtc.so.12;
        // pre-register it (plus its builtins dependency) under absolute paths
        // so cudnn internal dlopen("libnvrtc.so.12") hits a loaded SONAME.
        let nvrtc_names: &[&str] = if cfg!(windows) {
            &["nvrtc-builtins64_134.dll", "nvrtc64_130_0.dll"]
        } else {
            &["libnvrtc-builtins.so.12", "libnvrtc.so.12"]
        };
        let mut nvrtc_libs = Vec::new();
        for name in nvrtc_names {
            let cand = format!("{nvrtc_dir}/{name}");
            if std::path::Path::new(&cand).exists() {
                match unsafe { Library::new(&cand) } {
                    Ok(l) => nvrtc_libs.push(l),
                    Err(e) => return Err(format!("dlopen {}: {}", cand, e)),
                }
            }
        }
        unsafe {
            let lib = Library::new(&wrap).map_err(|e| format!("dlopen {}: {}", wrap, e))?;
            let init: libloading::Symbol<FnInit> = lib
                .get(b"wrap_init")
                .map_err(|e| format!("sym wrap_init: {}", e))?;
            let init_dir =
                std::ffi::CString::new(cudnn_dir.clone()).map_err(|e| format!("cstr: {}", e))?;
            let rc = init(init_dir.as_ptr());
            if rc != 0 {
                let errfn: libloading::Symbol<FnErr> = lib.get(b"wrap_last_error").unwrap();
                return Err(format!(
                    "wrap_init: {} (rc={})",
                    cstr_to_string(errfn()),
                    rc
                ));
            }
            let create: libloading::Symbol<FnCreate> = lib
                .get(b"wrap_create")
                .map_err(|e| format!("sym wrap_create: {}", e))?;
            // The handle binds to the current context; make that ours.
            ctx.bind_to_thread()
                .map_err(|e| format!("bind ctx: {}", e))?;
            // Bind every symbol before creating a handle so lookup failures cannot leak it.
            let set_stream: FnSetStream =
                *lib.get(b"wrap_set_stream").map_err(|e| e.to_string())?;
            let build: FnBuild = *lib.get(b"wrap_sdpa_build").map_err(|e| e.to_string())?;
            let exec: FnExec = *lib.get(b"wrap_sdpa_exec").map_err(|e| e.to_string())?;
            let free_graph: FnFreeGraph = *lib.get(b"wrap_sdpa_free").map_err(|e| e.to_string())?;
            let destroy: FnDestroy = *lib.get(b"wrap_destroy").map_err(|e| e.to_string())?;
            let err: FnErr = *lib.get(b"wrap_last_error").map_err(|e| e.to_string())?;
            let version: FnVersion = *lib.get(b"wrap_version").map_err(|e| e.to_string())?;
            let version = version();
            let mut handle = std::ptr::null_mut();
            let rc = create(&mut handle);
            if rc != 0 {
                if !handle.is_null() {
                    destroy(handle);
                }
                return Err(format!("wrap_create rc={rc}"));
            }
            Ok(CudnnSdpa {
                ctx: ctx.clone(),
                handle,
                set_stream,
                build,
                exec,
                free_graph,
                destroy,
                err,
                version,
                plans: HashMap::new(),
                workspace: None,
                stream: None,
                #[cfg(test)]
                fail_workspace_allocation: false,
                _lib: lib,
                _nvrtc: nvrtc_libs,
            })
        }
    }

    fn last_error(&self) -> String {
        unsafe { cstr_to_string((self.err)()) }
    }

    fn allocate_workspace(
        &self,
        stream: &Arc<CudaStream>,
        bytes: usize,
    ) -> Result<DeviceBuffer<u8>, String> {
        #[cfg(test)]
        if self.fail_workspace_allocation {
            return Err("injected allocation failure".into());
        }
        DeviceBuffer::zeroed(stream, bytes).map_err(|e| e.to_string())
    }
    pub fn version(&self) -> i32 {
        self.version
    }
    pub fn workspace_bytes(&self) -> usize {
        self.workspace.as_ref().map_or(0, |w| w.len())
    }
    pub fn plans_json(&self) -> serde_json::Value {
        let mut plans: Vec<_> = self
            .plans
            .iter()
            .map(|(k, p)| (k.axis, k.b, k.s, p.workspace, p.heuristic))
            .collect();
        plans.sort();
        serde_json::json!(plans)
    }
    /// Prepare without touching Q/K/V. One instance owns one context and one stream.
    pub fn prepare(
        &mut self,
        stream: &Arc<CudaStream>,
        axis: usize,
        bands: i64,
        frames: i64,
    ) -> Result<(), String> {
        self.ctx.bind_to_thread().map_err(|e| e.to_string())?;
        if !Arc::ptr_eq(&self.ctx, stream.context()) {
            return Err("cuDNN stream belongs to another context".into());
        }
        let raw = stream.cu_stream() as usize;
        if self.stream.is_some_and(|s| s != raw) {
            return Err("cuDNN instance is bound to another stream".into());
        }
        let rc = unsafe { (self.set_stream)(self.handle, raw as *mut c_void) };
        if rc != 0 {
            return Err(format!("cudnnSetStream rc={rc}: {}", self.last_error()));
        }
        self.stream = Some(raw);
        let key = PlanKey::new(axis, bands, frames)?;
        if self.plans.contains_key(&key) {
            return Ok(());
        }
        let mut errors = Vec::new();
        for heuristic in [0, 1, 2] {
            let mut graph = std::ptr::null_mut();
            let mut ws = 0i64;
            let rc = unsafe {
                (self.build)(
                    self.handle,
                    key.b,
                    key.h,
                    key.s,
                    key.d,
                    key.q.as_ptr(),
                    key.k.as_ptr(),
                    key.v.as_ptr(),
                    key.o.as_ptr(),
                    f32::from_bits(key.scale_bits),
                    heuristic,
                    key.dtype,
                    &mut graph,
                    &mut ws,
                )
            };
            if rc != 0 || graph.is_null() {
                if !graph.is_null() {
                    unsafe { (self.free_graph)(graph) };
                }
                errors.push(format!(
                    "heuristic {heuristic}: rc={rc} {}",
                    self.last_error()
                ));
                continue;
            }
            if ws < 0 || ws as u64 > 64 * 1024 * 1024 {
                unsafe { (self.free_graph)(graph) };
                errors.push(format!(
                    "heuristic {heuristic}: workspace {ws} exceeds 64MiB"
                ));
                continue;
            }
            if ws as usize > self.workspace_bytes() {
                match self.allocate_workspace(stream, ws as usize) {
                    Ok(w) => self.workspace = Some(w),
                    Err(e) => {
                        unsafe { (self.free_graph)(graph) };
                        return Err(format!("SDPA workspace allocation ({ws} bytes): {e}"));
                    }
                }
            }
            self.plans.insert(
                key,
                Plan {
                    graph,
                    workspace: ws as usize,
                    heuristic,
                },
            );
            return Ok(());
        }
        Err(format!(
            "SDPA axis {axis} B={} S={}: {}",
            key.b,
            key.s,
            errors.join("; ")
        ))
    }
    /// Execute only a prebuilt graph; any failure aborts inference, never re-rotates Q.
    pub fn execute(
        &self,
        stream: &Arc<CudaStream>,
        axis: usize,
        bands: i64,
        frames: i64,
        q: u64,
        k: u64,
        v: u64,
        o: u64,
    ) -> Result<(), String> {
        if self.stream != Some(stream.cu_stream() as usize)
            || !Arc::ptr_eq(&self.ctx, stream.context())
        {
            return Err("SDPA execute stream/context mismatch".into());
        }
        self.ctx.bind_to_thread().map_err(|e| e.to_string())?;
        let key = PlanKey::new(axis, bands, frames)?;
        let plan = self.plans.get(&key).ok_or("SDPA execute before prepare")?;
        let workspace = self
            .workspace
            .as_ref()
            .map_or(std::ptr::null_mut(), |w| w.cu_deviceptr() as *mut c_void);
        let rc = unsafe {
            (self.exec)(
                plan.graph,
                q as *mut c_void,
                k as *mut c_void,
                v as *mut c_void,
                o as *mut c_void,
                workspace,
            )
        };
        if rc != 0 {
            return Err(format!(
                "SDPA execute axis={axis} rc={rc}: {}",
                self.last_error()
            ));
        }
        Ok(())
    }
}
impl Drop for CudnnSdpa {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            eprintln!("cuDNN cleanup context: {e}");
            return;
        }
        for (_, p) in self.plans.drain() {
            unsafe { (self.free_graph)(p.graph) };
        }
        if !self.handle.is_null() {
            let rc = unsafe { (self.destroy)(self.handle) };
            if rc != 0 {
                eprintln!("cudnnDestroy rc={rc}");
            }
            self.handle = std::ptr::null_mut();
        }
        self.workspace.take();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_dimensions_keep_axis_strides_distinct() {
        let time = PlanKey::new(0, 62, 62).unwrap();
        let freq = PlanKey::new(1, 62, 62).unwrap();
        assert_eq!((time.b, time.s), (freq.b, freq.s));
        assert_ne!(time, freq);
        assert_ne!(time.q, freq.q);
        let mut plans = HashMap::new();
        plans.insert(time, 0);
        plans.insert(freq, 1);
        assert_eq!(plans.len(), 2);
    }
    #[test]
    fn scale_dtype_and_output_stride_are_part_of_identity() {
        let a = PlanKey::new(0, 62, 259).unwrap();
        let mut b = a.clone();
        b.scale_bits = 0.25f32.to_bits();
        assert_ne!(a, b);
        b = a.clone();
        b.dtype = 1;
        assert_ne!(a, b);
        b = a.clone();
        b.o[2] += 1;
        assert_ne!(a, b);
    }

    use std::cell::RefCell;
    #[derive(Default)]
    struct Fake {
        mode: u8,
        builds: usize,
        graphs: usize,
        handles: usize,
    }
    thread_local! {static FAKE:RefCell<Fake>=RefCell::new(Fake::default());}
    unsafe extern "C" fn fake_stream(_: *mut c_void, _: *mut c_void) -> i32 {
        0
    }
    unsafe extern "C" fn fake_error() -> *const c_char {
        c"simulated SDK failure".as_ptr()
    }
    unsafe extern "C" fn fake_build(
        _: *mut c_void,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        q: *const i64,
        _: *const i64,
        _: *const i64,
        _: *const i64,
        _: f32,
        heur: i32,
        _: i32,
        graph: *mut *mut c_void,
        ws: *mut i64,
    ) -> i32 {
        FAKE.with(|state| {
            let mut s = state.borrow_mut();
            s.builds += 1;
            if s.mode == 1 && unsafe { *q } != 1536 || s.mode == 3 && heur != 2 {
                return -1;
            }
            unsafe {
                *graph = Box::into_raw(Box::new(1u8)).cast();
                *ws = if s.mode == 2 { (64 << 20) + 1 } else { 4096 };
            }
            s.graphs += 1;
            0
        })
    }
    unsafe extern "C" fn fake_exec(
        _: *mut c_void,
        _: *mut c_void,
        _: *mut c_void,
        _: *mut c_void,
        _: *mut c_void,
        _: *mut c_void,
    ) -> i32 {
        -4
    }
    unsafe extern "C" fn fake_free(p: *mut c_void) {
        unsafe {
            drop(Box::from_raw(p.cast::<u8>()));
        }
        FAKE.with(|s| s.borrow_mut().graphs -= 1);
    }
    unsafe extern "C" fn fake_destroy(p: *mut c_void) -> i32 {
        unsafe {
            drop(Box::from_raw(p.cast::<u8>()));
        }
        FAKE.with(|s| s.borrow_mut().handles -= 1);
        0
    }
    fn fake(mode: u8) -> Option<(Arc<CudaContext>, Arc<CudaStream>, CudnnSdpa)> {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("SKIP cuDNN GPU lifecycle test: {e}");
                return None;
            }
        };
        let stream = ctx.new_stream().unwrap();
        FAKE.with(|s| {
            *s.borrow_mut() = Fake {
                mode,
                handles: 1,
                ..Fake::default()
            }
        });
        let lib = unsafe {
            Library::new(if cfg!(windows) {
                "nvcuda.dll"
            } else {
                "libcuda.so.1"
            })
            .unwrap()
        };
        let c = CudnnSdpa {
            ctx: ctx.clone(),
            handle: Box::into_raw(Box::new(0u8)).cast(),
            set_stream: fake_stream,
            build: fake_build,
            exec: fake_exec,
            free_graph: fake_free,
            destroy: fake_destroy,
            err: fake_error,
            version: 0,
            plans: HashMap::new(),
            workspace: None,
            stream: None,
            fail_workspace_allocation: false,
            _lib: lib,
            _nvrtc: Vec::new(),
        };
        Some((ctx, stream, c))
    }
    #[test]
    fn workspace_cache_execution_error_and_drop() {
        let Some((_ctx, stream, mut c)) = fake(0) else {
            return;
        };
        let sentinels = DeviceBuffer::from_host(&stream, &[0x3c003c00u32; 32]).unwrap();
        c.prepare(&stream, 0, 62, 62).unwrap();
        c.prepare(&stream, 1, 62, 62).unwrap();
        c.prepare(&stream, 0, 62, 62).unwrap();
        assert_eq!(c.plans.len(), 2);
        assert_eq!(c.workspace_bytes(), 4096);
        assert_eq!(sentinels.to_host_vec(&stream).unwrap(), [0x3c003c00u32; 32]);
        let p = sentinels.cu_deviceptr();
        assert!(
            c.execute(&stream, 0, 62, 62, p, p, p, p)
                .unwrap_err()
                .contains("simulated SDK failure")
        );
        FAKE.with(|s| assert_eq!(s.borrow().builds, 2));
        drop(c);
        FAKE.with(|s| {
            assert_eq!(s.borrow().graphs, 0);
            assert_eq!(s.borrow().handles, 0);
        });
    }
    #[test]
    fn unsupported_axis_keeps_other_plan_and_stream_mismatch_fails() {
        let Some((ctx, stream, mut c)) = fake(1) else {
            return;
        };
        c.prepare(&stream, 0, 62, 62).unwrap();
        assert!(c.prepare(&stream, 1, 62, 62).is_err());
        assert_eq!(c.plans.len(), 1);
        assert!(c.prepare(&ctx.new_stream().unwrap(), 0, 62, 62).is_err());
        drop(c);
        FAKE.with(|s| assert_eq!(s.borrow().graphs, 0));
    }
    #[test]
    fn oversized_workspace_and_failed_allocation_free_graphs() {
        let Some((_ctx, stream, mut c)) = fake(2) else {
            return;
        };
        assert!(c.prepare(&stream, 0, 62, 259).is_err());
        assert_eq!(c.workspace_bytes(), 0);
        assert!(c.plans.is_empty());
        FAKE.with(|s| {
            assert_eq!(s.borrow().builds, 3);
            assert_eq!(s.borrow().graphs, 0);
        });
        drop(c);
        let Some((_ctx, stream, mut c)) = fake(0) else {
            return;
        };
        c.fail_workspace_allocation = true;
        assert!(
            c.prepare(&stream, 0, 62, 259)
                .unwrap_err()
                .contains("injected allocation failure")
        );
        assert!(c.plans.is_empty());
        FAKE.with(|s| assert_eq!(s.borrow().graphs, 0));
        drop(c);
    }
    #[test]
    fn heuristic_fallback_is_bounded_and_cached() {
        let Some((_ctx, stream, mut c)) = fake(3) else {
            return;
        };
        c.prepare(&stream, 0, 62, 259).unwrap();
        assert_eq!(c.plans.values().next().unwrap().heuristic, 2);
        FAKE.with(|s| assert_eq!(s.borrow().builds, 3));
        drop(c);
    }
}
