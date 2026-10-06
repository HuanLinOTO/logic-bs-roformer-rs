use cuda_core::CudaContext;
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};

const ATTN_SCALE: f32 = 0.125;

type FnInit = unsafe extern "C" fn(*const c_char) -> c_int;
type FnCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type FnSetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type FnBuild = unsafe extern "C" fn(
    *mut c_void,            // handle
    i64, i64, i64, i64,     // b, heads, s, d
    *const i64, *const i64, *const i64, *const i64, // q/k/v/o strides (elements)
    f32,                    // attn scale
    c_int, c_int,           // heur mode, io dtype
    *mut *mut c_void,       // graph out
    *mut i64,               // workspace size out
) -> c_int;
type FnExec = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void) -> c_int;
type FnFreeGraph = unsafe extern "C" fn(*mut c_void);
type FnErr = unsafe extern "C" fn() -> *const c_char;

unsafe impl Send for CudnnSdpa {}

pub struct CudnnSdpa {
    _lib: Library,
    _nvrtc: Vec<Library>,
    handle: *mut c_void,
    set_stream: FnSetStream,
    build: FnBuild,
    exec: FnExec,
    free_graph: FnFreeGraph,
    err: FnErr,
    plans: HashMap<(i64, i64), *mut c_void>,
}

fn first_existing(paths: &[String]) -> Option<String> {
    paths.iter().find(|p| std::path::Path::new(p).exists()).cloned()
}

fn cstr_to_string(p: *const c_char) -> String {
    if p.is_null() { return "(null)".into(); }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
}
impl CudnnSdpa {
    pub fn load(ctx: &CudaContext) -> Result<CudnnSdpa, String> {
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
        let wrap_fallback = if cfg!(windows) { "cudnn_sdpa_wrap.dll" } else { "libcudnn_sdpa_wrap.so" };
        let wrap = first_existing(&[wrap_env.clone(), wrap_fallback.into()])
            .ok_or_else(|| format!("sdpa wrapper not found: {}", wrap_env))?;
        if cfg!(windows) {
            // LoadLibrary 解析依赖 DLL 时只搜应用目录/System32/PATH，不含
            // 目标 DLL 所在目录（Linux 由 RPATH+RTLD_GLOBAL 处理）。把 cudnn
            // bin 目录注入进程 PATH，子库（ops→graph 等）才能互相解析。
            let cur = std::env::var("PATH").unwrap_or_default();
            let mut extra: Vec<&str> = Vec::new();
            if !cur.split(';').any(|p| p.trim_end_matches('\\').eq_ignore_ascii_case(&cudnn_dir)) {
                extra.push(&cudnn_dir);
            }
            // cudart/nvrtc 裸名解析也需要 toolkit bin/x64 在 PATH 上。
            if !nvrtc_dir.is_empty()
                && !cur.split(';').any(|p| p.trim_end_matches('\\').eq_ignore_ascii_case(&nvrtc_dir))
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
            let init: libloading::Symbol<FnInit> =
                lib.get(b"wrap_init").map_err(|e| format!("sym wrap_init: {}", e))?;
            let init_dir = std::ffi::CString::new(cudnn_dir.clone()).map_err(|e| format!("cstr: {}", e))?;
            let rc = init(init_dir.as_ptr());
            if rc != 0 {
                let errfn: libloading::Symbol<FnErr> = lib.get(b"wrap_last_error").unwrap();
                return Err(format!("wrap_init: {} (rc={})", cstr_to_string(errfn()), rc));
            }
            let create: libloading::Symbol<FnCreate> =
                lib.get(b"wrap_create").map_err(|e| format!("sym wrap_create: {}", e))?;
            // The handle binds to the current context; make that ours.
            ctx.bind_to_thread().map_err(|e| format!("bind ctx: {}", e))?;
            let mut handle: *mut c_void = std::ptr::null_mut();
            let rc = create(&mut handle);
            if rc != 0 { return Err(format!("wrap_create rc={}", rc)); }
            let set_stream: FnSetStream = *lib.get(b"wrap_set_stream").map_err(|e| format!("sym wrap_set_stream: {}", e))?;
            let build: FnBuild = *lib.get(b"wrap_sdpa_build").map_err(|e| format!("sym wrap_sdpa_build: {}", e))?;
            let exec: FnExec = *lib.get(b"wrap_sdpa_exec").map_err(|e| format!("sym wrap_sdpa_exec: {}", e))?;
            let free_graph: FnFreeGraph = *lib.get(b"wrap_sdpa_free").map_err(|e| format!("sym wrap_sdpa_free: {}", e))?;
            let err: FnErr = *lib.get(b"wrap_last_error").map_err(|e| format!("sym wrap_last_error: {}", e))?;
            Ok(CudnnSdpa { _lib: lib, _nvrtc: nvrtc_libs, handle, set_stream, build, exec, free_graph, err, plans: HashMap::new() })
        }
    }

    fn last_error(&self) -> String {
        unsafe { cstr_to_string((self.err)()) }
    }

    /// Run one attention axis. q/v point into qkv16 (v already offset by
    /// 1024 fp16 elements = 2048 bytes), k into the roped k16r buffer, o
    /// into scaled16. All folded token-major layouts.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        &mut self,
        stream: usize,
        axis: usize,
        bands: i64,
        seq: i64,
        q: u64,
        k: u64,
        v: u64,
        o: u64,
    ) -> Result<(), String> {
        let (b, s) = if axis == 0 { (bands, seq) } else { (seq, bands) };
        // element strides (b, h, s, d)
        let (q_str, k_str, o_str) = if axis == 0 {
            ([1536, 64, bands * 1536, 1], [512, 64, bands * 512, 1], [512, 64, bands * 512, 1])
        } else {
            ([bands * 1536, 64, 1536, 1], [bands * 512, 64, 512, 1], [bands * 512, 64, 512, 1])
        };
        let v_str = q_str;
        let plan = match self.plans.get(&(b, s)) {
            Some(p) => *p,
            None => {
                let mut graph: *mut c_void = std::ptr::null_mut();
                let mut ws: i64 = 0;
                let rc = unsafe { (self.build)(self.handle, b, 8, s, 64,
                    q_str.as_ptr(), k_str.as_ptr(), v_str.as_ptr(), o_str.as_ptr(),
                    ATTN_SCALE, 0, 0, &mut graph, &mut ws) };
                if rc != 0 {
                    return Err(format!("sdpa build (b={},s={}): {} (rc={})", b, s, self.last_error(), rc));
                }
                // ws==0 for every shape measured on the 3080; if that ever
                // changes the caller must provide a workspace buffer.
                assert!(ws == 0, "cudnn sdpa wants {}B workspace; buffer not wired", ws);
                self.plans.insert((b, s), graph);
                graph
            }
        };
        unsafe {
            (self.set_stream)(self.handle, stream as *mut c_void);
            let rc = (self.exec)(plan, q as *mut c_void, k as *mut c_void,
                v as *mut c_void, o as *mut c_void, std::ptr::null_mut());
            if rc != 0 {
                return Err(format!("sdpa exec: {} (rc={})", self.last_error(), rc));
            }
        }
        Ok(())
    }
}

impl Drop for CudnnSdpa {
    fn drop(&mut self) {
        for p in self.plans.values() {
            unsafe { (self.free_graph)(*p) };
        }
    }
}
