//! Minimal runtime (dlopen) bindings to NVIDIA cuFFT.
//!
//! Loads `libcufft.so` lazily and exposes just the plan/exec API this model
//! needs: batched 1-D R2C (STFT) and C2R (ISTFT) over fp32, with
//! user-provided device buffers. Plans are RAII (destroyed on drop) and are
//! meant to be created once and reused for the fixed chunk shapes.

use cuda_core::sys::CUdeviceptr;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void};
use std::ptr;

#[derive(Debug)]
pub struct Cufft {
    _lib: Library,
    plan_many: unsafe extern "C" fn(*mut c_int, c_int, *const c_int, *const c_int, c_int, c_int, *const c_int, c_int, c_int, c_int, c_int) -> c_int,
    exec_r2c: unsafe extern "C" fn(c_int, *mut c_void, *mut c_void) -> c_int,
    exec_c2r: unsafe extern "C" fn(c_int, *mut c_void, *mut c_void) -> c_int,
    destroy: unsafe extern "C" fn(c_int) -> c_int,
}

const CUFFT_R2C: c_int = 0x2A;
const CUFFT_C2R: c_int = 0x2C;

fn result_str(code: c_int) -> &'static str {
    match code {
        0 => "SUCCESS",
        1 => "INVALID_PLAN",
        2 => "ALLOC_FAILED",
        3 => "INVALID_TYPE",
        4 => "INVALID_VALUE",
        5 => "INTERNAL_ERROR",
        6 => "EXEC_FAILED",
        7 => "SETUP_FAILED",
        8 => "INVALID_SIZE",
        9 => "UNALIGNED_DATA",
        10 => "INCOMPLETE_PARAMETER_LIST",
        11 => "INVALID_DEVICE",
        12 => "PARSE_ERROR",
        13 => "NO_WORKSPACE",
        14 => "NOT_IMPLEMENTED",
        15 => "LICENSE_ERROR",
        16 => "NOT_SUPPORTED",
        _ => "UNKNOWN",
    }
}

impl Cufft {
    /// dlopen libcufft: explicit toolkit paths first, then the loader.
    pub fn load() -> Result<Cufft, String> {
        let mut candidates: Vec<String> = Vec::new();
        let mut try_paths: Vec<std::path::PathBuf> = Vec::new();
        for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"] {
            if let Ok(p) = std::env::var(var) {
                try_paths.push(std::path::PathBuf::from(p));
            }
        }
        try_paths.push(std::path::PathBuf::from("/usr/local/cuda"));
        for p in &try_paths {
            for sub in ["lib64", "lib"] {
                // toolkit installs may ship only the versioned runtime (no
                // dev symlink), so probe versioned names in the explicit
                // directories too.
                for name in ["libcufft.so", "libcufft.so.12", "libcufft.so.11", "libcufft.so.10"] {
                    candidates.push(format!("{}/{}/{name}", p.display(), sub));
                }
            }
        }
        candidates.push("libcufft.so.12".into());
        candidates.push("libcufft.so.11".into());
        candidates.push("libcufft.so.10".into());
        candidates.push("libcufft.so".into());

        let mut last = String::new();
        for cand in &candidates {
            match unsafe { Library::new(cand) } {
                Ok(lib) => return Ok(unsafe { Self::bind(lib) }),
                Err(e) => last = format!("{cand}: {e}"),
            }
        }
        Err(format!("libcufft not found; last error: {last}"))
    }

    unsafe fn bind(lib: Library) -> Cufft {
        unsafe {
            let plan_many = *lib
                .get(b"cufftPlanMany")
                .expect("cufftPlanMany symbol");
            let exec_r2c = *lib.get(b"cufftExecR2C").expect("cufftExecR2C symbol");
            let exec_c2r = *lib.get(b"cufftExecC2R").expect("cufftExecC2R symbol");
            let destroy = *lib.get(b"cufftDestroy").expect("cufftDestroy symbol");
            Cufft { _lib: lib, plan_many, exec_r2c, exec_c2r, destroy }
        }
    }

    /// Batched 1-D real FFT plan. `n` points per transform, `batch`
    /// transforms, contiguous layouts: R2C input dist n / output dist n/2+1,
    /// C2R input dist n/2+1 / output dist n.
    pub fn plan(&self, n: usize, batch: usize, forward: bool) -> Result<CufftPlan, String> {
        let mut handle: c_int = 0;
        let dims = [n as c_int];
        let (idist, odist) = if forward {
            (n as c_int, (n / 2 + 1) as c_int)
        } else {
            ((n / 2 + 1) as c_int, n as c_int)
        };
        let rc = unsafe {
            (self.plan_many)(
                &mut handle,
                1,
                dims.as_ptr(),
                ptr::null(),
                1,
                idist,
                ptr::null(),
                1,
                odist,
                if forward { CUFFT_R2C } else { CUFFT_C2R },
                batch as c_int,
            )
        };
        if rc != 0 {
            return Err(format!("cufftPlanMany: {}", result_str(rc)));
        }
        Ok(CufftPlan { handle, cufft: self })
    }
}

pub struct CufftPlan<'a> {
    handle: c_int,
    cufft: &'a Cufft,
}

impl CufftPlan<'_> {
    /// R2C: input `n * batch` reals, output `(n/2+1) * batch` complex
    /// (interleaved f32 pairs).
    pub fn exec_r2c(&self, input: CUdeviceptr, output: CUdeviceptr) -> Result<(), String> {
        let rc = unsafe {
            (self.cufft.exec_r2c)(
                self.handle,
                input as *mut c_void,
                output as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("cufftExecR2C: {}", result_str(rc)));
        }
        Ok(())
    }

    /// C2R: input `(n/2+1) * batch` complex, output `n * batch` reals
    /// (unnormalized: multiply by 1/n afterwards).
    pub fn exec_c2r(&self, input: CUdeviceptr, output: CUdeviceptr) -> Result<(), String> {
        let rc = unsafe {
            (self.cufft.exec_c2r)(
                self.handle,
                input as *mut c_void,
                output as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("cufftExecC2R: {}", result_str(rc)));
        }
        Ok(())
    }
}

impl Drop for CufftPlan<'_> {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe { (self.cufft.destroy)(self.handle) };
        }
    }
}

// Keep unused-symbol lints quiet for the c_char import retained for clarity.
#[allow(dead_code)]
fn _t(_x: Option<c_char>) {}
