//! Minimal runtime (dlopen) bindings to NVIDIA cuBLAS.
//! Provides SGEMM (fp32 matrix multiply) via the highly optimized cuBLAS
//! library, replacing our custom SIMT GEMM for ~30x speedup.

use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use cuda_core::sys::CUdeviceptr;

#[derive(Debug)]
pub struct Cublas {
    _lib: Library,
    handle: *mut c_void,
    sgemm: unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int, c_int, c_int,
        *const f32, *const f32, c_int, *const f32, c_int,
        *const f32, *mut f32, c_int) -> c_int,
    set_stream: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
}

// CUBLAS_OP_N = 0, CUBLAS_OP_T = 1
const OP_N: c_int = 0;
const OP_T: c_int = 1;

// SAFETY: cuBLAS handle is thread-safe (cuBLAS docs) and we're single-threaded.
unsafe impl Send for Cublas {}
unsafe impl Sync for Cublas {}

impl Cublas {
    pub fn load() -> Result<Cublas, String> {
        let mut candidates: Vec<String> = Vec::new();
        for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"] {
            if let Ok(p) = std::env::var(var) {
                for sub in ["lib64", "lib"] {
                    for name in ["libcublas.so", "libcublas.so.12", "libcublas.so.11"] {
                        candidates.push(format!("{p}/{sub}/{name}"));
                    }
                }
            }
        }
        candidates.push("libcublas.so.12".into());
        candidates.push("libcublas.so.11".into());
        candidates.push("libcublas.so".into());

        let mut last = String::new();
        for cand in &candidates {
            match unsafe { Library::new(cand) } {
                Ok(lib) => return Ok(unsafe { Self::bind(lib) }),
                Err(e) => last = format!("{cand}: {e}"),
            }
        }
        Err(format!("libcublas not found; last: {last}"))
    }

    unsafe fn bind(lib: Library) -> Cublas {
        unsafe {
            let create: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> c_int> =
                lib.get(b"cublasCreate_v2").expect("cublasCreate_v2");
            let mut handle: *mut c_void = std::ptr::null_mut();
            let rc = create(&mut handle);
            assert_eq!(rc, 0, "cublasCreate failed: {rc}");
            let sgemm = *lib.get(b"cublasSgemm").expect("cublasSgemm symbol");
            let set_stream = *lib.get(b"cublasSetStream_v2").expect("cublasSetStream_v2");
            let _ = rc;
            Cublas { _lib: lib, handle, sgemm, set_stream }
        }
    }

    /// Set the CUDA stream for subsequent operations.
    /// stream_ptr: raw CUstream pointer (from cuda_core).
    pub fn set_stream_raw(&self, stream_ptr: *mut c_void) -> Result<(), String> {
        let rc = unsafe { (self.set_stream)(self.handle, stream_ptr) };
        if rc != 0 { Err(format!("cublasSetStream: {rc}")) } else { Ok(()) }
    }

    /// Y[M,N] = X[M,K] · W[N,K]^T + bias
    /// X: row-major (M,K), W: row-major (N,K), Y: row-major (M,N)
    ///
    /// cuBLAS uses column-major. For row-major C = A·B^T:
    /// In col-major: C^T = B · A^T, i.e. C_col = op(B)·op(A) where
    /// we compute C_col(N,M) = W_col(K,N)^T · X_col(K,M)
    /// which is cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
    ///   alpha, W, K, X, K, beta, Y, N)
    ///
    /// Note: W stored as (N,K) row-major = (K,N) col-major.
    ///       X stored as (M,K) row-major = (K,M) col-major.
    ///       Y stored as (M,N) row-major = (N,M) col-major.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_nt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x_ptr: CUdeviceptr,
        w_ptr: CUdeviceptr,
        y_ptr: CUdeviceptr,
        alpha: f32,
        beta: f32,
    ) -> Result<(), String> {
        // C_col(N,M) = op(W_col)·op(X_col)
        // W_col is (K,N), op=T gives (N,K)
        // X_col is (K,M), op=N gives (K,M)
        // Result: (N,K)·(K,M) = (N,M) col-major = (M,N) row-major ✓
        let rc = unsafe {
            (self.sgemm)(
                self.handle,
                OP_T,     // transa: W_col(K,N) → (N,K)
                OP_N,     // transb: X_col(K,M) stays (K,M)
                n as c_int, // rows of output (col-major)
                m as c_int, // cols of output (col-major)
                k as c_int, // inner dim
                &alpha,
                w_ptr as *const f32,
                k as c_int, // lda: leading dim of W_col = K
                x_ptr as *const f32,
                k as c_int, // ldb: leading dim of X_col = K
                &beta,
                y_ptr as *mut f32,
                n as c_int, // ldc: leading dim of Y_col = N
            )
        };
        if rc != 0 { Err(format!("cublasSgemm: error {rc}")) } else { Ok(()) }
    }
}
