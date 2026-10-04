//! lbrr — BS-RoFormer six-stem music separation, pure Rust + cuda-oxide.
//!
//! Phase 0 scaffold: verifies the vendor/ path-dependency toolchain end-to-end
//! with a trivial vector-add kernel on the local GPU.

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
        if let Some(out) = c.get_mut(idx) {
            let i = idx.get();
            *out = a[i] + b[i];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
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
    println!("lbrr scaffold: vecadd N={N} max_err={max_err:e}");
    assert!(max_err < 1e-6, "vecadd mismatch");
    println!("OK");
}
