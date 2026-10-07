//! STFT/ISTFT orchestration on top of the cufft bindings.
//!
//! Fixed shapes for this model: n_fft 2048 / hop 512 / hann 2048 / center
//! reflect. One forward plan covers both channels (batch = 2*T frames).

use crate::cufft::{Cufft, CufftPlan};
use cuda_core::{CudaContext, DeviceBuffer, Stream};
use std::sync::Arc;

pub struct Stft {
    // Drop the plan before its owning library (fields drop in declaration order).
    fwd: CufftPlan<'static>,
    _fft: Arc<Cufft>,
    window: Vec<f32>,
}

pub const N_FFT: usize = 2048;
pub const HOP: usize = 512;
pub const FREQ_BINS: usize = N_FFT / 2 + 1; // 1025

/// Periodic hann window computed in f64 then cast — bit-compatible with
/// torch.hann_window(win_length) defaults.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / n as f64).cos())) as f32)
        .collect()
}

/// Number of center-mode STFT frames for a signal of len samples.
pub fn num_frames(len: usize) -> usize {
    len / HOP + 1
}

impl Stft {
    /// Plan for exactly 2*frames forward transforms (both channels).
    pub fn new(fft: Arc<Cufft>, frames: usize) -> Result<Self, String> {
        // The plan only stores function pointers copied out of the Cufft
        // vtable; _fft above keeps the library alive for exactly as long
        // (fields drop in declaration order).
        let fft_ref: &Cufft = &fft;
        let fwd = unsafe {
            let stat: &'static Cufft = std::mem::transmute(fft_ref);
            stat.plan(N_FFT, 2 * frames, true)?
        };
        Ok(Stft { _fft: fft, fwd, window: hann_window(N_FFT) })
    }

    pub fn set_stream(&self, stream: &cuda_core::CudaStream) -> Result<(), String> {
        self.fwd.set_stream(stream)
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    pub fn exec_fwd(&self, frames_dev: &DeviceBuffer<f32>, spec: &mut DeviceBuffer<f32>) -> Result<(), String> {
        self.fwd.exec_r2c(frames_dev.cu_deviceptr(), spec.cu_deviceptr())
    }
}